"""
stream_processor.py
--------------------
Incremental (streaming) acquisition and processing of the sensor CSV
produced by `data_generator.py`. Implements a `StreamProcessor` class that:

  1. Acquires new records incrementally (tails the CSV file, reading only
     rows that have arrived since the last check -- it never re-reads the
     whole file), so it behaves correctly even while the generator is still
     writing to it.
  2. Cleans / validates each incoming record (type coercion, range checks,
     duplicate/missing-value handling) with Pandas.
  3. Maintains a per-sensor rolling window (a `collections.deque`) used to
     compute real-time indicators with NumPy/Pandas:
        - rolling mean (moving average) of temperature
        - rolling standard deviation of temperature
        - rolling min/max of CO2
        - event rate (records per second) per sensor
        - global event count
  4. Runs a rule-based anomaly detector over each new record plus the
     rolling context.
  5. Persists processed, enriched records to an output CSV/SQLite table for
     later, reproducible analysis.

This module has no dependency on how the data was produced -- it would work
identically against a real API/message-queue source, as long as records are
appended to (or otherwise pushed into) something this class can poll.
"""

from __future__ import annotations

import collections
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Deque, Dict, List, Optional

import numpy as np
import pandas as pd

FIELDNAMES = ["timestamp", "sensor_id", "temperature", "humidity", "co2"]

# ---- Rule-based thresholds (domain knowledge for a classroom/lab) ----------
TEMP_HIGH_C = 27.0          # sustained high temperature -> possible HVAC failure
CO2_HIGH_PPM = 1000.0       # poor ventilation / overcrowding threshold (ASHRAE-style guidance)
ZSCORE_THRESHOLD = 3.0      # statistical outlier threshold on rolling z-score
FROZEN_REPEAT_COUNT = 5     # N identical consecutive readings => sensor likely stuck
MIN_HISTORY = 10            # readings of history required before the z-score rule activates


@dataclass
class SensorState:
    """Rolling state kept per sensor for incremental computation."""
    window: Deque[dict] = field(default_factory=lambda: collections.deque(maxlen=30))
    count: int = 0
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    last_values: Deque[tuple] = field(default_factory=lambda: collections.deque(maxlen=FROZEN_REPEAT_COUNT))


class StreamProcessor:
    def __init__(self, source_csv: str, output_csv: str, sqlite_path: Optional[str] = None,
                 window_size: int = 30, on_record: Optional[Callable[[dict], None]] = None):
        self.source_csv = source_csv
        self.output_csv = output_csv
        self.sqlite_path = sqlite_path
        self.window_size = window_size
        self.on_record = on_record  # optional callback, e.g. for live console/plot updates

        self._bytes_read = 0          # incremental-read cursor into source_csv
        self._header_seen = False
        self.states: Dict[str, SensorState] = collections.defaultdict(
            lambda: SensorState(window=collections.deque(maxlen=self.window_size))
        )
        self.total_count = 0
        self.anomaly_count = 0
        self._results: List[dict] = []
        self._start_time = time.time()

        self._init_output()

    # ------------------------------------------------------------------ #
    # Incremental acquisition
    # ------------------------------------------------------------------ #
    def _init_output(self):
        os.makedirs(os.path.dirname(self.output_csv) or ".", exist_ok=True)
        if not os.path.exists(self.output_csv):
            pd.DataFrame(columns=FIELDNAMES + [
                "rolling_avg_temp", "rolling_std_temp", "rolling_min_co2",
                "rolling_max_co2", "sensor_rate_hz", "is_anomaly", "anomaly_reason"
            ]).to_csv(self.output_csv, index=False)

        if self.sqlite_path:
            conn = sqlite3.connect(self.sqlite_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS readings (
                    timestamp TEXT, sensor_id TEXT, temperature REAL, humidity REAL,
                    co2 REAL, rolling_avg_temp REAL, rolling_std_temp REAL,
                    rolling_min_co2 REAL, rolling_max_co2 REAL, sensor_rate_hz REAL,
                    is_anomaly INTEGER, anomaly_reason TEXT
                )
            """)
            conn.commit()
            conn.close()

    def poll_new_rows(self) -> pd.DataFrame:
        """Read only the bytes appended to source_csv since the last poll.
        This is the 'incremental acquisition' step: an O(new-data) read, not
        a re-scan of the whole file, mirroring how one would tail a live feed."""
        if not os.path.exists(self.source_csv):
            return pd.DataFrame(columns=FIELDNAMES)

        # Binary mode + byte offsets: only COMPLETE lines (ending in a newline)
        # are consumed, so a row the generator is still writing is left for the
        # next poll instead of being parsed half-written.
        with open(self.source_csv, "rb") as f:
            f.seek(self._bytes_read)
            raw = f.read()
        last_nl = raw.rfind(b"\n")
        if last_nl < 0:
            return pd.DataFrame(columns=FIELDNAMES)
        self._bytes_read += last_nl + 1
        new_text = raw[:last_nl + 1].decode("utf-8")

        if not new_text.strip():
            return pd.DataFrame(columns=FIELDNAMES)

        lines = [l for l in new_text.splitlines() if l.strip()]
        if not self._header_seen:
            # first chunk includes the header written by the generator
            self._header_seen = True
            data_lines = lines[1:] if lines and lines[0].startswith("timestamp") else lines
        else:
            data_lines = lines

        if not data_lines:
            return pd.DataFrame(columns=FIELDNAMES)

        from io import StringIO
        chunk = "timestamp,sensor_id,temperature,humidity,co2\n" + "\n".join(data_lines)
        df = pd.read_csv(StringIO(chunk))
        return self._clean(df)

    # ------------------------------------------------------------------ #
    # Cleaning / transformation (Pandas + NumPy)
    # ------------------------------------------------------------------ #
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.dropna(subset=["sensor_id"]).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])
        for col in ["temperature", "humidity", "co2"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # basic physical-range validation -- drop physically impossible sensor glitches
        df = df[(df["temperature"].between(-20, 80)) & (df["humidity"].between(0, 100)) & (df["co2"] >= 0)]
        df = df.dropna()
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Real-time indicators + anomaly detection (per record)
    # ------------------------------------------------------------------ #
    def _process_record(self, row: pd.Series) -> dict:
        sid = row["sensor_id"]
        st = self.states[sid]
        ts = row["timestamp"].to_pydatetime()

        # Statistics of the window BEFORE this reading: used as the reference
        # for the z-score, so a spike cannot dilute its own baseline.
        prev_temps = np.array([r["temperature"] for r in st.window], dtype=float)
        prev_mean = float(np.mean(prev_temps)) if len(prev_temps) else 0.0
        prev_std = float(np.std(prev_temps)) if len(prev_temps) > 1 else 0.0

        st.window.append({"timestamp": ts, "temperature": row["temperature"], "co2": row["co2"]})
        st.count += 1
        st.first_ts = st.first_ts or ts
        st.last_ts = ts
        st.last_values.append((row["temperature"], row["humidity"], row["co2"]))

        temps = np.array([r["temperature"] for r in st.window], dtype=float)
        co2s = np.array([r["co2"] for r in st.window], dtype=float)

        rolling_avg_temp = float(np.mean(temps))
        rolling_std_temp = float(np.std(temps)) if len(temps) > 1 else 0.0
        rolling_min_co2 = float(np.min(co2s))
        rolling_max_co2 = float(np.max(co2s))

        # Reports per second for this sensor: (count - 1) intervals elapsed
        # between the first and the latest reading; undefined (0) for the first one.
        elapsed = (st.last_ts - st.first_ts).total_seconds()
        sensor_rate_hz = (st.count - 1) / elapsed if st.count > 1 and elapsed > 0 else 0.0

        is_anomaly, reason = self._detect_anomaly(
            row, prev_mean, prev_std, len(prev_temps), st)

        result = {
            "timestamp": ts.isoformat(),
            "sensor_id": sid,
            "temperature": row["temperature"],
            "humidity": row["humidity"],
            "co2": row["co2"],
            "rolling_avg_temp": round(rolling_avg_temp, 3),
            "rolling_std_temp": round(rolling_std_temp, 3),
            "rolling_min_co2": rolling_min_co2,
            "rolling_max_co2": rolling_max_co2,
            "sensor_rate_hz": round(sensor_rate_hz, 3),
            "is_anomaly": int(is_anomaly),
            "anomaly_reason": reason or "",
        }
        return result

    def _detect_anomaly(self, row: pd.Series, ref_mean: float, ref_std: float,
                        ref_n: int, st: SensorState):
        """Rule-based multi-condition anomaly detector. Returns (bool, reason).
        `ref_*` describe the sensor's rolling window *before* this reading."""
        reasons = []

        # Rule 1: absolute thresholds (domain rules)
        if row["temperature"] >= TEMP_HIGH_C:
            reasons.append(f"high_temperature>{TEMP_HIGH_C}C")
        if row["co2"] >= CO2_HIGH_PPM:
            reasons.append(f"high_co2>{CO2_HIGH_PPM}ppm")

        # Rule 2: statistical outlier via z-score against the previous window
        if ref_n >= MIN_HISTORY and ref_std > 1e-6:
            z = (row["temperature"] - ref_mean) / ref_std
            if abs(z) >= ZSCORE_THRESHOLD:
                reasons.append(f"temp_zscore={z:.2f}")

        # Rule 3: stuck/frozen sensor -- identical readings repeated N times
        if len(st.last_values) == FROZEN_REPEAT_COUNT and len(set(st.last_values)) == 1:
            reasons.append("sensor_stuck_repeated_values")

        return (len(reasons) > 0), "; ".join(reasons)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _persist(self, records: List[dict]):
        if not records:
            return
        df = pd.DataFrame(records)
        df.to_csv(self.output_csv, mode="a", header=False, index=False)

        if self.sqlite_path:
            conn = sqlite3.connect(self.sqlite_path)
            df.to_sql("readings", conn, if_exists="append", index=False)
            conn.close()

    # ------------------------------------------------------------------ #
    # Public driving loop
    # ------------------------------------------------------------------ #
    def process_available(self) -> List[dict]:
        """Poll for newly-arrived rows, process each incrementally, persist,
        and return the list of processed records (for live display)."""
        new_df = self.poll_new_rows()
        if new_df.empty:
            return []

        processed = []
        for _, row in new_df.iterrows():
            result = self._process_record(row)
            processed.append(result)
            self.total_count += 1
            self.anomaly_count += result["is_anomaly"]
            if self.on_record:
                self.on_record(result)

        self._persist(processed)
        self._results.extend(processed)
        return processed

    def run_until_idle(self, poll_interval: float = 0.3, max_idle_polls: int = 10):
        """Keep polling the source file until no new data has arrived for
        `max_idle_polls` consecutive polls (i.e., the generator has finished)."""
        idle = 0
        while idle < max_idle_polls:
            processed = self.process_available()
            if processed:
                idle = 0
            else:
                idle += 1
            time.sleep(poll_interval)
        return self._results

    def global_stats(self) -> dict:
        elapsed = max(time.time() - self._start_time, 1e-6)
        return {
            "total_records": self.total_count,
            "total_anomalies": self.anomaly_count,
            "overall_rate_hz": round(self.total_count / elapsed, 3),
            "sensors_seen": list(self.states.keys()),
        }

    def results_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._results)

    def verify_with_pandas(self) -> dict:
        """Cross-check the incremental NumPy indicators against an offline
        pandas `groupby().rolling()` computation over the full result set.
        Both must agree (max absolute difference ~ 0)."""
        df = self.results_dataframe()
        if df.empty:
            return {}
        g = df.groupby("sensor_id")["temperature"]
        pd_mean = g.rolling(self.window_size, min_periods=1).mean().reset_index(level=0, drop=True)
        pd_std = g.rolling(self.window_size, min_periods=1).std(ddof=0).reset_index(level=0, drop=True)
        return {
            "max_abs_diff_rolling_mean": float((pd_mean.sort_index() - df["rolling_avg_temp"]).abs().max()),
            "max_abs_diff_rolling_std": float((pd_std.sort_index().fillna(0) - df["rolling_std_temp"]).abs().max()),
        }
