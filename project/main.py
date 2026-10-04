"""
main.py
-------
Orchestrates the end-to-end demo:

  1. Starts the data generator in a background thread, writing sensor
     readings incrementally to data/stream.csv (simulating a live feed).
  2. Runs the StreamProcessor in the main thread, polling for new rows,
     computing real-time indicators, running anomaly detection, printing a
     live-updating console view, and persisting results to
     output/processed_stream.csv and output/processed_stream.sqlite.
  3. After the stream ends, produces:
       - a summary table (output/summary_table.csv, printed to console)
       - a list of detected anomalies (output/anomalies.csv)
       - a console log of the whole run (output/console_log.txt)
       - a visualization (output/dashboard.png) with temperature trend +
         rolling average, CO2 trend, rolling std and ingestion rate.
       - a cross-check of the NumPy indicators against pandas rolling().

Run:
    python main.py --n 300 --interval 0.15 --seed 42
"""

import argparse
import os
import threading
import time

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_generator import run as generate_stream
from stream_processor import StreamProcessor

DATA_DIR = "data"
OUTPUT_DIR = "output"
SOURCE_CSV = os.path.join(DATA_DIR, "stream.csv")
PROCESSED_CSV = os.path.join(OUTPUT_DIR, "processed_stream.csv")
SQLITE_PATH = os.path.join(OUTPUT_DIR, "processed_stream.sqlite")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary_table.csv")
ANOMALIES_CSV = os.path.join(OUTPUT_DIR, "anomalies.csv")
LOG_PATH = os.path.join(OUTPUT_DIR, "console_log.txt")
DASHBOARD_PNG = os.path.join(OUTPUT_DIR, "dashboard.png")


_log_file = None


def log(line: str = ""):
    """Print to the console and mirror the line into output/console_log.txt."""
    print(line)
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()


def console_callback(record: dict):
    flag = "  ANOMALY!" if record["is_anomaly"] else ""
    log(
        f"[{record['timestamp']}] {record['sensor_id']:<14} "
        f"T={record['temperature']:6.2f}C  H={record['humidity']:5.1f}%  "
        f"CO2={record['co2']:7.1f}ppm  "
        f"avgT={record['rolling_avg_temp']:6.2f}  stdT={record['rolling_std_temp']:5.2f}  "
        f"rate={record['sensor_rate_hz']:5.2f}Hz{flag}"
        + (f"  [{record['anomaly_reason']}]" if record["is_anomaly"] else "")
    )


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row of headline stats per sensor -- the required 'at least one table'."""
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby("sensor_id").agg(
        records=("temperature", "count"),
        avg_temp=("temperature", "mean"),
        max_temp=("temperature", "max"),
        min_temp=("temperature", "min"),
        std_temp=("temperature", "std"),
        avg_co2=("co2", "mean"),
        max_co2=("co2", "max"),
        anomalies=("is_anomaly", "sum"),
    ).round(2).reset_index()
    grouped["anomaly_rate_%"] = (grouped["anomalies"] / grouped["records"] * 100).round(1)
    return grouped


def build_dashboard(df: pd.DataFrame, path: str):
    """One or more charts -- the required 'at least one visualization'."""
    if df.empty:
        log("[main] no data to plot")
        return

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    anomalies = df[df["is_anomaly"] == 1]

    for sid, g in df.groupby("sensor_id"):
        axes[0].plot(g["timestamp"], g["temperature"], marker=".", linewidth=0.8, label=sid)
        axes[0].plot(g["timestamp"], g["rolling_avg_temp"], linewidth=1.6, linestyle="--",
                     color=axes[0].lines[-1].get_color(), alpha=0.7)
    axes[0].scatter(anomalies["timestamp"], anomalies["temperature"], color="red", zorder=5,
                    marker="x", s=60, label="anomaly")
    axes[0].axhline(27.0, color="gray", linestyle=":", label="temp threshold (27 C)")
    axes[0].set_ylabel("Temperature (C)")
    axes[0].set_title("Temperature per sensor (solid) with rolling average (dashed) + anomalies")
    axes[0].legend(fontsize=7, ncol=4)

    for sid, g in df.groupby("sensor_id"):
        axes[1].plot(g["timestamp"], g["co2"], marker=".", linewidth=0.8, label=sid)
    axes[1].scatter(anomalies["timestamp"], anomalies["co2"], color="red", zorder=5,
                    marker="x", s=60, label="anomaly")
    axes[1].axhline(1000, color="gray", linestyle=":", label="CO2 threshold (1000 ppm)")
    axes[1].set_ylabel("CO2 (ppm)")
    axes[1].set_title("CO2 per sensor")
    axes[1].legend(fontsize=7, ncol=4)

    for sid, g in df.groupby("sensor_id"):
        axes[2].plot(g["timestamp"], g["rolling_std_temp"], linewidth=1.2, label=sid)
    axes[2].set_ylabel("Rolling std (C)")
    axes[2].set_title("Rolling standard deviation of temperature (window = 30 readings)")
    axes[2].legend(fontsize=7, ncol=5)

    counts = df.set_index("timestamp").resample("2s").size()
    axes[3].bar(counts.index + pd.Timedelta(seconds=1), counts.values,
                width=1.8 / 86400, color="steelblue")  # matplotlib date units = days
    axes[3].set_ylabel("Records / 2 s")
    axes[3].set_title("Ingestion rate over time (event count per 2-second bucket)")
    axes[3].set_xlabel("Time (UTC)")

    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main():
    parser = argparse.ArgumentParser(description="Run the real-time stream demo end-to-end.")
    parser.add_argument("--n", type=int, default=300, help="number of records the generator produces")
    parser.add_argument("--interval", type=float, default=0.15, help="seconds between generated records")
    parser.add_argument("--seed", type=int, default=42, help="random seed (makes data AND timestamps reproducible)")
    parser.add_argument("--wall-clock", action="store_true", help="use real wall-clock timestamps instead of the simulated clock")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for p in (PROCESSED_CSV, SQLITE_PATH):
        if os.path.exists(p):
            os.remove(p)
    global _log_file
    _log_file = open(LOG_PATH, "w", encoding="utf-8")

    gen_thread = threading.Thread(
        target=generate_stream,
        kwargs=dict(n_records=args.n, interval=args.interval, out_path=SOURCE_CSV,
                    anomaly_rate=0.04, seed=args.seed, fresh=True,
                    wall_clock=args.wall_clock),
        daemon=True,
    )
    gen_thread.start()
    time.sleep(0.5)  # let the generator create the file + header first

    processor = StreamProcessor(
        source_csv=SOURCE_CSV,
        output_csv=PROCESSED_CSV,
        sqlite_path=SQLITE_PATH,
        window_size=30,
        on_record=console_callback,
    )

    log("[main] processing stream incrementally (Ctrl+C to stop early)...\n")
    idle_polls = 0
    while gen_thread.is_alive() or idle_polls < 8:
        processed = processor.process_available()
        idle_polls = 0 if processed else idle_polls + 1
        time.sleep(0.2)

    stats = processor.global_stats()
    log("\n[main] stream finished. Global stats:")
    for k, v in stats.items():
        log(f"   {k}: {v}")

    df = processor.results_dataframe()
    summary = build_summary_table(df)
    summary.to_csv(SUMMARY_CSV, index=False)
    log(f"\n[main] summary table saved to {SUMMARY_CSV}\n")
    log(summary.to_string(index=False))

    events = df[df["is_anomaly"] == 1][["timestamp", "sensor_id", "temperature", "co2", "anomaly_reason"]]
    events.to_csv(ANOMALIES_CSV, index=False)
    log(f"\n[main] {len(events)} anomalies saved to {ANOMALIES_CSV}")

    check = processor.verify_with_pandas()
    log(f"[main] NumPy vs pandas rolling() cross-check: {check}")

    build_dashboard(df, DASHBOARD_PNG)
    log(f"[main] dashboard saved to {DASHBOARD_PNG}")
    _log_file.close()


if __name__ == "__main__":
    main()
