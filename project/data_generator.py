"""
data_generator.py
------------------
Simulates a live stream of environmental sensor readings from a "Smart Campus"
IoT deployment. Each record is one reading from one sensor node (a room, lab
or auditorium) and is appended to a CSV file one row at a time, with a small
delay, to emulate an incrementally arriving data source.

Scenario
--------
Domain:    IoT / Smart Campus environmental monitoring
Source:    5 simulated sensor nodes (classrooms / labs / auditorium). The network
           emits one record every `--interval` seconds (default 0.2 s); the node
           that reports is chosen at random, so each node reports on average
           once every 5 * interval seconds.
Schema:
    timestamp   : ISO-8601 datetime string, time the reading was taken (UTC)
    sensor_id   : string, e.g. "ROOM_A101"
    temperature : float, degrees Celsius
    humidity    : float, percent relative humidity
    co2         : float, parts per million (air-quality proxy)

Injected anomalies (so the detector has real events to catch)
-------------------------------------------------------------
    heat    : temperature +6..10 C         (absolute-threshold rule)
    spike   : temperature +2.5..3.5 C      (below the threshold, caught by z-score)
    co2     : CO2 +600..1200 ppm           (absolute-threshold rule)
    frozen  : the sensor repeats exactly the same values for 6 consecutive
              readings (stuck-sensor rule)

Reproducibility
---------------
When `--seed` is given the whole stream is deterministic, including timestamps:
they come from a simulated clock that starts at SIM_START and advances by
`interval` seconds per record. Without a seed (or with `--wall-clock`) real
wall-clock timestamps are used.

Usage
-----
    python data_generator.py --n 300 --interval 0.2 --out data/stream.csv --seed 1
"""

import argparse
import csv
import os
import random
import time
from datetime import datetime, timedelta, timezone

SENSORS = ["ROOM_A101", "ROOM_A102", "LAB_B201", "LAB_B202", "AUDITORIUM_C1"]

# Baseline "normal" operating ranges per sensor (mean, std) used as the
# generative model for realistic-looking noisy readings.
BASELINES = {
    "ROOM_A101": {"temp": (22.0, 0.4), "hum": (45.0, 3.0), "co2": (500, 40)},
    "ROOM_A102": {"temp": (22.5, 0.4), "hum": (44.0, 3.0), "co2": (520, 40)},
    "LAB_B201":  {"temp": (21.0, 0.5), "hum": (40.0, 3.5), "co2": (600, 60)},
    "LAB_B202":  {"temp": (21.5, 0.5), "hum": (41.0, 3.5), "co2": (610, 60)},
    "AUDITORIUM_C1": {"temp": (23.0, 0.6), "hum": (50.0, 4.0), "co2": (700, 80)},
}

FIELDNAMES = ["timestamp", "sensor_id", "temperature", "humidity", "co2"]
ANOMALY_TYPES = ["heat", "spike", "co2", "frozen"]
FROZEN_LENGTH = 6                       # readings a stuck sensor repeats
SIM_START = datetime(2026, 9, 1, 8, 0, 0, tzinfo=timezone.utc)


def make_reading(sensor_id: str, timestamp: datetime, anomaly: str = None) -> dict:
    """Generate one synthetic reading, optionally with an injected anomaly
    ('heat', 'spike', 'co2'). 'frozen' is handled by `run` (it needs state)."""
    base = BASELINES[sensor_id]
    temp = random.gauss(*base["temp"])
    hum = random.gauss(*base["hum"])
    co2 = random.gauss(*base["co2"])

    if anomaly == "heat":
        temp += random.uniform(6, 10)      # overheating / fire risk
    elif anomaly == "spike":
        temp += random.uniform(2.5, 3.5)   # sudden but sub-threshold jump
    elif anomaly == "co2":
        co2 += random.uniform(600, 1200)   # overcrowding / poor ventilation

    return {
        "timestamp": timestamp.isoformat(timespec="milliseconds"),
        "sensor_id": sensor_id,
        "temperature": round(temp, 2),
        "humidity": round(hum, 2),
        "co2": round(co2, 1),
    }


def run(n_records: int, interval: float, out_path: str, anomaly_rate: float = 0.03,
        seed: int = None, fresh: bool = True, wall_clock: bool = False):
    """Write `n_records` readings to `out_path`, one every `interval` seconds."""
    if seed is not None:
        random.seed(seed)
    simulated = (seed is not None) and not wall_clock

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_header = fresh or not os.path.exists(out_path)
    mode = "w" if fresh else "a"

    frozen = {}   # sensor_id -> (values, readings still to repeat)

    with open(out_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            f.flush()

        for i in range(n_records):
            ts = (SIM_START + timedelta(seconds=i * interval)) if simulated \
                else datetime.now(timezone.utc)
            sensor_id = random.choice(SENSORS)
            injected = None

            if sensor_id in frozen:
                # sensor is currently stuck: repeat the exact same values
                values, left = frozen[sensor_id]
                row = {"timestamp": ts.isoformat(timespec="milliseconds"),
                       "sensor_id": sensor_id, **values}
                if left - 1 > 0:
                    frozen[sensor_id] = (values, left - 1)
                else:
                    del frozen[sensor_id]
            else:
                if random.random() < anomaly_rate:
                    injected = random.choice(ANOMALY_TYPES)
                row = make_reading(sensor_id, ts, None if injected == "frozen" else injected)
                if injected == "frozen":
                    values = {"temperature": row["temperature"],
                              "humidity": row["humidity"], "co2": row["co2"]}
                    frozen[sensor_id] = (values, FROZEN_LENGTH - 1)

            writer.writerow(row)
            f.flush()  # make the row visible to a concurrent reader immediately
            if injected:
                print(f"[generator] injected '{injected}' anomaly -> {row}")
            time.sleep(interval)

    print(f"[generator] finished writing {n_records} records to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a live sensor data stream.")
    parser.add_argument("--n", type=int, default=300, help="number of records to generate")
    parser.add_argument("--interval", type=float, default=0.2, help="seconds between records")
    parser.add_argument("--out", type=str, default="data/stream.csv", help="output CSV path")
    parser.add_argument("--anomaly-rate", type=float, default=0.03, help="probability of an anomalous reading")
    parser.add_argument("--seed", type=int, default=None, help="random seed (makes the stream fully reproducible)")
    parser.add_argument("--wall-clock", action="store_true", help="use real timestamps even when a seed is set")
    parser.add_argument("--append", action="store_true", help="append instead of overwriting the CSV")
    args = parser.parse_args()

    run(args.n, args.interval, args.out, args.anomaly_rate, args.seed,
        fresh=not args.append, wall_clock=args.wall_clock)
