"""
tests.py -- small unit tests for the stream processor.  Run:  python tests.py
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import data_generator
from stream_processor import StreamProcessor


def make_processor(tmp):
    return StreamProcessor(os.path.join(tmp, "src.csv"), os.path.join(tmp, "out.csv"),
                           os.path.join(tmp, "out.sqlite"), window_size=30)


def append_rows(path, rows, header=False):
    with open(path, "a", newline="") as f:
        if header:
            f.write("timestamp,sensor_id,temperature,humidity,co2\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def ts(i):
    return f"2026-09-01T08:00:{i:02d}.000+00:00"


class ProcessorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = make_processor(self.tmp.name)
        self.src = self.p.source_csv

    def tearDown(self):
        self.tmp.cleanup()

    def test_incremental_read_only_new_rows(self):
        append_rows(self.src, [(ts(0), "S1", 22.0, 45, 500)], header=True)
        self.assertEqual(len(self.p.process_available()), 1)
        self.assertEqual(self.p.process_available(), [])          # nothing new
        append_rows(self.src, [(ts(1), "S1", 22.1, 45, 500), (ts(2), "S1", 22.2, 45, 500)])
        self.assertEqual(len(self.p.process_available()), 2)      # only the 2 new rows
        self.assertEqual(self.p.total_count, 3)

    def test_partial_line_is_not_consumed(self):
        append_rows(self.src, [(ts(0), "S1", 22.0, 45, 500)], header=True)
        with open(self.src, "a") as f:
            f.write(f"{ts(1)},S1,22.1,45")                         # half-written row
        self.assertEqual(len(self.p.process_available()), 1)
        with open(self.src, "a") as f:
            f.write(",500\n")                                      # row completed
        self.assertEqual(len(self.p.process_available()), 1)

    def test_cleaning_drops_invalid_rows(self):
        append_rows(self.src, [(ts(0), "S1", 22.0, 45, 500),
                               (ts(1), "S1", 999, 45, 500),        # impossible temperature
                               (ts(2), "S1", 22.0, 150, 500),      # impossible humidity
                               (ts(3), "S1", "abc", 45, 500)],     # not numeric
                    header=True)
        self.assertEqual(len(self.p.process_available()), 1)

    def test_high_temperature_and_co2_rules(self):
        append_rows(self.src, [(ts(0), "S1", 28.0, 45, 500), (ts(1), "S1", 22.0, 45, 1200)], header=True)
        out = self.p.process_available()
        self.assertIn("high_temperature", out[0]["anomaly_reason"])
        self.assertIn("high_co2", out[1]["anomaly_reason"])

    def test_zscore_rule(self):
        rng = np.random.default_rng(0)
        rows = [(ts(i), "S1", round(22 + rng.normal(0, 0.3), 2), 45 + i % 3, 500 + i) for i in range(20)]
        rows.append((ts(20), "S1", 25.5, 45, 530))                # +3.5 C, below 27 C threshold
        append_rows(self.src, rows, header=True)
        out = self.p.process_available()
        self.assertEqual(out[-1]["is_anomaly"], 1)
        self.assertIn("temp_zscore", out[-1]["anomaly_reason"])
        self.assertNotIn("high_temperature", out[-1]["anomaly_reason"])

    def test_stuck_sensor_rule(self):
        rows = [(ts(i), "S1", 22.0 + 0.1 * i, 45, 500 + i) for i in range(6)]
        rows += [(ts(10 + i), "S1", 22.5, 44.0, 520.0) for i in range(5)]   # 5 identical readings
        append_rows(self.src, rows, header=True)
        out = self.p.process_available()
        self.assertIn("sensor_stuck", out[-1]["anomaly_reason"])
        self.assertEqual(out[-2]["is_anomaly"], 0)

    def test_rate_is_finite_and_sane(self):
        append_rows(self.src, [(ts(i), "S1", 22.0 + 0.1 * i, 45, 500 + i) for i in range(5)], header=True)
        out = self.p.process_available()
        self.assertEqual(out[0]["sensor_rate_hz"], 0.0)            # first reading: undefined
        self.assertAlmostEqual(out[-1]["sensor_rate_hz"], 1.0, places=3)   # 1 reading / second

    def test_matches_pandas_rolling(self):
        rng = np.random.default_rng(1)
        rows = [(ts(i), "S1", round(22 + rng.normal(0, 0.4), 2), 45 + i % 3, 500 + i) for i in range(45)]
        append_rows(self.src, rows, header=True)
        self.p.process_available()
        diff = self.p.verify_with_pandas()
        self.assertLess(diff["max_abs_diff_rolling_mean"], 1e-3)   # outputs are rounded to 3 decimals
        self.assertLess(diff["max_abs_diff_rolling_std"], 1e-3)

    def test_storage_written(self):
        append_rows(self.src, [(ts(0), "S1", 22.0, 45, 500)], header=True)
        self.p.process_available()
        self.assertEqual(len(pd.read_csv(self.p.output_csv)), 1)


class GeneratorTests(unittest.TestCase):
    def run_gen(self, path, seed):
        data_generator.run(120, 0.0, path, anomaly_rate=0.2, seed=seed)

    def test_seed_makes_stream_identical_including_timestamps(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.csv"), os.path.join(d, "b.csv")
            self.run_gen(a, 7)
            self.run_gen(b, 7)
            self.assertEqual(open(a).read(), open(b).read())

    def test_frozen_sensor_repeats_values(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.csv")
            self.run_gen(a, 3)
            df = pd.read_csv(a)
            longest = 0
            for _, g in df.groupby("sensor_id"):
                key = g[["temperature", "humidity", "co2"]].astype(str).agg("|".join, axis=1)
                run = best = 1
                for x, y in zip(key, key[1:]):
                    run = run + 1 if x == y else 1
                    best = max(best, run)
                longest = max(longest, best)
            self.assertGreaterEqual(longest, data_generator.FROZEN_LENGTH)


if __name__ == "__main__":
    unittest.main(verbosity=2)
