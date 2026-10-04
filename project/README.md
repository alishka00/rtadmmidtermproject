# Real-Time Data Collection and Stream Processing System

A Smart Campus **environmental sensor monitoring** system built in Python. It
continuously acquires streaming data, cleans and analyzes it incrementally with
Pandas/NumPy, computes real-time indicators, detects rule-based anomalies, and
displays and stores the results.

**Team:** Zharkyn Mukhambetiyar, Alisher Mendulda, Nurkeldi Samuratov
**Course coverage:** Weeks 1–4 — Python for real-time data analysis, data
acquisition and stream processing, real-time processing with Pandas and NumPy.

## 1. Scenario

Five simulated IoT sensor nodes (`ROOM_A101`, `ROOM_A102`, `LAB_B201`,
`LAB_B202`, `AUDITORIUM_C1`) report to a campus network. The network emits one
record every `--interval` seconds (default 0.15 s in the demo); the reporting
node is chosen at random, so each node reports on average once every
`5 × interval` seconds (≈ 0.75 s, i.e. ≈ 1.3 Hz per sensor). Every record has
the schema:

| field         | type            | meaning                                                |
|---------------|-----------------|--------------------------------------------------------|
| `timestamp`   | ISO-8601 string | UTC time the reading was captured                      |
| `sensor_id`   | string          | which room/node produced the reading                   |
| `temperature` | float (°C)      | ambient temperature                                    |
| `humidity`    | float (%)       | relative humidity                                      |
| `co2`         | float (ppm)     | CO₂ concentration, a proxy for air quality / occupancy |

No live campus feed is available, so `data_generator.py` produces a realistic,
reproducible synthetic stream (Gaussian noise around per-sensor baselines) and
injects four kinds of labeled anomalies: `heat` (+6…10 °C), `spike`
(+2.5…3.5 °C, below the absolute threshold), `co2` (+600…1200 ppm) and
`frozen` (a sensor repeating identical values for 6 consecutive readings).

## 2. Project layout

```
project/
├── data_generator.py     # simulates/writes the live stream (data/stream.csv)
├── stream_processor.py   # incremental acquisition + Pandas/NumPy processing + detection + storage
├── main.py               # orchestrates generator + processor, builds table, log & dashboard
├── tests.py              # unit tests (python tests.py)
├── requirements.txt
├── Technical_Report.docx # technical report
├── data/
│   └── stream.csv                  # raw generated stream (input)
└── output/
    ├── processed_stream.csv        # every processed record + indicators + anomaly flag
    ├── processed_stream.sqlite     # same data in SQLite (table `readings`)
    ├── summary_table.csv           # required table: per-sensor summary stats
    ├── anomalies.csv               # every flagged record with the triggering rule(s)
    ├── console_log.txt             # full live console output of the reference run
    ├── console_excerpt.png         # excerpt of the console log (used in the report)
    └── dashboard.png               # required visualization (4 panels)
```

## 3. Installation

Python 3.9+ is required.

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Running the demo

One command runs the generator and the processor together and produces all outputs:

```bash
python main.py --n 300 --interval 0.15 --seed 42
```

Arguments:
- `--n`: number of records the simulated stream produces (default 300)
- `--interval`: seconds between generated records — lower = faster stream (default 0.15)
- `--seed`: random seed (default 42). With a seed the **entire run is
  reproducible, including timestamps** (simulated clock starting at
  2026-09-01 08:00:00 UTC)
- `--wall-clock`: use real timestamps instead of the simulated clock

The run takes about `n × interval` seconds (≈ 45 s for the defaults). While it
runs, one live line is printed **per incoming record**, e.g.:

```
[2026-09-01T08:00:11.250000+00:00] LAB_B202       T= 24.49C  H= 40.1%  CO2=  627.2ppm  avgT= 21.75  stdT= 0.85  rate= 1.39Hz  ANOMALY!  [temp_zscore=5.96]
```

At the end it prints global stream statistics, the per-sensor summary table, a
NumPy-vs-pandas cross-check, and saves the files listed above.

### Running the generator and processor separately

```bash
python data_generator.py --n 500 --interval 0.1 --out data/stream.csv --seed 1
python -c "
from stream_processor import StreamProcessor
p = StreamProcessor('data/stream.csv', 'output/processed_stream.csv', 'output/processed_stream.sqlite')
p.run_until_idle()
print(p.global_stats())
"
```

`stream_processor.py` only ever reads the *new, complete lines* appended to
`data/stream.csv` since the last poll (`poll_new_rows`, byte-offset cursor),
which is the incremental-acquisition requirement. It works the same whether the
file is being written concurrently (as in `main.py`) or is already complete.

### Tests

```bash
python tests.py
```

11 unit tests cover incremental reading, half-written lines, cleaning, each
detection rule, the rate indicator, agreement with `pandas.rolling()`,
storage, and generator reproducibility.

## 5. Real-time indicators (per sensor, rolling window of 30 readings)

1. **Rolling average temperature** (`rolling_avg_temp`) — NumPy mean
2. **Rolling standard deviation of temperature** (`rolling_std_temp`) — NumPy std
3. **Rolling min/max CO₂** (`rolling_min_co2`, `rolling_max_co2`)
4. **Per-sensor report rate** (`sensor_rate_hz`) — `(count − 1) / elapsed` seconds
   between the sensor's first and latest reading (0 for its first reading)
5. **Global event count and anomaly count** (`processor.global_stats()`)

The indicators are computed incrementally from a per-sensor `deque` with NumPy
(O(window) per record, no re-scan of history). `verify_with_pandas()` recomputes
the rolling mean/std offline with `groupby().rolling()` and confirms both agree
(difference < 1e-3, which is only the 3-decimal rounding of the stored values).
The processor's default window is 30 readings, the same value `main.py` uses.

## 6. Anomaly detection rules

A reading is flagged anomalous if **any** of the following holds:
- `temperature >= 27.0 °C` (possible HVAC failure / overheating)
- `co2 >= 1000 ppm` (poor ventilation / overcrowding, ASHRAE-style guidance)
- `|z-score| >= 3.0`, where the z-score compares the reading with the mean/std
  of the sensor's **previous** 30 readings (needs ≥ 10 readings of history)
- 5 consecutive identical `(temperature, humidity, co2)` readings from the same
  sensor (sensor stuck/frozen)

## 7. Storage / reproducibility

Every processed record is appended to **both**:
- `output/processed_stream.csv` (portable, human-readable)
- `output/processed_stream.sqlite` (table `readings`, queryable with SQL)

Running `main.py` twice with the same `--n`, `--interval` and `--seed` produces
byte-identical `data/stream.csv`, `output/processed_stream.csv` and
`output/summary_table.csv`.

## 8. Extending to a real source

To point the system at a real feed, replace `data_generator.py` with anything
that appends rows in the same 5-column CSV schema (a REST polling loop, an MQTT
subscriber writing to disk, a Kafka consumer, …) — `stream_processor.py`
requires no other changes.
