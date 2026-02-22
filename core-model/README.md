# RolledBadger Load-Response Model

Personalized stress/load prediction for running route recommendation. Predicts the physiological stress a run will produce for a specific athlete (not pace or performance).

Based on the pipeline in the project README and methodology from *41598_2025_Article_25369* (personalized training response).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r core-model/requirements.txt
```

## Run

From the project root:

```bash
python core-model/load_response_model.py
```

Or with the project venv:

```bash
.venv/bin/python core-model/load_response_model.py
```

## Output

- Loads and cleans `sample_data/activities.csv` (Run only, distance > 0.5 mi, valid time).
- Trains a **GradientBoostingRegressor** on `[distance, intensity, fatigue_before_run]` → `load` (time-based 80/20 split).
- Prints **Train MAE** and **Test MAE**.
- Computes **current fatigue** (as of last run date).
- Demonstrates **predict_stress(distance, pace, fatigue_today)** and **classify_zone(stress)** (Easy / Moderate / Hard from 40th and 70th percentiles of load).
- Prints **feature importance** and athlete profile (speed-sensitive vs volume-sensitive).

## Use from route selector

`main()` returns a dict with:

- `model`, `baseline_pace`, `current_fatigue`
- `easy_threshold`, `moderate_threshold`
- `predict_stress(d, p, f)` and `classify_zone(s)` callables

Use these to evaluate candidate routes and match a target training zone.
