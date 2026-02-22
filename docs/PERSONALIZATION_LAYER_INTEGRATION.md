# Personalization Layer — Integration Guide for Strava API

This document is for engineers linking Strava APIs to the RolledBadger personalization layer. It defines the expected input, where data comes from, and how to call the layer (CLI or Python).

---

## What the personalization layer does

- **Input:** One athlete’s run history (CSV or equivalent data).
- **Output:** Confidence-weighted personalized stress predictions for proposed runs (e.g. 3 mi easy, 5 mi steady, 5 mi hard).
- **Mechanism:** Loads a pre-trained **global model** (`global_model.pkl`), trains a **personal model** on the user’s CSV, then blends predictions: `final_stress = (1 - alpha) * global_pred + alpha * personal_pred`. `alpha` is derived from run count (confidence).

---

## Where the personalization layer gets its data

- **Personal model (per-user):** Trained only on the **user activity data you pass in** (e.g. the CSV path or the same data in another form). No other source is used for personalization.
- **Global model:** Loaded from disk (`global_model.pkl`), already trained on a separate dataset. Path is configurable (see below).

So for “where does the personalization data come from?” — **only from the user file/stream you provide** (e.g. the Strava export or API-derived equivalent).

---

## Required input format

The layer expects **one CSV per athlete** in Strava export style. Your Strava integration should produce (or write) a CSV that matches this.

### Required columns (exact names)

| Column           | Meaning              | Type / format |
|------------------|----------------------|----------------|
| **Activity Type**| Activity type label  | Text; must include `"Run"` for runs we use. |
| **Distance**     | Distance             | Numeric, **miles**. |
| **Elapsed Time** | Duration             | Numeric, **seconds**. |
| **Activity Date**| When the run happened| Date/time (e.g. `"Nov 4, 2022, 1:19:14 PM"` or ISO). |

- The loader accepts **Strava’s full export** (many columns). Only the four above are required; duplicates (e.g. two “Distance” columns) are handled by using the first occurrence.
- If your data comes from the Strava API instead of an export, build a CSV (or DataFrame) with at least these four columns and the same names.

### Filtering the layer applies (you don’t need to do this)

- Keeps only rows with **Activity Type == "Run"**.
- Drops rows with **Distance ≤ 0.5** miles or missing/invalid **Elapsed Time**.

So your job is to supply a CSV (or equivalent) with the four columns above; the layer will filter to runs and valid rows.

### Example header (minimal)

```text
Activity Type,Distance,Elapsed Time,Activity Date
Run,2.58,992,"Nov 4, 2022, 1:19:14 PM"
Run,4.41,2008,"Nov 6, 2022, 9:30:40 PM"
```

### Example header (Strava full export — also supported)

Same four columns must appear (first occurrence used if duplicated):

```text
Activity ID,Activity Date,Activity Name,Activity Type,...,Elapsed Time,Distance,...
```

Reference sample files in the repo: `sample_data/user1.csv`, `user2.csv`, etc. (Strava export format).

---

## How to run the personalization layer

### 1. CLI (file path)

User data is passed as a **path to one CSV file** (one athlete).

```bash
# From repo root, with venv
.venv/bin/python core-model/personal_predictor.py <path_to_user_csv> [path_to_global_model.pkl]
```

Examples:

```bash
.venv/bin/python core-model/personal_predictor.py sample_data/user1.csv
.venv/bin/python core-model/personal_predictor.py /tmp/strava_export_12345.csv /path/to/global_model.pkl
```

- First argument: **user CSV path** (required) — this is the only source of personalization data.
- Second argument: **global model path** (optional). Default is `global_model.pkl` in the repo root.

### 2. Python API (for programmatic use)

You can call the same logic from Python so the “file” you pass is whatever path your Strava pipeline writes (or a path to a temp file you wrote from API data).

```python
from pathlib import Path
import sys

# Add repo root / core-model so imports work
sys.path.insert(0, str(Path("path/to/routeRunner/core-model").resolve()))
from personal_predictor import (
    load_global_model,
    process_user_csv,
    train_personal_model,
    personalization_weight,
    predict_final_stress,
)

# 1) Path to CSV for this user (e.g. downloaded from Strava or built from API)
user_csv_path = "sample_data/user1.csv"

# 2) Load global model (default path: repo root / global_model.pkl)
global_model = load_global_model("path/to/routeRunner/global_model.pkl")

# 3) Process user CSV → dataframe + baseline_pace + fatigue_today
df, baseline_pace, fatigue_today = process_user_csv(user_csv_path)

# 4) Train personal model on this user's data
personal_model, run_count = train_personal_model(df)
alpha = personalization_weight(run_count)

# 5) Predict stress for a proposed run (distance mi, pace min/mi, fatigue_today)
stress = predict_final_stress(
    distance=5.0,
    pace=8.0,
    fatigue_today=fatigue_today,
    global_model=global_model,
    personal_model=personal_model,
    alpha=alpha,
    baseline_pace=baseline_pace,
)
```

- **Personalization data** in this flow is again **only** the CSV at `user_csv_path` (or the DataFrame you could build to match `process_user_csv`’s output).

---

## Output you can expect

- **CLI:** Prints runs analyzed, personalization level (alpha), confidence label, and example stress values (e.g. 3 mi easy, 5 mi steady, 5 mi hard).
- **API:** `predict_final_stress(...)` returns a single float (stress). You can also use `run_count`, `alpha`, `baseline_pace`, `fatigue_today` for UI or logic.

Confidence labels (for UX):

- alpha < 0.25 → "Learning your patterns"
- alpha < 0.6  → "Partially personalized"
- else         → "Highly personalized"

---

## Artifacts and paths

| Item              | Location / meaning |
|-------------------|---------------------|
| Personalization script | `core-model/personal_predictor.py` |
| Global model      | `global_model.pkl` (repo root by default; can override with second CLI arg or `load_global_model(path)`). |
| Sample user CSVs  | `sample_data/user1.csv`, `user2.csv`, … (Strava export format). |

---

## Prompt-ready summary (for your Strava integration prompts)

You can paste the following into your prompts so the integration knows exactly what to target.

```text
RolledBadger personalization layer input:

- One CSV per athlete (Strava export or equivalent).
- Required columns: "Activity Type", "Distance", "Elapsed Time", "Activity Date".
  - Activity Type: string, e.g. "Run".
  - Distance: numeric, miles.
  - Elapsed Time: numeric, seconds.
  - Activity Date: date/time string (e.g. "Nov 4, 2022, 1:19:14 PM" or ISO).
- Only "Run" activities are used; rows with Distance <= 0.5 miles or missing Elapsed Time are dropped.
- The personal model is trained only on this CSV; the global model is loaded from global_model.pkl.

Call:
  CLI: python core-model/personal_predictor.py <user_csv_path> [global_model.pkl]
  Python: process_user_csv(user_csv_path) → (df, baseline_pace, fatigue_today); train_personal_model(df) → (model, run_count); predict_final_stress(distance, pace, fatigue_today, global_model, personal_model, alpha, baseline_pace) → stress.

Reference format: sample_data/user1.csv (Strava export with many columns; the four required columns must be present).
```

---

## Quick reference: key functions

| Function | Purpose |
|----------|--------|
| `load_global_model(path)` | Load `global_model.pkl` (joblib or pickle). |
| `process_user_csv(path)` | Read CSV, filter to runs, compute features; return `(df, baseline_pace, fatigue_today)`. |
| `train_personal_model(df)` | Train personal GBR on `df`; return `(model, run_count)`. |
| `personalization_weight(run_count)` | `alpha = min(0.95, run_count / (run_count + 35))`. |
| `predict_final_stress(distance, pace, fatigue_today, global_model, personal_model, alpha, baseline_pace)` | Blended stress for one proposed run. |

All of this is in `core-model/personal_predictor.py`.
