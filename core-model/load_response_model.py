"""
RolledBadger Load-Response Model
Personalized stress/load prediction for running route recommendation.
Based on 41598_2025_Article_25369 (personalized training response).
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Scikit-learn only (no neural nets)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from workout_interpreter import (
    build_athlete_profile,
    build_prescription,
    build_full_recommendation,
    print_full_recommendation,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
ACTIVITIES_PATH = SAMPLE_DATA_DIR / "activities.csv"

# ---------------------------------------------------------------------------
# STEP 1 — Load and Clean Data
# ---------------------------------------------------------------------------
def load_and_clean_activities(csv_path: Path = ACTIVITIES_PATH) -> pd.DataFrame:
    """Load Strava export, keep runs only, add derived columns, sort by date."""
    df = pd.read_csv(csv_path)
    # Strava export has duplicate column names; use positions for first Elapsed Time, Distance
    # Columns: 0=Activity ID, 1=Activity Date, 2=Name, 3=Activity Type, 4=Description, 5=Elapsed Time, 6=Distance
    df = df.iloc[:, :20].copy()
    df = df.rename(columns={
        df.columns[1]: "Activity Date",
        df.columns[3]: "Activity Type",
        df.columns[5]: "Elapsed Time",
        df.columns[6]: "Distance",
    })
    df = df[["Activity Date", "Activity Type", "Elapsed Time", "Distance"]].copy()

    # Keep only running activities
    df = df[df["Activity Type"] == "Run"].copy()

    # Coerce numeric columns (Strava export can have mixed types)
    df["Elapsed Time"] = pd.to_numeric(df["Elapsed Time"], errors="coerce")
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    df = df.dropna(subset=["Elapsed Time", "Distance"])

    # Remove invalid rows: distance <= 0.5 miles, missing elapsed time
    df = df[df["Distance"] > 0.5].copy()
    df = df[df["Elapsed Time"] > 0]

    # Create derived columns
    df["distance"] = df["Distance"].astype(float)
    df["time_min"] = df["Elapsed Time"].astype(float) / 60.0
    df["pace"] = df["time_min"] / df["distance"]

    # Parse Activity Date → datetime
    df["Activity Date"] = pd.to_datetime(df["Activity Date"], format="mixed")
    df = df.sort_values("Activity Date", ascending=True).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# STEP 2 — Athlete Baseline Fitness
# ---------------------------------------------------------------------------
def add_baseline_and_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute baseline pace (median) and relative intensity = baseline_pace / pace."""
    baseline_pace = df["pace"].median()
    df = df.copy()
    df["baseline_pace"] = baseline_pace
    df["intensity"] = baseline_pace / df["pace"]
    return df, baseline_pace


# ---------------------------------------------------------------------------
# STEP 3 — Per-Run Training Load
# ---------------------------------------------------------------------------
def add_training_load(df: pd.DataFrame) -> pd.DataFrame:
    """raw_load = distance * intensity, then load = raw_load / mean(raw_load)."""
    df = df.copy()
    raw_load = df["distance"] * df["intensity"]
    df["raw_load"] = raw_load
    df["load"] = raw_load / raw_load.mean()
    return df


# ---------------------------------------------------------------------------
# STEP 4 — Fatigue Model (Recency Weighted Load)
# ---------------------------------------------------------------------------
LAMBDA = 0.15


def add_fatigue_before_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each run t: fatigue_before_run_t = sum(load_i * exp(-λ * days_between(i, t)))
    over all runs i before t.
    """
    df = df.copy()
    dates = pd.to_datetime(df["Activity Date"]).dt.normalize()
    loads = df["load"].values
    n = len(df)
    fatigue = np.zeros(n)
    for t in range(n):
        dt_t = dates.iloc[t]
        total = 0.0
        for i in range(n):
            if i == t:
                continue
            dt_i = dates.iloc[i]
            days = (dt_t - dt_i).days
            if days <= 0:
                continue
            total += loads[i] * np.exp(-LAMBDA * days)
        fatigue[t] = total
    df["fatigue_before_run"] = fatigue
    return df


def compute_current_fatigue(df: pd.DataFrame) -> float:
    """Current fatigue using last run date as 'today'."""
    dates = pd.to_datetime(df["Activity Date"]).dt.normalize()
    loads = df["load"].values
    last_date = dates.iloc[-1]
    total = 0.0
    for i in range(len(df)):
        days = (last_date - dates.iloc[i]).days
        if days <= 0:
            continue
        total += loads[i] * np.exp(-LAMBDA * days)
    return total


# ---------------------------------------------------------------------------
# STEP 5 — Target Variable
# ---------------------------------------------------------------------------
# y = load (model learns stress = f(distance, intensity, fatigue_before_run))


# ---------------------------------------------------------------------------
# STEP 6 — Train Personalized Model
# ---------------------------------------------------------------------------
FEATURE_COLS = ["distance", "intensity", "fatigue_before_run"]
TARGET_COL = "load"


def train_model(df: pd.DataFrame):
    """Time-based split: first 80% train, last 20% test. GradientBoostingRegressor."""
    n = len(df)
    split_idx = int(0.8 * n)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    print("Train MAE:", round(train_mae, 4))
    print("Test MAE:", round(test_mae, 4))

    return model


# ---------------------------------------------------------------------------
# STEP 7 — Predict Stress for Future Run
# ---------------------------------------------------------------------------
def predict_stress(
    distance: float,
    pace: float,
    fatigue_today: float,
    model: GradientBoostingRegressor,
    baseline_pace: float,
) -> float:
    """intensity = baseline_pace / pace; return model.predict([distance, intensity, fatigue_today])."""
    intensity = baseline_pace / pace
    X = pd.DataFrame(
        [[distance, intensity, fatigue_today]],
        columns=FEATURE_COLS,
    )
    return float(model.predict(X)[0])


# ---------------------------------------------------------------------------
# STEP 8 — Convert Stress → Training Zone
# ---------------------------------------------------------------------------
def get_zone_thresholds(df: pd.DataFrame):
    """easy = 40th percentile, moderate = 70th percentile of load."""
    easy_threshold = np.percentile(df["load"], 40)
    moderate_threshold = np.percentile(df["load"], 70)
    return easy_threshold, moderate_threshold


def classify_zone(
    stress: float,
    easy_threshold: float,
    moderate_threshold: float,
) -> str:
    if stress < easy_threshold:
        return "Easy"
    elif stress < moderate_threshold:
        return "Moderate"
    else:
        return "Hard"


# ---------------------------------------------------------------------------
# STEP 9 — Athlete Insight (Feature Importance)
# ---------------------------------------------------------------------------
def print_athlete_profile(model: GradientBoostingRegressor) -> None:
    """Compare feature importance: intensity vs distance → speed vs volume sensitive."""
    imp = model.feature_importances_
    names = FEATURE_COLS
    for name, i in zip(names, imp):
        print(f"  {name}: {i:.4f}")
    intensity_imp = imp[names.index("intensity")]
    distance_imp = imp[names.index("distance")]
    if intensity_imp > distance_imp:
        print("\nAthlete profile: speed-sensitive (intensity matters more than distance).")
    else:
        print("\nAthlete profile: volume-sensitive (distance matters more than intensity).")


# ---------------------------------------------------------------------------
# Main: run full pipeline
# ---------------------------------------------------------------------------
def main():
    print("RolledBadger Load-Response Model")
    print("=" * 50)

    # Step 1
    df = load_and_clean_activities()
    print(f"\nStep 1 — Loaded {len(df)} runs after cleaning.")

    # Step 2
    df, baseline_pace = add_baseline_and_intensity(df)
    print(f"Step 2 — Baseline pace (median): {baseline_pace:.2f} min/mi.")

    # Step 3
    df = add_training_load(df)
    print("Step 3 — Per-run training load computed and normalized.")

    # Step 4
    df = add_fatigue_before_run(df)
    current_fatigue = compute_current_fatigue(df)
    print(f"Step 4 — Fatigue model applied. Current fatigue (as of last run): {current_fatigue:.4f}.")

    # Step 5 — target y = load (already in df)

    # Step 6
    print("\nStep 6 — Training GradientBoostingRegressor (80/20 time split):")
    model = train_model(df)

    # Step 7 — predict_stress available via function above; demonstrate
    print("\nStep 7 — Example: predict_stress(3.0 miles, 8.0 min/mi, current_fatigue):")
    example_stress = predict_stress(3.0, 8.0, current_fatigue, model, baseline_pace)
    print(f"  Predicted stress: {example_stress:.4f}")

    # Step 8
    easy_th, mod_th = get_zone_thresholds(df)
    print(f"\nStep 8 — Zone thresholds: Easy < {easy_th:.4f} < Moderate < {mod_th:.4f} < Hard")
    zone = classify_zone(example_stress, easy_th, mod_th)
    print(f"  Example run (3 mi @ 8 min/mi) → {zone}")

    # Step 9
    print("\nStep 9 — Feature importance (athlete profile):")
    print_athlete_profile(model)

    # Workout Interpreter: stress → coaching prescription
    predict_stress_fn = lambda d, p, f: predict_stress(d, p, f, model, baseline_pace)
    profile = build_athlete_profile(df, predict_stress_fn, fatigue_col="fatigue_before_run")
    # Example: 4.2 mi @ ~8 min/mi (est 34 min), Steady/Aerobic stimulus
    example_distance, example_pace = 4.2, 8.1
    prescription = build_prescription(
        example_distance, example_pace, current_fatigue, predict_stress_fn, profile
    )
    speed_sensitive = model.feature_importances_[FEATURE_COLS.index("intensity")] > model.feature_importances_[FEATURE_COLS.index("distance")]
    fatigue_p50 = float(np.percentile(df["fatigue_before_run"], 50))
    full_rec = build_full_recommendation(
        prescription, example_distance, example_pace, current_fatigue, profile,
        speed_sensitive=speed_sensitive, fatigue_p65=fatigue_p50,
    )
    print(f"\nStep 10 — Workout Interpreter (example: {example_distance} mi @ {example_pace} min/mi):")
    print_full_recommendation(full_rec)

    # Return for use by route selector
    return {
        "model": model,
        "baseline_pace": baseline_pace,
        "current_fatigue": current_fatigue,
        "easy_threshold": easy_th,
        "moderate_threshold": mod_th,
        "predict_stress": predict_stress_fn,
        "classify_zone": lambda s: classify_zone(s, easy_th, mod_th),
        "athlete_profile": profile,
        "build_prescription": lambda d, p, f: build_prescription(d, p, f, predict_stress_fn, profile),
    }


if __name__ == "__main__":
    main()
