"""
RolledBadger Hierarchical Load-Response Model
Cold-start capable: global physiology model + per-user adaptation, blended by run count.
Uses: pandas, numpy, scikit-learn, datetime, glob, os. No deep learning.
"""

import os
import pickle
import glob
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = ["distance", "intensity", "fatigue_before_run"]
TARGET_COL = "load"
LAMBDA = 0.15

# ---------------------------------------------------------------------------
# STEP 1 — Preprocess One Athlete
# ---------------------------------------------------------------------------
def _read_strava_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load CSV; handle Strava export (duplicate column names) or clean columns."""
    path = Path(path)
    df = pd.read_csv(path)
    # Required: Activity Type, Distance, Elapsed Time, Activity Date
    needed = {"Activity Type", "Distance", "Elapsed Time", "Activity Date"}
    if needed.issubset(df.columns):
        # Use first occurrence if duplicates (Strava adds .1, .2)
        cols = {}
        for n in needed:
            for c in df.columns:
                if c == n or (c.startswith(n) and c[len(n):].lstrip(".").isdigit()):
                    cols[n] = c
                    break
        df = df.rename(columns={v: k for k, v in cols.items()})
        df = df[list(needed)].copy()
    else:
        # By position: 1=Activity Date, 3=Activity Type, 5=Elapsed Time, 6=Distance
        df = df.iloc[:, :20].copy()
        df = df.rename(columns={
            df.columns[1]: "Activity Date",
            df.columns[3]: "Activity Type",
            df.columns[5]: "Elapsed Time",
            df.columns[6]: "Distance",
        })
        df = df[["Activity Date", "Activity Type", "Elapsed Time", "Distance"]].copy()
    return df


def process_athlete_csv(path: Union[str, Path], athlete_id: str) -> pd.DataFrame:
    """
    Load one athlete CSV, keep runs, compute distance, pace, intensity, load, fatigue.
    Returns dataframe with: athlete_id, distance, intensity, fatigue_before_run, load.
    """
    df = _read_strava_csv(path)
    df["Activity Type"] = df["Activity Type"].astype(str).str.strip()
    df = df[df["Activity Type"] == "Run"].copy()
    df["Elapsed Time"] = pd.to_numeric(df["Elapsed Time"], errors="coerce")
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    df = df.dropna(subset=["Elapsed Time", "Distance"])
    df = df[df["Distance"] > 0.5].copy()
    df = df[df["Elapsed Time"] > 0]
    if df.empty:
        df["athlete_id"] = athlete_id
        df["distance"] = pd.Series(dtype=float)
        df["intensity"] = pd.Series(dtype=float)
        df["fatigue_before_run"] = pd.Series(dtype=float)
        df["load"] = pd.Series(dtype=float)
        return df

    df["distance"] = df["Distance"].astype(float)
    df["time_min"] = df["Elapsed Time"].astype(float) / 60.0
    df["pace"] = df["time_min"] / df["distance"]
    df["Activity Date"] = pd.to_datetime(df["Activity Date"], format="mixed")
    df = df.sort_values("Activity Date", ascending=True).reset_index(drop=True)

    baseline_pace = df["pace"].median()
    df["intensity"] = baseline_pace / df["pace"]
    raw_load = df["distance"] * df["intensity"]
    load_mean = raw_load.mean()
    df["load"] = (raw_load / load_mean) if load_mean > 0 else raw_load

    dates = df["Activity Date"].dt.normalize()
    loads = df["load"].values
    n = len(df)
    fatigue = np.zeros(n)
    for t in range(n):
        dt_t = dates.iloc[t]
        total = 0.0
        for i in range(n):
            if i == t:
                continue
            days = (dt_t - dates.iloc[i]).days
            if days <= 0:
                continue
            total += loads[i] * np.exp(-LAMBDA * days)
        fatigue[t] = total
    df["fatigue_before_run"] = fatigue
    df["athlete_id"] = athlete_id

    return df[["athlete_id", "distance", "intensity", "fatigue_before_run", "load", "Activity Date", "pace"]].copy()


# ---------------------------------------------------------------------------
# STEP 2 — Build Global Dataset
# ---------------------------------------------------------------------------
def build_global_dataset(folder: Union[str, Path]) -> pd.DataFrame:
    """For each CSV in folder, assign athlete_id, process, append. Return combined dataframe."""
    folder = Path(folder)
    if not folder.is_dir():
        return pd.DataFrame(columns=["athlete_id", "distance", "intensity", "fatigue_before_run", "load"])
    all_dfs = []
    for i, csv_path in enumerate(sorted(glob.glob(os.path.join(folder, "*.csv")))):
        df = process_athlete_csv(csv_path, f"athlete_{i}")
        if len(df) > 0:
            all_dfs.append(df)
    if not all_dfs:
        return pd.DataFrame(columns=["athlete_id", "distance", "intensity", "fatigue_before_run", "load"])
    return pd.concat(all_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# STEP 3 — Train Global Model (with train/test split)
# ---------------------------------------------------------------------------
def train_global_model(
    global_df: pd.DataFrame,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[GradientBoostingRegressor, float, float]:
    """
    Train on 80% of global dataset, hold out 20% for test.
    Returns (model, train_mae, test_mae).
    """
    if len(global_df) == 0:
        model = GradientBoostingRegressor(random_state=random_state)
        model.fit(np.array([[0.0, 0.0, 0.0]]), np.array([0.0]))
        return model, 0.0, 0.0
    df = global_df.copy()
    if "Activity Date" in df.columns:
        df = df.sort_values("Activity Date", ascending=True).reset_index(drop=True)
    n = len(df)
    split_idx = max(1, int(n * (1 - test_fraction)))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]
    model = GradientBoostingRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    return model, float(train_mae), float(test_mae)


def save_global_model(model: GradientBoostingRegressor, path: Union[str, Path]) -> None:
    """Save model as global_model.pkl."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_global_model(path: Union[str, Path]) -> GradientBoostingRegressor:
    """Load global_model.pkl."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# STEP 4 — Train Personal Model
# ---------------------------------------------------------------------------
def train_personal_model(
    user_csv: Union[str, Path],
    athlete_id: str = "user",
    test_fraction: float = 0.2,
) -> Tuple[GradientBoostingRegressor, int, float, float]:
    """Process user CSV, train on 80%, test on 20% (time-based). Return (model, run_count, train_mae, test_mae)."""
    df = process_athlete_csv(user_csv, athlete_id)
    run_count = len(df)
    if run_count == 0:
        model = GradientBoostingRegressor(random_state=42)
        model.fit(np.array([[0.0, 0.0, 0.0]]), np.array([0.0]))
        return model, 0, 0.0, 0.0
    if "Activity Date" in df.columns:
        df = df.sort_values("Activity Date", ascending=True).reset_index(drop=True)
    n = len(df)
    split_idx = max(1, int(n * (1 - test_fraction)))
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
    return model, run_count, float(train_mae), float(test_mae)


# ---------------------------------------------------------------------------
# STEP 5 — Personalization Weight
# ---------------------------------------------------------------------------
def personalization_weight(run_count: int) -> float:
    """alpha = min(0.95, run_count / 120)."""
    return min(0.95, run_count / 120)


# ---------------------------------------------------------------------------
# STEP 6 — Final Stress Prediction
# ---------------------------------------------------------------------------
def predict_final_stress(
    distance: float,
    pace: float,
    fatigue_today: float,
    global_model: GradientBoostingRegressor,
    personal_model: GradientBoostingRegressor,
    alpha: float,
    baseline_pace: float,
) -> float:
    """Blend global and personal predictions: (1-alpha)*global + alpha*personal."""
    intensity = baseline_pace / pace
    X = pd.DataFrame([[distance, intensity, fatigue_today]], columns=FEATURE_COLS)
    global_pred = global_model.predict(X)[0]
    personal_pred = personal_model.predict(X)[0]
    final_stress = (1.0 - alpha) * global_pred + alpha * personal_pred
    return float(final_stress)


# ---------------------------------------------------------------------------
# STEP 7 — CLI demo helpers
# ---------------------------------------------------------------------------
def get_user_baseline_pace_and_fatigue(user_csv: Union[str, Path]) -> Tuple[float, float]:
    """Process user CSV; return (baseline_pace, current_fatigue). current_fatigue as of last run date."""
    df = process_athlete_csv(user_csv, "user")
    if len(df) == 0 or "pace" not in df.columns:
        return 8.0, 0.0
    baseline_pace = float(df["pace"].median())
    dates = pd.to_datetime(df["Activity Date"]).dt.normalize()
    loads = df["load"].values
    last_date = dates.iloc[-1]
    current_fatigue = sum(
        loads[i] * np.exp(-LAMBDA * (last_date - dates.iloc[i]).days)
        for i in range(len(df))
    )
    return baseline_pace, float(current_fatigue)


def predict_stress_global(
    distance: float,
    pace: float,
    fatigue_today: float,
    global_model: GradientBoostingRegressor,
    baseline_pace: float,
) -> float:
    """Predict stress using global model only. intensity = baseline_pace / pace."""
    intensity = baseline_pace / pace
    X = pd.DataFrame([[distance, intensity, fatigue_today]], columns=FEATURE_COLS)
    return float(global_model.predict(X)[0])


# ---------------------------------------------------------------------------
# CLI — Train and test global model only (all data in sample_data)
# ---------------------------------------------------------------------------
def run_global_model(
    data_folder: Union[str, Path],
    global_model_path: Union[str, Path] = "global_model.pkl",
) -> GradientBoostingRegressor:
    """
    Build dataset from all CSVs in data_folder, train global model (80/20 train/test),
    save to global_model_path, print Train/Test MAE and example predictions.
    Returns the trained model.
    """
    data_folder = Path(data_folder)
    global_path = Path(global_model_path)
    if not data_folder.is_dir():
        print("Data folder not found:", data_folder)
        model, _, _ = train_global_model(pd.DataFrame(columns=FEATURE_COLS + [TARGET_COL]))
        return model
    global_df = build_global_dataset(data_folder)
    n_csvs = len(sorted(glob.glob(os.path.join(str(data_folder), "*.csv"))))
    if len(global_df) == 0:
        print("No runs found in", data_folder)
        model, _, _ = train_global_model(pd.DataFrame(columns=FEATURE_COLS + [TARGET_COL]))
        return model
    global_model, train_mae, test_mae = train_global_model(global_df)
    save_global_model(global_model, global_path)
    print("RolledBadger — Global model")
    print("=" * 40)
    print(f"Data: {n_csvs} CSVs, {len(global_df)} runs")
    print(f"Train MAE: {train_mae:.4f}  |  Test MAE: {test_mae:.4f}")
    print(f"Saved: {global_path}")
    median_pace = float(global_df["pace"].median()) if "pace" in global_df.columns else 8.0
    median_fatigue = float(global_df["fatigue_before_run"].median())
    print()
    print("Example predictions (global model, median pace/fatigue):")
    s3 = predict_stress_global(3.0, median_pace + 1.0, median_fatigue, global_model, median_pace)
    s5 = predict_stress_global(5.0, median_pace, median_fatigue, global_model, median_pace)
    print(f"  3 mi easy:     stress = {s3:.3f}")
    print(f"  5 mi moderate: stress = {s5:.3f}")
    return global_model


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    SAMPLE_DATA = PROJECT_ROOT / "sample_data"
    GLOBAL_PKL = PROJECT_ROOT / "global_model.pkl"
    run_global_model(SAMPLE_DATA, global_model_path=GLOBAL_PKL)