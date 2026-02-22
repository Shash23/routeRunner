"""
RolledBadger PERSONAL prediction layer (confidence-weighted).
Loads global_model.pkl, builds a personal model from user CSV, blends by confidence.
Uses: pandas, numpy, scikit-learn, datetime, joblib, sys. No deep learning.
"""

import sys
from pathlib import Path
from typing import Tuple, Union
from typing import Optional
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = ["distance", "intensity", "fatigue_before_run"]
TARGET_COL = "load"
LAMBDA = 0.15

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GLOBAL_PKL = PROJECT_ROOT / "global_model.pkl"

_GLOBAL_MODEL_CACHE: Optional[GradientBoostingRegressor] = None
_GLOBAL_MODEL_LAST_LOADED: Optional[float] = None
_GLOBAL_MODEL_CHECK_INTERVAL = 0.1  # seconds For now we may expect it to change often since we are in development
_GLOBAL_MODEL_LAST_CHECKED: Optional[float] = None
# ---------------------------------------------------------------------------
# STEP 1 — Load Global Model
# ---------------------------------------------------------------------------
def load_global_model(path: Union[str, Path] = DEFAULT_GLOBAL_PKL):
    """Load global model from pkl (joblib or pickle)."""
    global _GLOBAL_MODEL_CACHE, _GLOBAL_MODEL_LAST_LOADED, _GLOBAL_MODEL_LAST_CHECKED
    now = time.time()
    if _GLOBAL_MODEL_LAST_CHECKED is not None and now - _GLOBAL_MODEL_LAST_CHECKED < _GLOBAL_MODEL_CHECK_INTERVAL:
        return _GLOBAL_MODEL_CACHE
    _GLOBAL_MODEL_LAST_CHECKED = now

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Global model not found: {path}")

    last_changed = path.stat().st_mtime
    if _GLOBAL_MODEL_CACHE is not None and _GLOBAL_MODEL_LAST_LOADED is not None and last_changed == _GLOBAL_MODEL_LAST_LOADED:
        return _GLOBAL_MODEL_CACHE

    try:
        _GLOBAL_MODEL_CACHE = joblib.load(path)
        _GLOBAL_MODEL_LAST_LOADED = last_changed
    except Exception:
        import pickle
        with open(path, "rb") as f:
             _GLOBAL_MODEL_CACHE = pickle.load(f)
             _GLOBAL_MODEL_LAST_LOADED = last_changed
    return _GLOBAL_MODEL_CACHE

# ---------------------------------------------------------------------------
# STEP 2 — Process User CSV
# ---------------------------------------------------------------------------
def _read_strava_csv(path: Path) -> pd.DataFrame:
    """Load CSV; handle Strava export or clean columns."""
    df = pd.read_csv(path)
    needed = {"Activity Type", "Distance", "Elapsed Time", "Activity Date"}
    if needed.issubset(df.columns):
        cols = {}
        for n in needed:
            for c in df.columns:
                if c == n or (isinstance(c, str) and c.startswith(n) and c[len(n):].lstrip(".").isdigit()):
                    cols[n] = c
                    break
        df = df.rename(columns={v: k for k, v in cols.items()})
        df = df[list(needed)].copy()
    else:
        df = df.iloc[:, :20].copy()
        df = df.rename(columns={
            df.columns[1]: "Activity Date",
            df.columns[3]: "Activity Type",
            df.columns[5]: "Elapsed Time",
            df.columns[6]: "Distance",
        })
        df = df[["Activity Date", "Activity Type", "Elapsed Time", "Distance"]].copy()
    return df


def process_user_csv(path: Union[str, Path]) -> Tuple[pd.DataFrame, float, float]:
    """
    Load user CSV, keep runs, compute distance, pace, intensity, load, fatigue_before_run.
    Returns (dataframe with distance, intensity, fatigue_before_run, load), baseline_pace, fatigue_today.
    fatigue_today = last fatigue_before_run.
    """
    path = Path(path)
    df = _read_strava_csv(path)
    df["Activity Type"] = df["Activity Type"].astype(str).str.strip()
    df = df[df["Activity Type"] == "Run"].copy()
    df["Elapsed Time"] = pd.to_numeric(df["Elapsed Time"], errors="coerce")
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    df = df.dropna(subset=["Elapsed Time", "Distance"])
    df = df[df["Distance"] > 0.5].copy()
    df = df[df["Elapsed Time"] > 0]
    if df.empty:
        empty_df = pd.DataFrame(columns=FEATURE_COLS + [TARGET_COL])
        return empty_df, 8.0, 0.0

    df["distance"] = df["Distance"].astype(float)
    df["time_min"] = df["Elapsed Time"].astype(float) / 60.0
    df["pace"] = df["time_min"] / df["distance"]
    df["Activity Date"] = pd.to_datetime(df["Activity Date"], format="mixed")
    df = df.sort_values("Activity Date", ascending=True).reset_index(drop=True)

    baseline_pace = float(df["pace"].median())
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
    fatigue_today = float(fatigue[-1])  # last fatigue_before_run

    return df[FEATURE_COLS + [TARGET_COL]].copy(), baseline_pace, fatigue_today


# ---------------------------------------------------------------------------
# STEP 3 — Train Personal Model
# ---------------------------------------------------------------------------
def train_personal_model(df: pd.DataFrame) -> Tuple[GradientBoostingRegressor, int]:
    """X = [distance, intensity, fatigue_before_run], y = load. Return (model, run_count)."""
    run_count = len(df)
    if run_count == 0:
        model = GradientBoostingRegressor(random_state=42)
        model.fit(np.array([[0.0, 0.0, 0.0]]), np.array([0.0]))
        return model, 0
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    return model, run_count


# ---------------------------------------------------------------------------
# STEP 4 — Personalization Confidence Weight
# ---------------------------------------------------------------------------
def personalization_weight(run_count: int) -> float:
    """k = 35. alpha = run_count / (run_count + k), min(alpha, 0.95)."""
    k = 35
    alpha = run_count / (run_count + k)
    return min(alpha, 0.95)


# ---------------------------------------------------------------------------
# STEP 5 — Final Stress Prediction
# ---------------------------------------------------------------------------
def predict_final_stress(
    distance: float,
    pace: float,
    fatigue_today: float,
    global_model,
    personal_model,
    alpha: float,
    baseline_pace: float,
) -> float:
    """Blend: final_stress = (1 - alpha)*global_pred + alpha*personal_pred."""
    intensity = baseline_pace / pace
    X = pd.DataFrame([[distance, intensity, fatigue_today]], columns=FEATURE_COLS)
    global_pred = global_model.predict(X)[0]
    personal_pred = personal_model.predict(X)[0]
    return float((1.0 - alpha) * global_pred + alpha * personal_pred)


# ---------------------------------------------------------------------------
# STEP 6 — CLI Runner
# ---------------------------------------------------------------------------
def confidence_label(alpha: float) -> str:
    if alpha < 0.25:
        return "Learning your patterns"
    elif alpha < 0.6:
        return "Partially personalized"
    else:
        return "Highly personalized"


def main(user_csv: Union[str, Path], global_model_path: Union[str, Path] = DEFAULT_GLOBAL_PKL) -> None:
    """Load global, process user, train personal, print confidence and example predictions."""
    user_csv = Path(user_csv)
    if not user_csv.is_file():
        print(f"User CSV not found: {user_csv}", file=sys.stderr)
        sys.exit(1)
    global_model = load_global_model(global_model_path)
    df, baseline_pace, fatigue_today = process_user_csv(user_csv)
    personal_model, run_count = train_personal_model(df)
    alpha = personalization_weight(run_count)
    label = confidence_label(alpha)

    # Pace offsets: easy = baseline + 90 sec, steady = baseline + 45 sec, hard = baseline - 20 sec (per mile, in min/mi)
    baseline_pace_min = baseline_pace  # already min/mi
    easy_pace = baseline_pace_min + 90 / 60.0
    steady_pace = baseline_pace_min + 45 / 60.0
    hard_pace = baseline_pace_min - 20 / 60.0

    s_3_easy = predict_final_stress(3.0, easy_pace, fatigue_today, global_model, personal_model, alpha, baseline_pace)
    s_5_steady = predict_final_stress(5.0, steady_pace, fatigue_today, global_model, personal_model, alpha, baseline_pace)
    s_5_hard = predict_final_stress(5.0, hard_pace, fatigue_today, global_model, personal_model, alpha, baseline_pace)

    print("RolledBadger — Personalized Prediction")
    print("=" * 40)
    print("Runs analyzed:", run_count)
    print("Personalization:", round(alpha, 3), "—", label)
    print()
    print("3 mi easy   → stress =", round(s_3_easy, 3))
    print("5 mi steady → stress =", round(s_5_steady, 3))
    print("5 mi hard   → stress =", round(s_5_hard, 3))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python personal_predictor.py user.csv [global_model.pkl]", file=sys.stderr)
        sys.exit(1)
    user_csv = sys.argv[1]
    global_pkl = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_GLOBAL_PKL
    main(user_csv, global_pkl)
