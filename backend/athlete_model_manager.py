import datetime
import sys
import tempfile
from pathlib import Path
from typing import Optional
import json

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from stravalib import Client

import strava_token_db

class AthleteModel:
    def __init__(self, athlete_id: int, model: GradientBoostingRegressor, baseline_pace: float, fatigue_today: float):
        self.athlete_id = athlete_id
        self.model = None
        self.baseline_pace = None
        self.fatigue_today = None

# Core-model personalization layer (process_user_csv, train_personal_model, load_global_model)
_CORE_MODEL_DIR = Path(__file__).resolve().parent.parent / "core-model"
if str(_CORE_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_MODEL_DIR))

from personal_predictor import process_user_csv, train_personal_model, personalization_weight, predict_final_stress, load_global_model

METERS_PER_MILE = 1609.344
_ACTIVITY_FURTHEST_IN_DAYS = 365
_MODELS_PATH = Path(__file__).resolve().parent / "models"
_MODELS_PATH.mkdir(parents=True, exist_ok=True)

def _get_model_path(athlete_id: int) -> Path:
    return _MODELS_PATH / f"{athlete_id}.pkl"

def _get_model_metadata_path(athlete_id: int) -> Path:
    return _MODELS_PATH / f"{athlete_id}.json"

def _save_model(model: AthleteModel) -> None:
    metadata = {
        "baseline_pace": model.baseline_pace,
        "fatigue_today": model.fatigue_today,
    }
    with open(_get_model_metadata_path(model.athlete_id), "w") as f:
        json.dump(metadata, f)
    joblib.dump(model.model, _get_model_path(model.athlete_id))

def _load_model(athlete_id: int) -> AthleteModel:
    with open(_get_model_metadata_path(athlete_id)) as f:
        metadata = json.load(f)
    with open(_get_model_path(athlete_id), "rb") as f:
        model = joblib.load(f)
    return AthleteModel(athlete_id, model, metadata["baseline_pace"], metadata["fatigue_today"])




def _normalize_activity_type(act_type) -> str:
    """Get a plain string from stravalib type (may be object with root='Run' or .value)."""
    if act_type is None:
        return ""
    if hasattr(act_type, "root"):
        return str(getattr(act_type, "root", "") or "")
    if hasattr(act_type, "value"):
        return str(getattr(act_type, "value", "") or "")
    return str(act_type).strip()


def _activities_to_dataframe(activities) -> pd.DataFrame:
    """Build a DataFrame with columns: Activity Type, Distance (mi), Elapsed Time (s), Activity Date (ISO)."""
    rows = []
    for activity in activities:
        act_type_raw = getattr(activity, "type", None) or getattr(activity, "sport_type", None)
        act_type_str = _normalize_activity_type(act_type_raw)
        if act_type_str.lower() != "run":
            continue
        distance_m = float(activity.distance) if activity.distance is not None else 0.0
        distance_mi = distance_m / METERS_PER_MILE
        elapsed = int(activity.elapsed_time) if activity.elapsed_time is not None else 0
        start = activity.start_date
        activity_date = start.isoformat() if start else ""
        rows.append({
            "Activity Type": "Run",
            "Distance": round(distance_mi, 4),
            "Elapsed Time": elapsed,
            "Activity Date": activity_date,
        })
    return pd.DataFrame(rows, columns=["Activity Type", "Distance", "Elapsed Time", "Activity Date"])


async def train_athlete_model(athlete_id: int) -> None:
    """
    Train the personal model using the athlete's Strava run data (DataFrame),
    following the personalization layer: process_user_csv → train_personal_model.
    Saves the trained model to backend/models/{athlete_id}.pkl.
    """
    token = strava_token_db.get_token(athlete_id)
    if token is None:
        print(f"No token found for athlete {athlete_id}", flush=True)
        return

    client = Client()
    client.access_token = token["access_token"]
    after = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=_ACTIVITY_FURTHEST_IN_DAYS)
    activities = client.get_activities(after=after)
    df = _activities_to_dataframe(activities)

    if df.empty:
        print(f"No run activities for athlete {athlete_id}", flush=True)
        return

    # Personalization layer expects CSV path: write df to temp CSV then process
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        tmp_path = Path(f.name)
    try:
        df.to_csv(tmp_path, index=False)
        processed_df, _baseline_pace, _fatigue_today = process_user_csv(tmp_path)
        print(f"Baseline pace: {_baseline_pace}, Fatigue today: {_fatigue_today}", flush=True)
    finally:
        tmp_path.unlink(missing_ok=True)

    personal_model, run_count = train_personal_model(processed_df)
    alpha = personalization_weight(run_count)

    model = AthleteModel(athlete_id, personal_model, _baseline_pace, _fatigue_today)
    _save_model(model)


async def predict_next_stress(athlete_id: int, distance: float, pace: float) -> float:
    model = _load_model(athlete_id)
    global_model = load_global_model()
    stress = predict_final_stress(
        distance=distance,
        pace=pace,
        fatigue_today=model.fatigue_today,
        global_model=global_model,
        personal_model=model.personal_model,
        alpha=model.alpha,
        baseline_pace=model.baseline_pace
    )
    return stress

