"""
Load athlete context for coach workflow endpoints.
Uses session, metadata JSON, and persisted personal model. No Strava calls.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

import session_id_manager
import athlete_model_manager

_DEFAULT_PROFILE = {"p40": 0.4, "p65": 0.55, "p80": 0.7, "p92": 0.85}


def get_athlete_context(session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Get athlete_id from session, load metadata and personal model from disk.
    Returns None if session invalid or model/metadata missing.
    """
    athlete_id = session_id_manager.get_athlete_id_from_session(session_id)
    if athlete_id is None:
        return None

    metadata_path = athlete_model_manager._get_model_metadata_path(athlete_id)
    if not metadata_path.is_file():
        return None

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception:
        return None

    pkl_path = athlete_model_manager._get_model_path(athlete_id)
    if not pkl_path.is_file():
        return None

    try:
        am = athlete_model_manager._load_model(athlete_id)
    except Exception:
        return None

    profile_thresholds = metadata.get("profile_thresholds") or _DEFAULT_PROFILE
    run_count = metadata.get("run_count", 0)

    return {
        "athlete_id": athlete_id,
        "model": am.model,
        "baseline_pace": am.baseline_pace,
        "fatigue_today": am.fatigue_today,
        "profile_thresholds": profile_thresholds,
        "alpha": am.alpha,
        "run_count": run_count,
    }
