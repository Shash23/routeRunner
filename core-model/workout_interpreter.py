"""
RolledBadger Workout Interpreter
Converts stress_score from the ML model into human coaching prescription:
training zone, workout type, pace range, heart rate range, elevation gain target.
Uses only Python standard library + numpy + pandas (no new ML models).
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# STEP 1 — Compute Athlete Distribution Statistics
# ---------------------------------------------------------------------------
def build_athlete_profile(
    df: pd.DataFrame,
    predict_stress: Callable[[float, float, float], float],
    fatigue_col: str = "fatigue_before_run",
) -> Dict[str, float]:
    """
    Using historical dataset: median_distance, median_pace, stress percentiles p40, p65, p80, p92.
    Stress values = predict_stress(distance, pace, fatigue) for each row.
    """
    median_distance = float(df["distance"].median())
    median_pace = float(df["pace"].median())
    stress_values = np.array([
        predict_stress(row.distance, row.pace, getattr(row, fatigue_col))
        for row in df.itertuples()
    ])
    p40 = float(np.percentile(stress_values, 40))
    p65 = float(np.percentile(stress_values, 65))
    p80 = float(np.percentile(stress_values, 80))
    p92 = float(np.percentile(stress_values, 92))
    return {
        "median_distance": median_distance,
        "median_pace": median_pace,
        "p40": p40,
        "p65": p65,
        "p80": p80,
        "p92": p92,
    }


# ---------------------------------------------------------------------------
# STEP 2 — Map Stress → Training Zone
# ---------------------------------------------------------------------------
def classify_training_zone(stress: float, profile: Dict[str, float]) -> str:
    p40, p65, p80, p92 = profile["p40"], profile["p65"], profile["p80"], profile["p92"]
    if stress < p40:
        return "Recovery"
    elif stress < p65:
        return "Easy"
    elif stress < p80:
        return "Steady"
    elif stress < p92:
        return "Tempo"
    else:
        return "Interval"


# ---------------------------------------------------------------------------
# STEP 3 — Training Zone → Official Workout Type
# ---------------------------------------------------------------------------
ZONE_TO_WORKOUT = {
    "Recovery": "Recovery Run",
    "Easy": "Easy Run",
    "Steady": "Base Run",
    "Tempo": "Tempo Run",
    "Interval": "Interval Workout",
}


def get_workout_type(zone: str, distance: float, profile: Dict[str, float]) -> str:
    if distance > 1.35 * profile["median_distance"]:
        return "Long Run"
    return ZONE_TO_WORKOUT[zone]


# ---------------------------------------------------------------------------
# STEP 4 — Estimate Heart Rate Range
# ---------------------------------------------------------------------------
HRMAX_DEFAULT = 200

ZONE_HR_PCT = {
    "Recovery": (0.60, 0.70),
    "Easy": (0.65, 0.75),
    "Steady": (0.70, 0.80),
    "Tempo": (0.80, 0.88),
    "Interval": (0.88, 0.95),
}


def get_hr_range(zone: str, hrmax: int = HRMAX_DEFAULT) -> Tuple[int, int]:
    low_pct, high_pct = ZONE_HR_PCT[zone]
    low_bpm = int(round(hrmax * low_pct))
    high_bpm = int(round(hrmax * high_pct))
    return (low_bpm, high_bpm)


# ---------------------------------------------------------------------------
# STEP 5 — Pace Recommendation
# ---------------------------------------------------------------------------
# Offsets in seconds per mile from median_pace (slower = positive, faster = negative)
ZONE_PACE_OFFSET_SEC = {
    "Recovery": (90, 150),
    "Easy": (60, 90),
    "Steady": (30, 60),
    "Tempo": (-10, 20),
    "Interval": (-60, -20),
}


def _format_pace(pace_min_per_mi: float) -> str:
    """Format pace as M:SS /mi (pace in min/mi)."""
    total_sec = pace_min_per_mi * 60
    if total_sec < 0:
        total_sec = 0
    m = int(total_sec // 60)
    s = int(round(total_sec % 60))
    if s == 60:
        s = 0
        m += 1
    return f"{m}:{s:02d}"


def get_pace_range(zone: str, profile: Dict[str, float]) -> str:
    """Return formatted string 'min:sec – min:sec /mi'."""
    median_pace = profile["median_pace"]
    median_sec = median_pace * 60
    low_off, high_off = ZONE_PACE_OFFSET_SEC[zone]
    low_sec = median_sec + low_off
    high_sec = median_sec + high_off
    if low_sec > high_sec:
        low_sec, high_sec = high_sec, low_sec
    if low_sec < 0:
        low_sec = 0
    low_pace_mi = low_sec / 60
    high_pace_mi = high_sec / 60
    return f"{_format_pace(low_pace_mi)}–{_format_pace(high_pace_mi)} /mi"


# ---------------------------------------------------------------------------
# STEP 6 — Elevation Gain Target
# ---------------------------------------------------------------------------
def get_elevation_target(
    distance: float,
    pace: float,
    fatigue_today: float,
    predict_stress: Callable[[float, float, float], float],
    profile: Dict[str, float],
) -> Tuple[float, float]:
    """
    flat_stress = predict_stress(distance, median_pace+60 sec, fatigue_today)
    stress_gap = predicted_stress - flat_stress
    elevation_gain_target = max(0, stress_gap / 0.03 * 100), range ±20%
    """
    median_pace = profile["median_pace"]
    pace_slower_60_sec = median_pace + (60 / 60.0)
    flat_stress = predict_stress(distance, pace_slower_60_sec, fatigue_today)
    predicted_stress = predict_stress(distance, pace, fatigue_today)
    stress_gap = predicted_stress - flat_stress
    elevation_ft = max(0.0, stress_gap / 0.03 * 100)
    low_ft = elevation_ft * 0.8
    high_ft = elevation_ft * 1.2
    return (low_ft, high_ft)


# ---------------------------------------------------------------------------
# STEP 7 — Build Final Prescription Object
# ---------------------------------------------------------------------------
def build_prescription(
    distance: float,
    pace: float,
    fatigue_today: float,
    predict_stress: Callable[[float, float, float], float],
    profile: Dict[str, float],
    hrmax: int = HRMAX_DEFAULT,
) -> Dict[str, Any]:
    stress = predict_stress(distance, pace, fatigue_today)
    zone = classify_training_zone(stress, profile)
    workout_type = get_workout_type(zone, distance, profile)
    hr_low, hr_high = get_hr_range(zone, hrmax)
    pace_range = get_pace_range(zone, profile)
    elev_low, elev_high = get_elevation_target(
        distance, pace, fatigue_today, predict_stress, profile
    )
    return {
        "workout_type": workout_type,
        "training_zone": zone,
        "predicted_stress": round(stress, 2),
        "pace_range": pace_range,
        "heart_rate_range": f"{hr_low}–{hr_high} bpm",
        "elevation_gain_target": f"{int(round(elev_low))}–{int(round(elev_high))} ft",
    }


# ---------------------------------------------------------------------------
# STEP 8 — CLI Demo Output (combined coaching format)
# ---------------------------------------------------------------------------
ZONE_TO_STIMULUS = {
    "Recovery": "Recovery",
    "Easy": "Aerobic Development",
    "Steady": "Aerobic Development",
    "Tempo": "Threshold",
    "Interval": "High Intensity",
}

ZONE_TO_FEEL = {
    "Recovery": "Very easy, shake-out effort",
    "Easy": "Comfortable conversational pace",
    "Steady": "Comfortable steady effort",
    "Tempo": "Sustainably hard, controlled discomfort",
    "Interval": "High effort, short bouts",
}


def get_recovery_status(current_fatigue: float, fatigue_p65: float = 4.0) -> str:
    """Map current fatigue to recovery status (simple bands)."""
    if fatigue_p65 <= 0:
        fatigue_p65 = 1.0
    ratio = current_fatigue / fatigue_p65
    if ratio < 0.5:
        return "Well Recovered"
    elif ratio < 1.25:
        return "Moderately Recovered"
    else:
        return "Fatigued"


def stress_to_effort(stress: float, profile: Dict[str, float]) -> int:
    """Map stress to predicted effort 1–10 using profile percentiles."""
    p40, p65, p80, p92 = profile["p40"], profile["p65"], profile["p80"], profile["p92"]
    if stress <= p40:
        return max(1, min(3, int(1 + 2 * (stress / p40))))
    elif stress <= p65:
        return max(3, min(5, int(3 + 2 * (stress - p40) / (p65 - p40))))
    elif stress <= p80:
        return max(5, min(6, int(5 + (stress - p65) / (p80 - p65))))
    elif stress <= p92:
        return max(6, min(8, int(6 + 2 * (stress - p80) / (p92 - p80))))
    else:
        return max(8, min(10, int(8 + 2 * min(1.0, (stress - p92) / (p92 - p40)))))


def get_why_bullets(
    zone: str,
    current_fatigue: float,
    elevation_mid_ft: float,
    recovery_status: str,
) -> list:
    """Generate 'Why this run' bullets."""
    bullets = []
    if "Fatigued" in recovery_status or current_fatigue > 3:
        bullets.append("Recent load is elevated")
    if elevation_mid_ft > 50:
        bullets.append("Terrain matches target stimulus")
        bullets.append("Flat routes would undertrain today")
    elif zone in ("Easy", "Recovery"):
        bullets.append("Low stress supports recovery")
    if not bullets:
        bullets.append("Pace and distance match today's target zone")
    return bullets


def get_athlete_profile_summary(speed_sensitive: bool) -> str:
    """One-line athlete profile for output."""
    if speed_sensitive:
        return "Speed-sensitive runner. Intensity impacts stress more than distance."
    return "Volume-sensitive runner. Distance impacts stress more than speed."


def build_full_recommendation(
    prescription: Dict[str, Any],
    distance: float,
    pace: float,
    current_fatigue: float,
    profile: Dict[str, float],
    speed_sensitive: bool = False,
    fatigue_p65: float = 4.0,
) -> Dict[str, Any]:
    """Add recovery status, suggested run, why-this-run, and athlete profile to prescription."""
    zone = prescription["training_zone"]
    stress = prescription["predicted_stress"]
    elev_str = prescription["elevation_gain_target"]
    elev_parts = elev_str.replace(" ft", "").replace("–", "-").split("-")
    elev_low, elev_high = int(elev_parts[0]), int(elev_parts[1])
    elev_mid = (elev_low + elev_high) // 2
    est_time_min = int(round(distance * pace))
    recovery_status = get_recovery_status(current_fatigue, fatigue_p65)
    recommended_stimulus = ZONE_TO_STIMULUS[zone]
    predicted_effort = stress_to_effort(stress, profile)
    feel = ZONE_TO_FEEL[zone]
    why_bullets = get_why_bullets(zone, current_fatigue, elev_mid, recovery_status)
    athlete_summary = get_athlete_profile_summary(speed_sensitive)
    return {
        **prescription,
        "recovery_status": recovery_status,
        "recommended_stimulus": recommended_stimulus,
        "predicted_effort": predicted_effort,
        "suggested_run": {
            "distance_mi": distance,
            "est_time_min": est_time_min,
            "elevation_ft": elev_mid,
            "feel": feel,
        },
        "why_this_run": why_bullets,
        "athlete_profile_summary": athlete_summary,
    }


def print_prescription(prescription: Dict[str, Any]) -> None:
    """Print legacy short RolledBadger recommendation."""
    print("\nRolledBadger Recommendation")
    print("-" * 40)
    print(f"Workout: {prescription['workout_type']}")
    print(f"Zone: {prescription['training_zone']}")
    print(f"Target Pace: {prescription['pace_range']}")
    print(f"Heart Rate: {prescription['heart_rate_range']}")
    print(f"Elevation Gain: {prescription['elevation_gain_target']}")
    print()


def print_full_recommendation(rec: Dict[str, Any]) -> None:
    """Print combined coaching output: recovery, stimulus, effort, suggested run, why, athlete profile."""
    sr = rec["suggested_run"]
    print("\nRolledBadger Recommendation")
    print("-" * 40)
    print(f"Recovery Status: {rec['recovery_status']}")
    print(f"Recommended Stimulus: {rec['recommended_stimulus']}")
    print(f"Predicted Effort: {rec['predicted_effort']} / 10")
    print()
    print("Suggested Run:")
    print(f"  Distance: {sr['distance_mi']:.1f} mi")
    print(f"  Est Time: {sr['est_time_min']} min")
    print(f"  Elevation: {sr['elevation_ft']} ft")
    print(f"  Feel: {sr['feel']}")
    print()
    print("Why this run:")
    for b in rec["why_this_run"]:
        print(f"  - {b}")
    print()
    print("Targets:")
    print(f"  Workout: {rec['workout_type']}")
    print(f"  Zone: {rec['training_zone']}")
    print(f"  Target Pace: {rec['pace_range']}")
    print(f"  Heart Rate: {rec['heart_rate_range']}")
    print(f"  Elevation Gain: {rec['elevation_gain_target']}")
    print()
    print("Athlete Profile:")
    print(f"  {rec['athlete_profile_summary']}")
    print()
