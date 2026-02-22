"""
RolledBadger Manual Edit system.

Manual edit does NOT directly force a route. It proposes a new training stimulus.
The system: interprets intent → computes requested stress → clamps safely →
reclassifies workout type → regenerates route.

Usage: build stress_predictor and athlete_profile (e.g. from core-model
personal_predictor + workout_interpreter.build_athlete_profile), then:

  result = process(
      distance_miles=4.0,
      hr_zone="Easy",
      baseline_pace=baseline_pace,
      fatigue_today=fatigue_today,
      today_recommended_stress=0.65,
      original_recommended_distance_mi=3.5,
      stress_predictor=stress_predictor,
      athlete_profile=athlete_profile,
      lat=43.07, lon=-89.45,
  )
  # result["route"], result["workout_type"], result["message"], result["was_adjusted"], etc.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

# Pace offsets (sec per mile) for HR zone → pace (baseline + offset in min/mi)
ZONE_PACE_OFFSET_SEC = {
    "recovery": 90,
    "easy": 60,
    "steady": 40,
    "tempo": 10,
    "interval": -20,
}

# Stress thresholds: map final_stress to workout type (athlete percentiles p40, p65, p80, p92)
def _reclassify_workout_type(
    final_stress: float,
    profile: Dict[str, float],
) -> str:
    """Reclassify workout type from clamped stress using athlete thresholds."""
    p40 = profile.get("p40", 0.4)
    p65 = profile.get("p65", 0.55)
    p80 = profile.get("p80", 0.7)
    p92 = profile.get("p92", 0.85)
    if final_stress < p40:
        return "Recovery Run"
    if final_stress < p65:
        return "Easy Run"
    if final_stress < p80:
        return "Steady Run"
    if final_stress < p92:
        return "Tempo Run"
    return "Interval Session"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def process(
    *,
    # User edit inputs (optional subset)
    distance_miles: Optional[float] = None,
    duration_minutes: Optional[float] = None,
    hr_zone: Optional[str] = None,
    # Context (required)
    baseline_pace: float,
    fatigue_today: float,
    today_recommended_stress: float,
    original_recommended_distance_mi: float,
    # Stress prediction and athlete profile
    stress_predictor: Callable[[float, float, float], float],
    athlete_profile: Dict[str, float],
    # Route regeneration
    lat: float,
    lon: float,
    radius_m: float = 3500,
    **route_kw: Any,
) -> Dict[str, Any]:
    """
    Run the Manual Edit pipeline and return route + metadata.

    Priority: hr_zone > duration > distance (for resolving conflicts).
    Returns: route (polyline), workout_type, predicted_stress, requested_stress, was_adjusted, message.
    """
    # ---- STEP 1: Resolve inputs (priority: hr_zone > duration > distance) ----
    use_hr_zone = hr_zone is not None and str(hr_zone).strip()
    use_duration = duration_minutes is not None and duration_minutes > 0
    use_distance = distance_miles is not None and distance_miles > 0

    # ---- STEP 2: Determine target pace ----
    if use_hr_zone:
        zone = str(hr_zone).strip().lower()
        offset_sec = ZONE_PACE_OFFSET_SEC.get(zone, 60)
        target_pace = baseline_pace + (offset_sec / 60.0)  # min/mi
    elif use_duration and use_distance:
        target_pace = duration_minutes / distance_miles  # min/mi
    else:
        target_pace = baseline_pace

    # ---- STEP 3: Determine target distance ----
    if use_distance:
        target_distance_mi = distance_miles
    else:
        target_distance_mi = original_recommended_distance_mi

    # ---- STEP 4: Compute requested stress ----
    requested_stress = stress_predictor(target_distance_mi, target_pace, fatigue_today)

    # ---- STEP 5: Safety clamp ----
    max_safe = today_recommended_stress * 1.25
    min_useful = today_recommended_stress * 0.55
    final_stress = _clamp(requested_stress, min_useful, max_safe)
    was_adjusted = requested_stress != final_stress

    # ---- STEP 6: Reclassify workout type ----
    workout_type = _reclassify_workout_type(final_stress, athlete_profile)

    # ---- STEP 7: Regenerate route ----
    from route_builder.builder import generate_route

    route_result = generate_route(
        lat,
        lon,
        workout_type=workout_type,
        target_stress=final_stress,
        target_distance_mi=target_distance_mi,
        baseline_pace_min_per_mi=baseline_pace,
        fatigue_today=fatigue_today,
        stress_predictor=stress_predictor,
        radius_m=radius_m,
        **route_kw,
    )

    # ---- STEP 8 & 9: Output and UX message ----
    if route_result.get("error"):
        return {
            "route": "",
            "workout_type": workout_type,
            "predicted_stress": round(final_stress, 3),
            "requested_stress": round(requested_stress, 3),
            "was_adjusted": was_adjusted,
            "message": f"Could not build route: {route_result['error']}",
            "error": route_result["error"],
            **{k: route_result[k] for k in ("distance", "elevation_gain", "intersections", "scenic_percent") if k in route_result},
        }

    message = (
        "Adjusted slightly to keep today's training safe."
        if was_adjusted
        else "Updated to match your request."
    )

    return {
        "route": route_result.get("polyline", ""),
        "workout_type": workout_type,
        "predicted_stress": round(final_stress, 3),
        "requested_stress": round(requested_stress, 3),
        "was_adjusted": was_adjusted,
        "message": message,
        "distance": route_result.get("distance", 0),
        "elevation_gain": route_result.get("elevation_gain", 0),
        "intersections": route_result.get("intersections", 0),
        "scenic_percent": route_result.get("scenic_percent", 0),
        "start_offset_m": route_result.get("start_offset_m"),
    }
