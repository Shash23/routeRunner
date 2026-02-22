#!/usr/bin/env python3
"""
Test the Manual Edit system.

Run from project root:
  python test_manual_edit.py [user.csv]

Uses sample_data/user1.csv by default. Requires global_model.pkl and network
(for route builder). Optionally saves route_map.html.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core-model"))

# Import after path setup
import personal_predictor
from workout_interpreter import build_athlete_profile
from manual_edit import process


def get_location():
    """IP-based location or default."""
    try:
        import requests
        r = requests.get("https://ipapi.co/json/", timeout=5)
        r.raise_for_status()
        d = r.json()
        return float(d["latitude"]), float(d["longitude"])
    except Exception:
        return 43.07, -89.45  # default


def main():
    user_csv = ROOT / "sample_data" / "user1.csv"
    if len(sys.argv) > 1:
        user_csv = Path(sys.argv[1])
    if not user_csv.is_file():
        print(f"CSV not found: {user_csv}", file=sys.stderr)
        sys.exit(1)

    global_path = ROOT / "global_model.pkl"
    if not global_path.is_file():
        print("Run the global model first (e.g. ./run) to create global_model.pkl", file=sys.stderr)
        sys.exit(1)

    print("Loading global model and user data...")
    global_model = personal_predictor.load_global_model(global_path)
    df, baseline_pace, fatigue_today = personal_predictor.process_user_csv(user_csv)
    personal_model, run_count = personal_predictor.train_personal_model(df)
    alpha = personal_predictor.personalization_weight(run_count)

    stress_predictor = lambda d, p, f: personal_predictor.predict_final_stress(
        d, p, f, global_model, personal_model, alpha, baseline_pace
    )
    # build_athlete_profile needs df with "pace"; personal_predictor returns df with "intensity" (pace = baseline_pace / intensity)
    df_for_profile = df.copy()
    df_for_profile["pace"] = baseline_pace / df_for_profile["intensity"].replace(0, 1)
    athlete_profile = build_athlete_profile(df_for_profile, stress_predictor)

    lat, lon = get_location()
    today_recommended = 0.65
    original_distance = 3.5

    print("Calling manual_edit.process(...)")
    result = process(
        distance_miles=3.5,
        hr_zone="Easy",
        baseline_pace=baseline_pace,
        fatigue_today=fatigue_today,
        today_recommended_stress=today_recommended,
        original_recommended_distance_mi=original_distance,
        stress_predictor=stress_predictor,
        athlete_profile=athlete_profile,
        lat=lat,
        lon=lon,
        radius_m=max(1500, int(3.5 * 1609.34 * 0.65)),
    )

    print("\n--- Manual Edit Result ---")
    print("workout_type:", result["workout_type"])
    print("predicted_stress:", result["predicted_stress"])
    print("requested_stress:", result["requested_stress"])
    print("was_adjusted:", result["was_adjusted"])
    print("message:", result["message"])
    print("distance:", result.get("distance"), "mi")
    if result.get("error"):
        print("error:", result["error"])
    print()

    if result.get("route"):
        try:
            from route_builder.view_route import save_route_map
            # view_route expects dict with "polyline"
            map_result = {"polyline": result["route"], "distance": result.get("distance", 0)}
            out_path = save_route_map(map_result, str(ROOT / "route_map.html"))
            print("Map saved:", out_path)
        except Exception as e:
            print("Could not save map:", e)
    else:
        print("No route polyline; skipping map save.")

    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
