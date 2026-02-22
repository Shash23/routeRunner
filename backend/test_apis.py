#!/usr/bin/env python3
"""
Test coach workflow APIs: /me, /recommendation/today, /route, /recommendation/adjust.
Bootstraps a test athlete from sample_data/user1.csv so no Strava connection is needed.
Run from project root: python backend/test_apis.py
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core-model"))

# Bootstrap test athlete (99999) from sample_data/user1.csv
def bootstrap_test_athlete():
    import json
    import joblib
    from personal_predictor import (
        load_global_model,
        predict_final_stress,
        process_user_csv,
        train_personal_model,
        personalization_weight,
    )
    from workout_interpreter import build_athlete_profile

    csv_path = ROOT / "sample_data" / "user1.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Need {csv_path} to bootstrap test athlete")
    global_path = ROOT / "global_model.pkl"
    if not global_path.is_file():
        raise FileNotFoundError("Need global_model.pkl at project root. Run ./run first.")

    global_model = load_global_model(global_path)
    processed_df, baseline_pace, fatigue_today = process_user_csv(csv_path)
    if processed_df.empty:
        raise RuntimeError("user1.csv produced empty processed df")
    personal_model, run_count = train_personal_model(processed_df)
    alpha = personalization_weight(run_count)
    stress_predictor = lambda d, p, f: predict_final_stress(
        d, p, f, global_model, personal_model, alpha, baseline_pace
    )
    df_for_profile = processed_df.copy()
    df_for_profile["pace"] = baseline_pace / processed_df["intensity"].replace(0, 1)
    profile = build_athlete_profile(df_for_profile, stress_predictor)
    profile_thresholds = {"p40": profile["p40"], "p65": profile["p65"], "p80": profile["p80"], "p92": profile["p92"]}

    models_dir = ROOT / "backend" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    aid = 99999
    metadata = {
        "baseline_pace": baseline_pace,
        "fatigue_today": fatigue_today,
        "alpha": alpha,
        "run_count": run_count,
        "profile_thresholds": profile_thresholds,
    }
    with open(models_dir / f"{aid}.json", "w") as f:
        json.dump(metadata, f)
    joblib.dump(personal_model, models_dir / f"{aid}.pkl")
    print(f"Bootstrapped test athlete {aid}")
    return aid


def run_tests():
    # Bootstrap before importing Server (so backend.models has 99999)
    test_athlete_id = bootstrap_test_athlete()

    # Import after path and bootstrap; backend on path so Server can load
    sys.path.insert(0, str(ROOT / "backend"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("Server", ROOT / "backend" / "Server.py")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    app = server.app
    session_id_manager = server.session_id_manager
    from fastapi.testclient import TestClient

    session_id = session_id_manager.create_session(test_athlete_id)
    client = TestClient(app)
    cookies = {"session_id": session_id}

    errors = []

    # 1. GET /me
    print("Testing GET /me ...")
    r = client.get("/me", cookies=cookies)
    if r.status_code != 200:
        errors.append(f"GET /me: status {r.status_code} body={r.text}")
    else:
        data = r.json()
        if not data.get("model_ready"):
            errors.append(f"GET /me: model_ready not True")
        if "baseline_pace" not in data or "fatigue_today" not in data:
            errors.append(f"GET /me: missing baseline_pace or fatigue_today")
        print("  OK", data.get("training_profile"), "confidence", data.get("confidence"))

    # 2. GET /recommendation/today
    print("Testing GET /recommendation/today ...")
    r = client.get("/recommendation/today", cookies=cookies)
    if r.status_code != 200:
        errors.append(f"GET /recommendation/today: status {r.status_code} body={r.text}")
    else:
        data = r.json()
        for k in ("readiness", "workout_type", "target_stress", "distance_miles", "pace_range", "hr_zone", "why"):
            if k not in data:
                errors.append(f"GET /recommendation/today: missing {k}")
        print("  OK", data.get("workout_type"), data.get("distance_miles"), "mi")

    # 3. GET /route (may take 2-3s, uses Overpass)
    print("Testing GET /route (may take a few seconds) ...")
    r = client.get("/route", cookies=cookies, params={"lat": 43.07, "lon": -89.45}, timeout=15)
    if r.status_code != 200:
        errors.append(f"GET /route: status {r.status_code} body={r.text[:200]}")
    else:
        data = r.json()
        if "polyline" not in data:
            errors.append("GET /route: missing polyline")
        if data.get("polyline") and len(data["polyline"]) < 10:
            errors.append("GET /route: polyline looks empty or stub")
        print("  OK polyline len=%d distance=%.2f mi" % (len(data.get("polyline", "")), data.get("distance_miles", 0)))

    # 4. POST /recommendation/adjust
    print("Testing POST /recommendation/adjust ...")
    r = client.post("/recommendation/adjust", cookies=cookies, json={"distance_miles": 4.0}, timeout=15)
    if r.status_code != 200:
        errors.append(f"POST /recommendation/adjust: status {r.status_code} body={r.text[:200]}")
    else:
        data = r.json()
        for k in ("was_adjusted", "message", "workout_type", "distance_miles", "predicted_stress", "polyline"):
            if k not in data:
                errors.append(f"POST /recommendation/adjust: missing {k}")
        print("  OK", data.get("workout_type"), data.get("message"))

    # 5. No session -> 401
    print("Testing 401 without session ...")
    r = client.get("/me")
    if r.status_code != 401:
        errors.append(f"GET /me without cookie: expected 401 got {r.status_code}")

    if errors:
        print("\nFAILED:")
        for e in errors:
            print(" ", e)
        return 1
    print("\nAll API tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run_tests())
