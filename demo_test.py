#!/usr/bin/env python3
"""
End-to-end judge demo workflow test for RolledBadger coaching app.
Simulates: CONNECT → TRAINING → READY → RECOMMEND → ROUTE → ADJUST → EXPLAIN.
Uses requests.Session() for cookie persistence. Run with backend at http://localhost:8000.
"""
import json
import sys
import time

import requests

BASE = "http://localhost:8000"


def section(title):
    print("\n" + "=" * 16 + " " + title + " " + "=" * 16)


def pp(data):
    print(json.dumps(data, indent=2))


def fail(msg):
    print(f"\nFATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    session = requests.Session()

    # --- Step 0: Setup ---
    section("SETUP")
    print("Using persistent session. Base URL:", BASE)

    # --- Step 1: Health ---
    section("CONNECT")
    try:
        r = session.get(f"{BASE}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        fail(f"Cannot connect to {BASE}. Is the backend server running?")
    if r.status_code != 200:
        fail(f"GET /health returned {r.status_code}")
    data = r.json()
    pp(data)
    if data.get("status") != "ok":
        fail("Health check did not return status ok")
    print("Server is healthy.")

    # --- Step 2: OAuth ---
    section("CONNECT")
    auth_url = f"{BASE}/auth/strava"
    print("Auth URL:", auth_url)
    print("\nOpen the URL above in your browser and authorize Strava, then press ENTER\n")
    input()
    print("After authorizing, the backend sets a session cookie in your browser.")
    print("Paste the 'session_id' cookie value here (from DevTools → Application → Cookies → localhost:8000), or press ENTER to try without:")
    cookie_val = input().strip()
    if cookie_val:
        session.cookies.set("session_id", cookie_val, domain="localhost")
        print("Cookie set. Continuing with session.")
    else:
        print("No cookie pasted. Next requests may get 401 if you did not paste the cookie.")

    # --- Step 3: Wait for training ---
    section("TRAINING")
    while True:
        r = session.get(f"{BASE}/training/status")
        if r.status_code == 401:
            fail("Not connected. Paste the session_id cookie from your browser after authorizing.")
        if r.status_code != 200:
            fail(f"GET /training/status returned {r.status_code}")
        data = r.json()
        state = data.get("state", "")
        runs = data.get("runs_loaded", 0)
        msg = data.get("message", "")
        if state == "ready":
            print(f"Ready. (runs_loaded: {runs}) {msg}")
            break
        if state == "failed":
            fail(f"Training failed: {msg}")
        print(f"  Training... (runs_loaded: {runs}) {msg}   ", end="\r", flush=True)
        time.sleep(3)

    # --- Step 4: Athlete profile ---
    section("ATHLETE PROFILE")
    r = session.get(f"{BASE}/me")
    if r.status_code != 200:
        fail(f"GET /me returned {r.status_code}")
    data = r.json()
    print("baseline_pace:", data.get("baseline_pace"))
    print("fatigue_today:", data.get("fatigue_today"))
    print("training_profile:", data.get("training_profile"))
    print("confidence:", data.get("confidence"))

    # --- Step 5: Today's recommendation ---
    section("RECOMMEND")
    r = session.get(f"{BASE}/recommendation/today")
    if r.status_code != 200:
        fail(f"GET /recommendation/today returned {r.status_code}")
    data = r.json()
    print("readiness:", data.get("readiness"))
    print("workout_type:", data.get("workout_type"))
    print("distance_miles:", data.get("distance_miles"))
    print("pace_range:", data.get("pace_range"))
    print("why:")
    for w in data.get("why") or []:
        print("  -", w)
    pp(data)

    # --- Step 6: Route ---
    section("ROUTE")
    r = session.get(f"{BASE}/route", params={"lat": 43.0731, "lon": -89.4012})
    if r.status_code != 200:
        fail(f"GET /route returned {r.status_code}")
    data = r.json()
    print("distance_miles:", data.get("distance_miles"))
    print("predicted_stress:", data.get("predicted_stress"))
    print("intersections:", data.get("intersections"))
    print("surface:", data.get("surface"))
    pl = data.get("polyline") or ""
    print("polyline length:", len(pl))

    # --- Step 7: Manual edit ---
    section("ADJUST")
    body = {"distance_miles": 4.2, "hr_zone": "Easy"}
    r = session.post(f"{BASE}/recommendation/adjust", json=body)
    if r.status_code != 200:
        fail(f"POST /recommendation/adjust returned {r.status_code}")
    data = r.json()
    print("was_adjusted:", data.get("was_adjusted"))
    print("message:", data.get("message"))
    print("workout_type:", data.get("workout_type"))
    print("predicted_stress:", data.get("predicted_stress"))

    # --- Step 8: Explainability ---
    section("EXPLAIN")
    r = session.get(f"{BASE}/recommendation/explain")
    if r.status_code != 200:
        fail(f"GET /recommendation/explain returned {r.status_code}")
    data = r.json()
    print("Why the AI chose this workout:")
    print("  decision:", data.get("decision"))
    print("  fatigue_score:", data.get("fatigue_score"))
    print("  personalization_strength:", data.get("personalization_strength"))
    print("  model_type:", data.get("model_type"))

    # --- Step 9: Debug ---
    section("DEBUG")
    r = session.get(f"{BASE}/predict_next_stress", params={"distance": 5, "pace": 8})
    if r.status_code != 200:
        fail(f"GET /predict_next_stress returned {r.status_code}")
    data = r.json()
    if "stress" in data:
        print("stress (5 mi @ 8 min/mi):", data["stress"])
    else:
        print(data)

    section("DONE")
    print("Demo workflow completed successfully.")


if __name__ == "__main__":
    main()
