# RolledBadger Backend — API Reference

All responses are JSON unless noted. Base URL assumed: `http://localhost:8000`.

---

## Auth & Session

- **Session** is stored in an HTTP-only cookie `session_id` (set after Strava OAuth). Most endpoints require this cookie.
- **401** = not connected (missing or invalid session).
- **503** = model still training (valid session but no personal model yet).

---

## Endpoints

### 1. GET `/`

- **Auth:** None.
- **Response:** HTML (`templates/index.html` — “Connect with Strava” and “Predict next stress”).
- **Purpose:** Frontend landing page.

---

### 2. GET `/health`

- **Auth:** None.
- **Response:** `{ "status": "ok" }`
- **Purpose:** Liveness check.

---

### 3. GET `/auth/strava`

- **Auth:** None.
- **Response:** 302 redirect to Strava OAuth authorization URL.
- **Purpose:** Start “Connect with Strava” flow.

---

### 4. GET `/auth/strava/callback?code=...`

- **Auth:** None (Strava redirects here with `code`).
- **Response:** 302 redirect to `/`, with `session_id` cookie set. Starts async personal model training for the athlete.
- **Purpose:** Exchange `code` for tokens, save tokens, create session, trigger training.

---

### 5. GET `/me`

- **Auth:** Session cookie required.
- **Response (200):**
  ```json
  {
    "model_ready": true,
    "baseline_pace": 8.5,
    "fatigue_today": 2.3,
    "training_profile": "distance-sensitive | intensity-sensitive",
    "confidence": 0.85
  }
  ```
- **Errors:** 401 not connected, 503 model still training.
- **Purpose:** Confirm personalization is ready; baseline pace, fatigue, profile, confidence.

---

### 6. GET `/training/status`

- **Auth:** Session cookie required.
- **Response (200):**
  ```json
  {
    "state": "training | ready | failed",
    "runs_loaded": 0,
    "message": "Initializing profile... | Building your training profile... | Training failed | Model ready"
  }
  ```
- **Errors:** 401 not connected.
- **Purpose:** Let frontend poll whether the athlete model is still training after Strava connect.

---

### 7. GET `/recommendation/today`

- **Auth:** Session cookie required.
- **Response (200):**
  ```json
  {
    "readiness": "Well Recovered | Moderately Recovered | Fatigued",
    "workout_type": "Recovery Run | Easy Run | Base Run | Tempo Run | Interval Workout",
    "target_stress": 0.55,
    "distance_miles": 3.5,
    "pace_range": "9:00–9:30 /mi",
    "hr_zone": "Recovery | Easy | Steady | Tempo | Interval",
    "why": ["bullet 1", "bullet 2"]
  }
  ```
- **Errors:** 401, 503.
- **Purpose:** Today’s recommended workout (no route). Result is cached for `/route`.

---

### 8. GET `/route`

- **Auth:** Session cookie required.
- **Query:**  
  - `lat` (optional, default 43.07)  
  - `lon` (optional, default -89.45)  
  - `for=today` (optional, default)
- **Response (200):**
  ```json
  {
    "polyline": "encoded_polyline_string",
    "distance_miles": 3.5,
    "elevation_gain": 0,
    "predicted_stress": 0.55,
    "intersections": 42,
    "surface": "mixed"
  }
  ```
- **Errors:** 401, 503, 500 (route generation failed).
- **Purpose:** Generate a route for today’s recommendation at the given location. Uses cached recommendation; typically 2–3 s due to Overpass.

---

### 9. POST `/recommendation/adjust`

- **Auth:** Session cookie required.
- **Body (JSON, all optional):**
  ```json
  {
    "distance_miles": 4.0,
    "duration_minutes": 36,
    "hr_zone": "Easy"
  }
  ```
- **Response (200):**
  ```json
  {
    "was_adjusted": false,
    "message": "Updated to match your request. | Adjusted slightly to keep today's training safe.",
    "workout_type": "Easy Run",
    "distance_miles": 4.0,
    "predicted_stress": 0.62,
    "polyline": "encoded_polyline_string"
  }
  ```
- **Errors:** 401, 503, 500.
- **Purpose:** Adjust recommendation by distance/duration/hr_zone; returns new workout + polyline (manual-edit flow).

---

### 10. GET `/recommendation/explain`

- **Auth:** Session cookie required.
- **Response (200):**
  ```json
  {
    "fatigue_score": 0.71,
    "baseline_pace": 8.52,
    "personalization_strength": 0.82,
    "decision": "Reduced intensity to avoid overload",
    "model_type": "personalized + global prior"
  }
  ```
- **Errors:** 401 not connected, 503 model still training.
- **Purpose:** Explain why today’s recommendation was chosen (fatigue, personalization, decision).

---

### 11. GET `/predict_next_stress?distance=5&pace=8`

- **Auth:** Session cookie required.
- **Query:** `distance` (miles), `pace` (min/mi). Defaults 5.0, 8.0.
- **Response (200):** `{ "stress": 0.72 }`
- **Response (200, error):** `{ "error": "Invalid or missing session. Connect with Strava first." }`
- **Purpose:** Single stress prediction for debugging; not part of coach workflow.

---

## Summary Table

| Method | Path | Auth | Purpose |
|--------|------|------|--------|
| GET | `/` | — | Serve frontend HTML |
| GET | `/health` | — | Health check |
| GET | `/auth/strava` | — | Redirect to Strava OAuth |
| GET | `/auth/strava/callback` | — | OAuth callback, set session, start training |
| GET | `/me` | Cookie | Personalization status, baseline, profile, confidence |
| GET | `/training/status` | Cookie | Training state (training / ready / failed), runs_loaded |
| GET | `/recommendation/today` | Cookie | Today’s recommendation (no route) |
| GET | `/recommendation/explain` | Cookie | Why recommendation was chosen (fatigue, decision) |
| GET | `/route` | Cookie | Route for today at lat/lon |
| POST | `/recommendation/adjust` | Cookie | Adjust recommendation, get new route |
| GET | `/predict_next_stress` | Cookie | Predict stress for distance/pace (debug) |

---

## Running the server

From project root (so `global_model.pkl` and `core-model` resolve):

```bash
source .venv/bin/activate
pip install fastapi uvicorn python-dotenv stravalib pandas scikit-learn joblib sqlalchemy httpx  # + route_builder deps if using /route
python -m uvicorn backend.Server:app --host 0.0.0.0 --port 8000
```

Or: `cd backend && PYTHONPATH=.. python -m uvicorn Server:app --host 0.0.0.0 --port 8000`
