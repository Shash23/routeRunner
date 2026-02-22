# State of the Backend APIs

Summary of what exists in `backend/` and how it fits together.

---

## Overview

The backend is a **FastAPI** app that:

1. Serves a simple frontend (HTML) and health check.
2. Handles **Strava OAuth** (connect, callback, token storage).
3. Ties a **session** (cookie) to an athlete after login.
4. **Trains a personal model** per athlete (async, after first Strava connect) using Strava activities and the core-model personal predictor.
5. Exposes **coach workflow APIs:** `/me`, `/recommendation/today`, `/route`, `/recommendation/adjust`, plus `/predict_next_stress` for debugging.

**Implemented:** Route builder, manual edit, today's recommendation, and athlete context (baseline_pace, fatigue_today, profile thresholds) are wired. See **docs/API_REFERENCE.md** for the full API summary.

---

## Files and Roles

| File | Purpose |
|------|--------|
| **Server.py** | FastAPI app: routes, Strava OAuth, session cookie, calls into athlete_model_manager. Entry: `uvicorn` on port 8000. |
| **StravaAPI.py** | Standalone script for **file-based** Strava tokens (`.local.env`). Used for ad-hoc API testing; **not** used by the FastAPI app. The app uses **strava_token_db** (per-athlete DB tokens) instead. |
| **strava_token_db.py** | SQLAlchemy + SQLite: store/retrieve Strava OAuth tokens by `athlete_id`. `save_token`, `get_token`. DB file: `backend/strava_tokens.db`. |
| **session_id_manager.py** | In-memory session store: `session_id` → `(athlete_id, expires_at)`. 10-minute timeout. Sets/reads HTTP-only cookie for the frontend. |
| **athlete_model_manager.py** | Fetches Strava activities, builds DataFrame, calls core-model `process_user_csv` / `train_personal_model` / `predict_final_stress` and `load_global_model`. Persists per-athlete model as `backend/models/{athlete_id}.pkl` and metadata as `{athlete_id}.json` (includes `run_count`, `profile_thresholds`). |
| **athlete_context.py** | Loads athlete from session + disk: `get_athlete_context(session_id)` → athlete_id, model, baseline_pace, fatigue_today, profile_thresholds, alpha, run_count. No Strava calls. |
| **templates/index.html** | Single page: “Connect with Strava” button and “Predict next stress” (5 mi @ 8 min/mi) with credentials so the session cookie is sent. |

---

## API Endpoints (Current)

| Method | Path | Purpose |
|--------|------|--------|
| GET | `/` | Serves `templates/index.html`. |
| GET | `/health` | Returns `{"status": "ok"}`. |
| GET | `/auth/strava` | Redirects to Strava OAuth authorization URL. |
| GET | `/auth/strava/callback?code=...` | Exchanges `code` for tokens, saves via `strava_token_db`, creates session, redirects to `/`, and **starts async** `train_athlete_model(athlete_id)`. |
| GET | `/me` | Session required. Returns `model_ready`, `baseline_pace`, `fatigue_today`, `training_profile`, `confidence`. 401 if not connected, 503 if model still training. |
| GET | `/recommendation/today` | Session required. Today's recommendation (readiness, workout_type, target_stress, distance_miles, pace_range, hr_zone, why). No route. Cached for `/route`. |
| GET | `/route?lat=&lon=&for=today` | Session required. Generates route for today at lat/lon (default Madison). Returns polyline, distance_miles, elevation_gain, predicted_stress, intersections, surface. 500 if route fails. |
| POST | `/recommendation/adjust` | Session required. Body: `distance_miles`, `duration_minutes`, `hr_zone` (optional). Returns adjusted workout + polyline (manual-edit flow). |
| GET | `/predict_next_stress?distance=5&pace=8` | Session required. Returns `{"stress": <float>}` or `{"error": "..."}`. For debugging; not part of coach workflow. |

Full request/response shapes and error codes: **docs/API_REFERENCE.md**.

---

## Dependencies and Config

- **Backend** uses: FastAPI, uvicorn, stravalib, python-dotenv, pandas, scikit-learn, joblib, SQLAlchemy. There is no single `backend/requirements.txt`; these are spread across the repo (e.g. core-model, or implied by imports).
- **Config:** `backend/.local.env` (or similar) must define `STRAVA_CLIENT_ID`, `STRAVA_CLIENT_SECRET`; optional `STRAVA_REDIRECT_URI` (default `http://localhost:8000/auth/strava/callback`).
- **Global model:** `athlete_model_manager.predict_next_stress` and `train_athlete_model` (via `load_global_model()`) assume `global_model.pkl` exists at **project root** (same as when running `./run` or `run.py`).

---

## Current Gaps / Limitations

1. **Session storage** — Sessions are in-memory; restart wipes all sessions (users must re-connect Strava).
2. **Token refresh** — Callback saves tokens; there is no explicit refresh in the request path when calling Strava (e.g. in `train_athlete_model`). If the access token is expired, the app would need to refresh using `refresh_token` (stravalib supports this); not clearly wired.
3. **StravaAPI.py vs Server** — Two patterns: `StravaAPI.py` uses `.local.env` for a single user's tokens; `Server.py` uses the DB and session for multi-athlete. They are independent.
4. **CWD and global model** — Server is typically run from project root so that `global_model.pkl` and `core-model` resolve; running from `backend/` may require PYTHONPATH.

---

## Capabilities now connected to the API

| Capability | Endpoint(s) |
|------------|-------------|
| **Route generation** | GET `/route` (lat, lon, uses today's recommendation). |
| **Exact-distance route** | Used inside `/route` and `/recommendation/adjust` via `generate_route(target_distance_mi=...)`. |
| **Manual edit** | POST `/recommendation/adjust` (body: distance_miles, duration_minutes, hr_zone). |
| **Today's recommendation / prescription** | GET `/recommendation/today` (readiness, workout_type, target_stress, distance_miles, pace_range, hr_zone, why). |
| **Athlete profile (baseline_pace, fatigue_today, thresholds)** | GET `/me` (baseline_pace, fatigue_today, training_profile, confidence); profile thresholds used internally for recommendation and adjust. |
| **Stress for (distance, pace)** | GET `/predict_next_stress` (debug). |

**Still not exposed:** Route map (HTML), global model training (one-off via `./run`).

---

## Summary

The backend provides **Strava OAuth**, per-athlete token storage, session cookie, one-off personal model training after connect, and a **coach workflow API**: `/me`, `/recommendation/today`, `/route`, `/recommendation/adjust`, plus `/predict_next_stress` for debugging. Route generation, manual edit, and today's recommendation are implemented. See **docs/API_REFERENCE.md** for the full API list and request/response shapes.
