# RolledBadger

**RolledBadger** is a personalized running coach that recommends **what to run today** based on your recovery state. It learns how your body responds to training, then picks workout type, distance, and—when available—a route that matches the right effort for today.

---

## What it does

- **Connect with Strava** → We use your run history (distance, time, date).
- **Personal model** → Fatigue and a stress predictor are trained per athlete.
- **Today’s recommendation** → Readiness, workout type (e.g. Easy Run), distance, pace range, HR zone, and “why” bullets.
- **Route** → Optional map route for the recommended run (OpenStreetMap; can fail on network/Overpass).
- **Manual adjust** → Change distance or HR zone; the coach adjusts safely and re-explains.
- **Explain** → “Why the AI chose this workout” (fatigue score, decision, personalization strength).

Most apps track what you did. RolledBadger suggests **what you should do today** so you stay in the adaptation zone, not the injury zone.

---

## Quick start

**1. Backend (required)**

```bash
cd /path/to/routeRunner
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.local.env` with your [Strava API](https://www.strava.com/settings/api) credentials:

```
STRAVA_CLIENT_ID=your_id
STRAVA_CLIENT_SECRET=your_secret
STRAVA_REDIRECT_URI=http://localhost:8000/auth/strava/callback
```

In Strava, set **Authorization Callback Domain** to `localhost`.

**2. Global model (one-time)**

Train the shared prior (needed for personalization):

```bash
./run
# or: python run.py
```

This produces `global_model.pkl` at the project root.

**3. Run the app**

```bash
python -m uvicorn backend.Server:app --host 0.0.0.0 --port 8000
```

**4. Open in browser**

Go to **http://localhost:8000/** → **Connect with Strava** → authorize → wait for “Building your training profile…” → then see recommendation, map (if route succeeds), and explanation.

---

## Repo layout

| Path | Purpose |
|------|--------|
| **backend/** | FastAPI server, Strava OAuth, session, coach APIs |
| **backend/Server.py** | App entry, routes (`/`, `/health`, `/auth/strava`, `/me`, `/training/status`, `/recommendation/*`, `/route`, etc.) |
| **backend/templates/index.html** | RolledBadger single-page UI |
| **backend/static/** | `styles.css`, `app.js` (fetch, polling, Leaflet map) |
| **backend/.local.env** | Strava credentials (not committed) |
| **core-model/** | Stress prediction, workout interpreter, athlete profile |
| **route_builder/** | OSM/Overpass graph, loop generation, exact-distance routes |
| **manual_edit.py** | Manual-edit pipeline (adjust distance/HR → clamp stress → reclassify → route) |
| **docs/API_REFERENCE.md** | Full API list and request/response shapes |
| **docs/BACKEND_API_STATE.md** | Backend state and gaps |
| **demo_test.py** | End-to-end judge demo script (health → OAuth → training → recommend → route → adjust → explain) |

---

## APIs (summary)

All session-protected endpoints use the `session_id` cookie set after Strava OAuth.

| Method | Path | Purpose |
|--------|------|--------|
| GET | `/` | Frontend (RolledBadger UI) |
| GET | `/health` | Liveness |
| GET | `/auth/strava` | Redirect to Strava OAuth |
| GET | `/auth/strava/callback` | OAuth callback; set cookie; start training |
| GET | `/training/status` | Poll training state (training / ready / failed), runs_loaded |
| GET | `/me` | Athlete profile (baseline_pace, fatigue_today, training_profile, confidence) |
| GET | `/recommendation/today` | Today’s recommendation (no route) |
| GET | `/route` | Route for today at lat/lon (polyline, distance, stress, etc.) |
| POST | `/recommendation/adjust` | Adjust by distance/duration/hr_zone; return new workout + polyline |
| GET | `/recommendation/explain` | Why the recommendation was chosen |
| GET | `/predict_next_stress` | Debug: stress for given distance/pace |

Details: **docs/API_REFERENCE.md**.

---

## Pipeline (concept)

1. **Data** — Strava runs (distance, time, date); optional HR/effort.
2. **Load** — Training load ≈ distance × intensity; intensity from pace vs baseline.
3. **Fatigue** — Decayed sum of recent load → readiness (e.g. fatigue_today).
4. **Personal model** — Predicts stress = f(distance, pace, fatigue) for this athlete.
5. **Recommendation** — Workout interpreter picks workout type, distance, pace range from prescription + profile.
6. **Route** — Route builder finds a run (e.g. target distance) and scores by predicted stress; returns polyline.
7. **Manual edit** — User changes distance/HR → stress clamped → reclassify workout → regenerate route.

---

## Technologies

- **Language:** Python 3
- **Core model:** pandas, scikit-learn (GradientBoostingRegressor), joblib
- **Route builder:** requests, networkx, polyline, geopy; OpenStreetMap (Overpass API)
- **Backend:** FastAPI, uvicorn, stravalib, python-dotenv, SQLAlchemy
- **Frontend:** Plain HTML/CSS/JS, fetch, Leaflet (CDN)
- **Data:** Strava API; pickle/joblib for models (`global_model.pkl`, `backend/models/{id}.pkl`)

---

## Optional: demo script and tests

**Judge-style demo** (server must be running; you authorize in browser and optionally paste session cookie):

```bash
python demo_test.py
```

**API tests** (bootstraps test athlete from `sample_data/user1.csv`; no Strava):

```bash
python backend/test_apis.py
```

---

## License and references

- Methodology aligned with personalized training-response modeling (see project references).
- Strava API used under Strava’s API terms.
