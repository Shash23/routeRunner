import asyncio
import base64
import json
import os
import time
import sys
from pathlib import Path

# Ensure backend dir is on path so "backend.Server" can import backend modules when run from project root
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from typing import Optional

from fastapi import Body, Cookie, FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from stravalib import Client

import session_id_manager
import athlete_model_manager
import athlete_context
from training_state_store import get_state, get_run_count, TrainingState

try:
    from backend import strava_token_db
except ImportError:
    import strava_token_db

# Project root and core-model for manual_edit, route_builder, workout_interpreter
_ROOT = Path(__file__).resolve().parent.parent
_CORE_MODEL_DIR = _ROOT / "core-model"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_CORE_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_MODEL_DIR))

from personal_predictor import load_global_model, predict_final_stress
from workout_interpreter import build_prescription, build_full_recommendation
from recommendation_explainer import build_explanation

# In-memory cache: athlete_id -> { last_recommendation, last_route }
athlete_runtime_state: dict = {}

# Default location (Madison WI)
_DEFAULT_LAT, _DEFAULT_LON = 43.07, -89.45

_BACKEND_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _BACKEND_DIR / "templates"
_ENV_PATH = _BACKEND_DIR / ".local.env"

app = FastAPI(title="RouteRunner API")
_STATIC_DIR = _BACKEND_DIR / "static"


@app.get("/static/styles.css")
def get_styles():
    return FileResponse(_STATIC_DIR / "styles.css", media_type="text/css")


@app.get("/static/app.js")
def get_app_js():
    return FileResponse(_STATIC_DIR / "app.js", media_type="application/javascript")


def _load_strava_env():
    load_dotenv(_ENV_PATH)
    client_id = os.getenv("STRAVA_CLIENT_ID")
    client_secret = os.getenv("STRAVA_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET must be set in .local.env")
    return client_id, client_secret


@app.get("/")
async def root():
    return FileResponse(_TEMPLATES_DIR / "index.html")


@app.get("/index.js")
async def serve_index_js():
    return FileResponse(_TEMPLATES_DIR / "index.js", media_type="application/javascript")


@app.get("/signout")
async def signout(session_id: str | None = Cookie(None)):
    """Clear session and redirect to home (sign-in popup will show)."""
    session_id_manager.delete_session(session_id)
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(session_id_manager.SESSION_COOKIE_NAME)
    response.delete_cookie("athlete")
    return response


@app.get("/health")
async def health():
    return {"status": "ok"}


# Chrome DevTools requests this when DevTools is open; return 204 to avoid 404 in logs
@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_well_known():
    return Response(status_code=204)


@app.get("/signin")
async def signin(session_id: str | None = Cookie(None)):
    """
    Sign-in entry point (use this from the app). If the application needs scope
    ("read", "activity:read") for the user, redirects to Strava OAuth; otherwise
    redirects to home.
    """
    #if session_id_manager.get_athlete_id_from_session(session_id) is not None:
    #    return RedirectResponse(url="/", status_code=302)
    return RedirectResponse(url="/auth/strava", status_code=302)


@app.get("/auth/strava")
async def auth_strava():
    """Redirect user to Strava to authorize the app."""
    print("Auth redirect...", flush=True)
    client_id, _ = _load_strava_env()
    redirect_uri = os.getenv("STRAVA_REDIRECT_URI", "http://localhost:8000/auth/strava/callback")
    client = Client()
    url = client.authorization_url(
        client_id=int(client_id),
        redirect_uri=redirect_uri,
        scope=["read", "activity:read"],
        approval_prompt="auto",
    )
    return RedirectResponse(url=url)


@app.get("/auth/strava/callback")
async def auth_strava_callback(code: str = Query(..., description="Authorization code from Strava")):
    """Exchange authorization code for tokens and persist to strava_token_db."""
    print("Auth callback...", flush=True)
    client_id, client_secret = _load_strava_env()
    client = Client()
    token_response = client.exchange_code_for_token(
        client_id=int(client_id),
        client_secret=client_secret,
        code=code,
    )
    athlete_id = token_response.get("athlete", {}).get("id")
    if athlete_id is None:
        client.access_token = token_response["access_token"]
        athlete_id = client.get_athlete().id
    strava_token_db.save_token(
        athlete_id=athlete_id,
        access_token=token_response["access_token"],
        refresh_token=token_response["refresh_token"],
        expires_at=token_response["expires_at"],
    )
    strava_expires_at = token_response.get("expires_at")
    session_id = session_id_manager.create_session(athlete_id, expires_at=strava_expires_at)
    response = RedirectResponse(url="/", status_code=302)
    if strava_expires_at is not None:
        max_age = max(1, int(strava_expires_at - time.time()))
    else:
        max_age = None
    session_id_manager.set_session_cookie(response, session_id, max_age=max_age)

    # Set name and profile URL in cookies so the frontend can show them without calling
    client.access_token = token_response["access_token"]
    athlete = client.get_athlete()
    name = f"{athlete.firstname} {athlete.lastname}".strip() or "Athlete"
    profile_url = getattr(athlete, "profile_medium", None) or getattr(athlete, "profile", None) or ""
    payload = json.dumps({"name": name, "profile_url": profile_url})
    response.set_cookie("athlete", base64.b64encode(payload.encode()).decode(), max_age=max_age or 600, httponly=False, samesite="lax")

    task = asyncio.create_task(athlete_model_manager.train_athlete_model(athlete_id))

    def _log_task_exception(t: asyncio.Task) -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            print(f"create_athlete_profile failed: {exc}", flush=True)

    task.add_done_callback(_log_task_exception)

    return response


def _require_context(session_id: str | None):
    """Return (None, 401) if not connected, (None, 503) if model still training, else (ctx, None)."""
    athlete_id = session_id_manager.get_athlete_id_from_session(session_id)
    if athlete_id is None:
        return None, 401
    ctx = athlete_context.get_athlete_context(session_id)
    if ctx is None:
        return None, 503
    return ctx, None


@app.get("/training/status")
async def training_status(session_id: str | None = Cookie(None)):
    """Training status for the connected athlete. 401 if not connected."""
    athlete_id = session_id_manager.get_athlete_id_from_session(session_id)
    if not athlete_id:
        return JSONResponse(status_code=401, content={"error": "Not connected"})

    state = get_state(athlete_id)

    if state is None:
        return {"state": "training", "runs_loaded": 0, "message": "Initializing profile..."}

    if state == TrainingState.TRAINING:
        return {"state": "training", "runs_loaded": 0, "message": "Building your training profile..."}

    if state == TrainingState.FAILED:
        return {"state": "failed", "runs_loaded": 0, "message": "Training failed"}

    return {
        "state": "ready",
        "runs_loaded": get_run_count(athlete_id),
        "message": "Model ready",
    }


@app.get("/me")
async def me(session_id: str | None = Cookie(None)):
    """Confirm personalization ready. 401 not connected, 503 model still training."""
    ctx, err = _require_context(session_id)
    if err == 401:
        return JSONResponse({"error": "Not connected. Connect with Strava first."}, status_code=401)
    if err == 503:
        return JSONResponse({"error": "Model still training. Try again in a minute."}, status_code=503)
    imp = ctx["model"].feature_importances_
    distance_imp = float(imp[0]) if len(imp) > 0 else 0.5
    intensity_imp = float(imp[1]) if len(imp) > 1 else 0.5
    training_profile = "distance-sensitive" if distance_imp > intensity_imp else "intensity-sensitive"
    confidence = min(1.0, ctx["run_count"] / 120.0)
    return {
        "model_ready": True,
        "baseline_pace": ctx["baseline_pace"],
        "fatigue_today": ctx["fatigue_today"],
        "training_profile": training_profile,
        "confidence": round(confidence, 3),
    }


def _build_stress_predictor(ctx):
    """Return (distance, pace, fatigue_today) -> stress for this athlete."""
    global_model = load_global_model()
    def predict(d, p, f):
        return predict_final_stress(
            d, p, f, global_model, ctx["model"], ctx["alpha"], ctx["baseline_pace"]
        )
    return predict


def _athlete_profile_from_ctx(ctx):
    """Profile dict for workout_interpreter (p40, p65, p80, p92, median_pace, median_distance)."""
    pt = ctx["profile_thresholds"]
    return {
        "p40": pt["p40"], "p65": pt["p65"], "p80": pt["p80"], "p92": pt["p92"],
        "median_pace": ctx["baseline_pace"],
        "median_distance": 3.5,
    }


def _get_recommendation_today(ctx):
    """Build today's recommendation; cache in athlete_runtime_state."""
    aid = ctx["athlete_id"]
    if aid in athlete_runtime_state and athlete_runtime_state[aid].get("last_recommendation"):
        return athlete_runtime_state[aid]["last_recommendation"]

    profile = _athlete_profile_from_ctx(ctx)
    predict_stress = _build_stress_predictor(ctx)
    distance = 3.5
    pace = ctx["baseline_pace"] + 60 / 60.0
    fatigue_today = ctx["fatigue_today"]

    prescription = build_prescription(
        distance, pace, fatigue_today, predict_stress, profile
    )
    full = build_full_recommendation(
        prescription, distance, pace, fatigue_today, profile,
        speed_sensitive=False, fatigue_p65=4.0,
    )

    rec = {
        "readiness": full["recovery_status"],
        "workout_type": full["workout_type"],
        "target_stress": full["predicted_stress"],
        "distance_miles": distance,
        "pace_range": full["pace_range"],
        "hr_zone": full["training_zone"],
        "why": full.get("why_this_run", []),
    }
    if aid not in athlete_runtime_state:
        athlete_runtime_state[aid] = {}
    athlete_runtime_state[aid]["last_recommendation"] = rec
    return rec


@app.get("/recommendation/explain")
async def explain_recommendation(session_id: str | None = Cookie(None)):
    """Explain why today's recommendation was chosen. 401 not connected, 503 model still training."""
    ctx = athlete_context.get_athlete_context(session_id)
    if ctx is None:
        athlete_id = session_id_manager.get_athlete_id_from_session(session_id)
        if athlete_id is None:
            return JSONResponse(status_code=401, content={"error": "Not connected"})
        return JSONResponse(status_code=503, content={"error": "Model still training"})

    rec = _get_recommendation_today(ctx)
    explanation = build_explanation(ctx, rec)
    return explanation


@app.get("/recommendation/today")
async def recommendation_today(session_id: str | None = Cookie(None)):
    """Today's recommendation. No route. 401/503 on auth/model."""
    ctx, err = _require_context(session_id)
    if err == 401:
        return JSONResponse({"error": "Not connected. Connect with Strava first."}, status_code=401)
    if err == 503:
        return JSONResponse({"error": "Model still training. Try again in a minute."}, status_code=503)
    rec = _get_recommendation_today(ctx)
    return rec


@app.get("/route")
async def route(
    session_id: str | None = Cookie(None),
    lat: float = Query(_DEFAULT_LAT, description="Latitude"),
    lon: float = Query(_DEFAULT_LON, description="Longitude"),
    for_today: str = Query("today", alias="for", description="for=today"),
):
    """Generate route for today's recommendation. 401/503/500."""
    ctx, err = _require_context(session_id)
    if err == 401:
        return JSONResponse({"error": "Not connected. Connect with Strava first."}, status_code=401)
    if err == 503:
        return JSONResponse({"error": "Model still training. Try again in a minute."}, status_code=503)

    rec = _get_recommendation_today(ctx)
    predict_stress = _build_stress_predictor(ctx)

    try:
        from route_builder.builder import generate_route
    except Exception as e:
        print(f"Route builder import failed: {e}", flush=True)
        return JSONResponse({"error": "Route generation failed."}, status_code=500)

    try:
        result = generate_route(
            lat,
            lon,
            workout_type=rec["workout_type"],
            target_stress=rec["target_stress"],
            target_distance_mi=rec["distance_miles"],
            baseline_pace_min_per_mi=ctx["baseline_pace"],
            fatigue_today=ctx["fatigue_today"],
            stress_predictor=predict_stress,
            radius_m=max(1500, int(rec["distance_miles"] * 1609.34 * 0.65)),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

    # No route found (e.g. no OSM data, no start node) — return 200 with error message so client can show it
    if result.get("error"):
        return {
            "polyline": result.get("polyline", ""),
            "distance_miles": result.get("distance", 0),
            "elevation_gain": result.get("elevation_gain", 0),
            "predicted_stress": result.get("predicted_stress", 0),
            "intersections": result.get("intersections", 0),
            "surface": "mixed",
            "error": result.get("error"),
        }

    aid = ctx["athlete_id"]
    if aid not in athlete_runtime_state:
        athlete_runtime_state[aid] = {}
    athlete_runtime_state[aid]["last_route"] = result

    return {
        "polyline": result.get("polyline", ""),
        "distance_miles": result.get("distance", 0),
        "elevation_gain": result.get("elevation_gain", 0),
        "predicted_stress": result.get("predicted_stress", 0),
        "intersections": result.get("intersections", 0),
        "surface": "mixed",
    }


class AdjustBody(BaseModel):
    distance_miles: Optional[float] = None
    duration_minutes: Optional[float] = None
    hr_zone: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


@app.post("/recommendation/adjust")
async def recommendation_adjust(
    session_id: str | None = Cookie(None),
    body: Optional[AdjustBody] = Body(default=None),
):
    """Adjust recommendation by distance/duration/hr_zone; return new route. 401/503/500."""
    ctx, err = _require_context(session_id)
    if err == 401:
        return JSONResponse({"error": "Not connected. Connect with Strava first."}, status_code=401)
    if err == 503:
        return JSONResponse({"error": "Model still training. Try again in a minute."}, status_code=503)

    rec = _get_recommendation_today(ctx)
    today_stress = rec["target_stress"]
    original_distance = rec["distance_miles"]
    b = body or AdjustBody()
    lat = b.lat if b.lat is not None else _DEFAULT_LAT
    lon = b.lon if b.lon is not None else _DEFAULT_LON

    athlete_profile = _athlete_profile_from_ctx(ctx)
    predict_stress = _build_stress_predictor(ctx)

    try:
        from manual_edit import process as manual_edit_process
    except Exception:
        return JSONResponse({"error": "Route generation failed."}, status_code=500)

    try:
        result = manual_edit_process(
            distance_miles=b.distance_miles,
            duration_minutes=b.duration_minutes,
            hr_zone=b.hr_zone,
            baseline_pace=ctx["baseline_pace"],
            fatigue_today=ctx["fatigue_today"],
            today_recommended_stress=today_stress,
            original_recommended_distance_mi=original_distance,
            stress_predictor=predict_stress,
            athlete_profile=athlete_profile,
            lat=lat,
            lon=lon,
            radius_m=3500,
        )
    except Exception:
        return JSONResponse({"error": "Route generation failed."}, status_code=500)

    if result.get("error"):
        return JSONResponse({"error": "Route generation failed."}, status_code=500)

    return {
        "was_adjusted": result.get("was_adjusted", False),
        "message": result.get("message", ""),
        "workout_type": result.get("workout_type", ""),
        "distance_miles": result.get("distance", 0),
        "predicted_stress": result.get("predicted_stress", 0),
        "polyline": result.get("route", ""),
    }


@app.get("/predict_next_stress")
async def predict_next_stress(
    session_id: str | None = Cookie(None),
    distance: float = Query(5.0, description="Distance in miles"),
    pace: float = Query(8.0, description="Pace in min/mi"),
):
    """Predict the next stress of the run for the athlete (session from cookie or invalid)."""
    athlete_id = session_id_manager.get_athlete_id_from_session(session_id)
    if athlete_id is None:
        return {"error": "Invalid or missing session. Connect with Strava first."}
    stress = await athlete_model_manager.predict_next_stress(athlete_id, distance, pace)
    return {"stress": stress}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
