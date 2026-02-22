import asyncio
import os
import secrets
import time
from pathlib import Path
import datetime

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, RedirectResponse
from stravalib import Client

try:
    from backend import strava_token_db
except ImportError:
    import strava_token_db

app = FastAPI(title="RouteRunner API")


_ACTIVITY_FURTHEST_IN_DAYS = 365

_BACKEND_DIR = Path(__file__).resolve().parent
_CORE_MODEL_DIR = _BACKEND_DIR.parent / "core-model"
_TEMPLATES_DIR = _BACKEND_DIR / "templates"
_ENV_PATH = _BACKEND_DIR / ".local.env"

SESSION_COOKIE_NAME = "session_id"
SESSION_TIMEOUT_SECONDS = 600  # 10 minutes
_sessions: dict[str, tuple[int, float]] = {}  # session_id -> (athlete_id, expires_at)


def _create_session(athlete_id: int) -> str:
    session_id = secrets.token_urlsafe(32)
    expires_at = time.time() + SESSION_TIMEOUT_SECONDS
    _sessions[session_id] = (athlete_id, expires_at)
    return session_id


def get_athlete_id_from_session(session_id: str | None) -> int | None:
    """Return athlete_id for a valid session, or None. Removes expired sessions."""
    if not session_id:
        return None
    now = time.time()
    # Prune expired
    expired = [sid for sid, (_, exp) in _sessions.items() if exp <= now]
    for sid in expired:
        del _sessions[sid]
    entry = _sessions.get(session_id)
    if entry is None:
        return None
    athlete_id, expires_at = entry
    if expires_at <= now:
        del _sessions[session_id]
        return None
    return athlete_id


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


@app.get("/health")
async def health():
    return {"status": "ok"}


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
    session_id = _create_session(athlete_id)
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=SESSION_TIMEOUT_SECONDS,
        httponly=True,
        samesite="lax",
    )
    task = asyncio.create_task(create_athlete_profile(athlete_id))

    def _log_task_exception(t: asyncio.Task) -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            print(f"create_athlete_profile failed: {exc}", flush=True)

    task.add_done_callback(_log_task_exception)

    return response


METERS_PER_MILE = 1609.344


def _normalize_activity_type(act_type) -> str:
    """Get a plain string from stravalib type (may be object with root='Run' or .value)."""
    if act_type is None:
        return ""
    if hasattr(act_type, "root"):
        return str(getattr(act_type, "root", "") or "")
    if hasattr(act_type, "value"):
        return str(getattr(act_type, "value", "") or "")
    return str(act_type).strip()


def _activities_to_dataframe(activities) -> pd.DataFrame:
    """Build a DataFrame with columns: Activity Type, Distance (mi), Elapsed Time (s), Activity Date (ISO)."""
    rows = []
    for activity in activities:
        act_type_raw = getattr(activity, "type", None) or getattr(activity, "sport_type", None)
        act_type_str = _normalize_activity_type(act_type_raw)
        if act_type_str.lower() != "run":
            continue
        distance_m = float(activity.distance) if activity.distance is not None else 0.0
        distance_mi = distance_m / METERS_PER_MILE
        elapsed = int(activity.elapsed_time) if activity.elapsed_time is not None else 0
        start = activity.start_date
        activity_date = start.isoformat() if start else ""
        rows.append({
            "Activity Type": "Run",
            "Distance": round(distance_mi, 4),
            "Elapsed Time": elapsed,
            "Activity Date": activity_date,
        })
    return pd.DataFrame(rows, columns=["Activity Type", "Distance", "Elapsed Time", "Activity Date"])


async def create_athlete_profile(athlete_id: int):
    """Create an athlete profile from the Strava data."""
    # Get the strava token from the database
    token = strava_token_db.get_token(athlete_id)
    if token is None:
        print(f"No token found for athlete {athlete_id}", flush=True)
        return

    # Get the Strava data
    client = Client()
    client.access_token = token["access_token"]
    activities = client.get_activities(after=datetime.now() - datetime.timedelta(days=_ACTIVITY_FURTHEST_IN_DAYS))
    df = _activities_to_dataframe(activities)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
