import os
import secrets
import time
from pathlib import Path

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

_BACKEND_DIR = Path(__file__).resolve().parent
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
    return response


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
