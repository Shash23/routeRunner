import asyncio
import os
from pathlib import Path

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, Query
from fastapi.responses import FileResponse, RedirectResponse
from stravalib import Client

import session_id_manager
import athlete_model_manager

try:
    from backend import strava_token_db
except ImportError:
    import strava_token_db

app = FastAPI(title="RouteRunner API")



_BACKEND_DIR = Path(__file__).resolve().parent
_CORE_MODEL_DIR = _BACKEND_DIR.parent / "core-model"
_TEMPLATES_DIR = _BACKEND_DIR / "templates"
_ENV_PATH = _BACKEND_DIR / ".local.env"


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
    session_id = session_id_manager.create_session(athlete_id)
    response = RedirectResponse(url="/", status_code=302)
    session_id_manager.set_session_cookie(response, session_id)

    task = asyncio.create_task(athlete_model_manager.train_athlete_model(athlete_id))

    def _log_task_exception(t: asyncio.Task) -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            print(f"create_athlete_profile failed: {exc}", flush=True)

    task.add_done_callback(_log_task_exception)

    return response

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
