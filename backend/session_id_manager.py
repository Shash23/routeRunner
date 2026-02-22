import secrets
import time
from typing import Optional

from fastapi.responses import Response
from stravalib import Client

try:
    from backend import strava_token_db
except ImportError:
    import strava_token_db

SESSION_COOKIE_NAME = "session_id"
# Fallback when no Strava expires_at is provided
SESSION_TIMEOUT_SECONDS = 600  # 10 minutes
# session_id -> (athlete_id, expires_at, client_or_none)
_store: dict[str, tuple[int, float, Optional[Client]]] = {}


def create_session(athlete_id: int, expires_at: Optional[float] = None) -> str:
    """
    Create a session for the athlete. If expires_at (Unix timestamp) is given
    (e.g. from Strava token response), the session uses that expiry; otherwise
    uses SESSION_TIMEOUT_SECONDS from now.
    """
    session_id = secrets.token_urlsafe(32)
    if expires_at is not None:
        session_expires_at = float(expires_at)
    else:
        session_expires_at = time.time() + SESSION_TIMEOUT_SECONDS
    _store[session_id] = (athlete_id, session_expires_at, None)
    return session_id


def get_athlete_id_from_session(session_id: str | None) -> int | None:
    """Return athlete_id for a valid session, or None. Removes expired sessions."""
    if not session_id:
        return None
    now = time.time()
    # Prune expired
    expired = [sid for sid, (_, exp, _) in _store.items() if exp <= now]
    for sid in expired:
        del _store[sid]
    entry = _store.get(session_id)
    if entry is None:
        return None
    athlete_id, expires_at, _ = entry
    if expires_at <= now:
        del _store[session_id]
        return None
    return athlete_id


def delete_session(session_id: str | None) -> None:
    """Remove the session from the store (e.g. on sign out)."""
    if session_id and session_id in _store:
        del _store[session_id]


def get_strava_client(session_id: str | None) -> Client | None:
    """
    Return a Strava Client for the given session, or None. Creates and caches
    the client in the session store when first needed.
    """
    if not session_id:
        return None
    athlete_id = get_athlete_id_from_session(session_id)
    if athlete_id is None:
        return None
    entry = _store.get(session_id)
    if entry is None:
        return None
    athlete_id, expires_at, client = entry
    if client is not None:
        return client
    token = strava_token_db.get_token(athlete_id)
    if token is None:
        return None
    client = Client()
    client.access_token = token["access_token"]
    _store[session_id] = (athlete_id, expires_at, client)
    return client


def set_session_cookie(
    response: Response,
    session_id: str,
    max_age: Optional[int] = None,
) -> None:
    """
    Set the session cookie. If max_age (seconds until expiry) is given
    (e.g. derived from Strava token expires_at), use it; otherwise use
    SESSION_TIMEOUT_SECONDS.
    """
    if max_age is not None and max_age > 0:
        age = max_age
    else:
        age = SESSION_TIMEOUT_SECONDS
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=age,
        httponly=True,
        samesite="lax",
    )