
import secrets
import time
from fastapi.responses import Response

SESSION_COOKIE_NAME = "session_id"
SESSION_TIMEOUT_SECONDS = 600  # 10 minutes
_sessions: dict[str, tuple[int, float]] = {}  # session_id -> (athlete_id, expires_at)


def create_session(athlete_id: int) -> str:
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


def set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=SESSION_TIMEOUT_SECONDS,
        httponly=True,
        samesite="lax",
    )