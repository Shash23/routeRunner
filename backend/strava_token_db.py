"""
Store and fetch Strava OAuth tokens per athlete using SQLAlchemy.
Use from other modules: save_token(...), get_token(athlete_id).
"""
from pathlib import Path

from sqlalchemy import String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

_BACKEND_DIR = Path(__file__).resolve().parent
_DB_PATH = _BACKEND_DIR / "strava_tokens.db"
_DB_URL = f"sqlite:///{_DB_PATH}"


class Base(DeclarativeBase):
    pass


class StravaToken(Base):
    __tablename__ = "strava_tokens"

    athlete_id: Mapped[int] = mapped_column(primary_key=True)
    access_token: Mapped[str] = mapped_column(String(256), nullable=False)
    refresh_token: Mapped[str] = mapped_column(String(256), nullable=False)
    expires_at: Mapped[int] = mapped_column(nullable=False)  # Unix timestamp


_engine = create_engine(_DB_URL, echo=False)
_Session = sessionmaker(_engine, expire_on_commit=False)


def create_tables():
    """Create tables if they do not exist. Call once at app startup if needed."""
    Base.metadata.create_all(_engine)


def save_token(
    athlete_id: int,
    access_token: str,
    refresh_token: str,
    expires_at: int,
) -> None:
    """Store or update a Strava token for the given athlete (by Strava athlete id)."""
    create_tables()
    with _Session() as session:
        row = session.get(StravaToken, athlete_id)
        if row:
            row.access_token = access_token
            row.refresh_token = refresh_token
            row.expires_at = expires_at
        else:
            session.add(
                StravaToken(
                    athlete_id=athlete_id,
                    access_token=access_token,
                    refresh_token=refresh_token,
                    expires_at=expires_at,
                )
            )
        session.commit()


def get_token(athlete_id: int) -> dict | None:
    """
    Fetch stored token for the given athlete id.
    Returns None if no token exists, else dict with access_token, refresh_token, expires_at.
    """
    with _Session() as session:
        row = session.get(StravaToken, athlete_id)
        if row is None:
            return None
        return {
            "access_token": row.access_token,
            "refresh_token": row.refresh_token,
            "expires_at": row.expires_at,
        }
