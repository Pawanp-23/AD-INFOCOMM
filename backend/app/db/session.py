"""Database engine and session factory (SQLAlchemy 2)."""

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


def _make_engine():
    url = get_settings().database_url
    if url.startswith("sqlite"):
        # Pool sized for FastAPI's 40-thread request pool: with the default 5 + 10 overflow,
        # bursts churned through short-lived connections and throughput collapsed under load.
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False, "timeout": 30},
            pool_size=20,
            max_overflow=20,
        )

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(conn, _record) -> None:
            cur = conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")  # per connection
            cur.execute("PRAGMA synchronous=NORMAL")  # safe with WAL, far fewer fsyncs
            cur.close()

        # WAL (concurrent readers) is a property of the database file: set it once, not on
        # every new connection, where it takes a lock.
        with engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        return engine
    return create_engine(url, pool_pre_ping=True)


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables (local development). Production schema is managed by migrations."""
    from app.db import models  # noqa: F401 - register models

    Base.metadata.create_all(engine)
