"""SQLite connection and setup for Mirume.

Mirume uses two separate SQLite databases, each with its own SQLAlchemy engine:

* ``jmdict.db``  – the read-mostly JMdict / kanjidic2 dictionary data used for
  JLPT classification. Generated on first run from the raw XML sources.
* ``mirume.db``  – the user's personal database: saved words, saved sentences,
  saved grammar patterns and word-encounter history.

Both database files live in the project-level ``data/`` directory. This module
exposes an engine, a session factory and a FastAPI-style session dependency for
each database, plus :func:`init_databases` which creates any missing tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from paths import DATA_DIR  # noqa: F401  (re-exported for callers that import it here)

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #

#: Absolute path to the ``data/`` directory holding both SQLite databases and
#: the downloaded dictionary sources. Resolved by :mod:`paths` — the repo's
#: own ``data/`` in development, a writable per-user directory in the packaged
#: app.

JMDICT_DB_PATH: Path = DATA_DIR / "jmdict.db"
MIRUME_DB_PATH: Path = DATA_DIR / "mirume.db"

JMDICT_DB_URL: str = f"sqlite:///{JMDICT_DB_PATH}"
MIRUME_DB_URL: str = f"sqlite:///{MIRUME_DB_PATH}"


# --------------------------------------------------------------------------- #
# Declarative bases
# --------------------------------------------------------------------------- #


class Base(DeclarativeBase):
    """Declarative base for the user's personal database (``mirume.db``).

    Every SQLAlchemy model that stores user data (saved words, sentences,
    grammar patterns, encounter history) inherits from this class.
    """


class JMdictBase(DeclarativeBase):
    """Declarative base for the dictionary database (``jmdict.db``).

    Dictionary models (JMdict entries, kanji data, JLPT tags) inherit from this
    class. Kept separate from :class:`Base` so the two databases never share a
    metadata namespace.
    """


# --------------------------------------------------------------------------- #
# Engines and session factories
# --------------------------------------------------------------------------- #


def _create_sqlite_engine(url: str) -> Engine:
    """Create a SQLAlchemy engine for a local SQLite database.

    ``check_same_thread`` is disabled because FastAPI serves requests from a
    thread pool and a single connection may legitimately be handed between
    threads. Access is still serialised per-session by SQLAlchemy.

    Args:
        url: A ``sqlite:///`` connection URL.

    Returns:
        A configured, not-yet-connected :class:`~sqlalchemy.Engine`.
    """
    return create_engine(
        url,
        connect_args={"check_same_thread": False},
        future=True,
    )


#: Engine bound to the user's personal database.
mirume_engine: Engine = _create_sqlite_engine(MIRUME_DB_URL)
#: Engine bound to the JMdict dictionary database.
jmdict_engine: Engine = _create_sqlite_engine(JMDICT_DB_URL)

#: Session factory for ``mirume.db``.
MirumeSession: sessionmaker[Session] = sessionmaker(
    bind=mirume_engine, autoflush=False, expire_on_commit=False, future=True
)
#: Session factory for ``jmdict.db``.
JMdictSession: sessionmaker[Session] = sessionmaker(
    bind=jmdict_engine, autoflush=False, expire_on_commit=False, future=True
)


# --------------------------------------------------------------------------- #
# Schema setup
# --------------------------------------------------------------------------- #


def init_databases() -> None:
    """Create all tables in both databases if they do not already exist.

    Importing :mod:`models` for its side effects registers the user models on
    :class:`Base`'s metadata before ``create_all`` runs. Dictionary tables are
    created too, so the schema is ready before the JMdict import populates it.

    Safe to call on every startup: ``create_all`` is a no-op for tables that
    already exist.
    """
    import jlpt  # noqa: F401  (registers Entry/Kanji on JMdictBase.metadata)
    import models  # noqa: F401  (registers user models on Base.metadata)

    Base.metadata.create_all(bind=mirume_engine)
    JMdictBase.metadata.create_all(bind=jmdict_engine)


# --------------------------------------------------------------------------- #
# Request-scoped session dependencies
# --------------------------------------------------------------------------- #


def get_mirume_session() -> Iterator[Session]:
    """Yield a session for ``mirume.db``, closing it when the caller is done.

    Intended for use as a FastAPI dependency::

        @app.post("/save/word")
        def save_word(db: Session = Depends(get_mirume_session)) -> ...:

    Yields:
        An open :class:`~sqlalchemy.orm.Session` bound to ``mirume.db``.
    """
    session = MirumeSession()
    try:
        yield session
    finally:
        session.close()


def get_jmdict_session() -> Iterator[Session]:
    """Yield a session for ``jmdict.db``, closing it when the caller is done.

    Intended for use as a FastAPI dependency for endpoints that need dictionary
    lookups.

    Yields:
        An open :class:`~sqlalchemy.orm.Session` bound to ``jmdict.db``.
    """
    session = JMdictSession()
    try:
        yield session
    finally:
        session.close()
