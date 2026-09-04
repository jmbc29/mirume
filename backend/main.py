"""FastAPI application entry point for the Mirume backend.

Run locally with::

    uvicorn main:app --reload --port 8123

This starts a server on ``http://127.0.0.1:8123`` (port 8123 to avoid a
conflict with another local project on 8000). Routes:

* ``GET  /health`` – liveness check; also reports the database paths and
  whether the JMdict dictionary has been built yet.
* ``POST /hover``   – accepts ``{x, y}`` screen coordinates. Reads the real
  on-screen text at that point via the macOS Accessibility API
  (:mod:`accessibility`), then runs it through the tokeniser and JLPT
  classifier. Falls back to a fixed placeholder sentence when no text is
  found (``source`` is ``"accessibility"`` vs ``"placeholder"``).

Before first use:

* Build the dictionary once::   python jlpt.py build
* Grant Accessibility permission to the terminal / Python (see
  :mod:`accessibility`); without it every hover falls back to the placeholder.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel, Field

from accessibility import (
    accessibility_permission_granted,
    get_focused_text,
    get_text_at_position,
)
from database import JMDICT_DB_PATH, MIRUME_DB_PATH, init_databases
from jlpt import (
    ClassifiedToken,
    KanjiInfo,
    classify_tokens,
    dictionary_ready,
    get_kanji_breakdown,
)
from tokeniser import tokenise

API_VERSION = "0.3.0"

#: Placeholder Japanese sentence returned by the /hover stub — "Studying
#: Japanese is hard, but it's fun." Chosen to span several JLPT levels so the
#: classifier output is visible. Replaced by real screen text detection later.
_PLACEHOLDER_TEXT = "日本語の勉強は難しいですが、面白いです"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Create database tables on startup and warn if the dictionary is missing.

    Runs once per process. Table creation is idempotent. The dictionary build
    is **not** triggered here (it downloads ~12 MB and takes ~1 min); instead a
    clear warning is printed telling the operator to run ``python jlpt.py
    build``.

    Args:
        _app: The FastAPI instance (unused).

    Yields:
        Control back to FastAPI for the lifetime of the application.
    """
    init_databases()
    if not dictionary_ready():
        print(
            "\n[mirume] JMdict dictionary is not built yet — /hover will return "
            "'unknown' for every word.\n"
            "[mirume] Run:  python jlpt.py build\n"
        )
    if not accessibility_permission_granted():
        print(
            "\n[mirume] macOS Accessibility permission not granted — /hover will "
            "always fall back to the placeholder sentence.\n"
            "[mirume] Grant it in System Settings > Privacy & Security > "
            "Accessibility for your terminal (or Python), then restart.\n"
        )
    yield


app = FastAPI(
    title="Mirume",
    description="Local backend for the Mirume Japanese-learning overlay.",
    version=API_VERSION,
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    """Body returned by ``GET /health``."""

    status: str = Field(description="``\"ok\"`` when the service is healthy.")
    version: str = Field(description="Backend API version.")
    mirume_db: str = Field(description="Filesystem path to the personal database.")
    jmdict_db: str = Field(description="Filesystem path to the dictionary database.")
    dictionary_ready: bool = Field(
        description="True once `python jlpt.py build` has populated jmdict.db."
    )
    accessibility_ready: bool = Field(
        description="True when macOS Accessibility permission is granted; when "
        "false, /hover always returns the placeholder sentence."
    )


class HoverRequest(BaseModel):
    """Body accepted by ``POST /hover``: a screen-space cursor position."""

    x: float = Field(description="Cursor X coordinate in screen points.")
    y: float = Field(description="Cursor Y coordinate in screen points.")


class TokenOut(BaseModel):
    """One classified word in a ``/hover`` response."""

    surface: str = Field(description="Word as it appears on screen.")
    reading: str | None = Field(description="Hiragana reading for furigana.")
    lemma: str = Field(description="Dictionary (base) form of the word.")
    part_of_speech: str | None = Field(description="Coarse Japanese POS tag.")
    is_content_word: bool = Field(
        description="True for nouns/verbs/adjectives/adverbs worth classifying."
    )
    jlpt_level: int | None = Field(
        description="JLPT level as an integer 5 (N5) … 1 (N1), or null if unknown."
    )
    jlpt_name: str = Field(description="JLPT level as 'N5'–'N1', or 'unknown'.")
    jlpt_estimated: bool = Field(
        description="True when the level was estimated from the word's kanji "
        "rather than taken from a JLPT vocab list."
    )
    meaning: str | None = Field(description="English glosses, or null if not found.")
    dictionary_reading: str | None = Field(
        description="The matched dictionary entry's primary reading, or null."
    )
    jmdict_id: int | None = Field(description="JMdict entry id, or null if not found.")

    @classmethod
    def from_classified(cls, token: ClassifiedToken) -> "TokenOut":
        """Build a :class:`TokenOut` from a :class:`jlpt.ClassifiedToken`.

        Args:
            token: A token produced by :func:`jlpt.classify_tokens`.

        Returns:
            The corresponding response model.
        """
        return cls(
            surface=token.surface,
            reading=token.reading,
            lemma=token.lemma,
            part_of_speech=token.part_of_speech,
            is_content_word=token.is_content_word,
            jlpt_level=token.jlpt_level,
            jlpt_name=token.jlpt_name,
            jlpt_estimated=token.jlpt_estimated,
            meaning=token.meaning,
            dictionary_reading=token.dictionary_reading,
            jmdict_id=token.jmdict_id,
        )


class KanjiOut(BaseModel):
    """One kanji in the ``/hover`` per-character breakdown."""

    literal: str = Field(description="The kanji character.")
    meaning: str = Field(description="English meanings joined with '; '.")
    reading_on: str = Field(description="On'yomi readings joined with '、'.")
    reading_kun: str = Field(description="Kun'yomi readings joined with '、'.")
    jlpt_level: int | None = Field(description="JLPT level 5 … 1, or null.")
    jlpt_name: str = Field(description="JLPT level as 'N5'–'N1', or 'unknown'.")

    @classmethod
    def from_kanji_info(cls, info: KanjiInfo) -> "KanjiOut":
        """Build a :class:`KanjiOut` from a :class:`jlpt.KanjiInfo`.

        Args:
            info: An item from :func:`jlpt.get_kanji_breakdown`.

        Returns:
            The corresponding response model.
        """
        return cls(
            literal=info.literal,
            meaning=info.meaning,
            reading_on=info.reading_on,
            reading_kun=info.reading_kun,
            jlpt_level=info.jlpt_level,
            jlpt_name=info.jlpt_name,
        )


class HoverResponse(BaseModel):
    """Body returned by ``POST /hover``."""

    text: str = Field(description="Detected Japanese text under the cursor.")
    source: str = Field(
        description="How the text was obtained: 'accessibility' (real screen "
        "text), 'placeholder' (fallback — nothing detected or no permission), "
        "or 'ocr' (not implemented yet)."
    )
    request_point: HoverRequest = Field(description="Echo of the requested coordinates.")
    tokens: list[TokenOut] = Field(description="Classified words, in reading order.")
    kanji: list[KanjiOut] = Field(
        description="Per-kanji breakdown for every distinct kanji in the text."
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the backend is up and where its databases live.

    Returns:
        A :class:`HealthResponse` with ``status="ok"``.
    """
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        mirume_db=str(MIRUME_DB_PATH),
        jmdict_db=str(JMDICT_DB_PATH),
        dictionary_ready=dictionary_ready(),
        accessibility_ready=accessibility_permission_granted(),
    )


@app.post("/hover", response_model=HoverResponse)
def hover(request: HoverRequest) -> HoverResponse:
    """Detect the text under the cursor and classify it by JLPT level.

    Reads real on-screen text at ``(x, y)`` via the macOS Accessibility API,
    then the currently-focused element as a fallback. If neither yields text
    (nothing there, or permission not granted) a fixed placeholder sentence is
    used. The resulting text is tokenised (:func:`tokeniser.tokenise`),
    classified against JMdict + the JLPT lists (:func:`jlpt.classify_tokens`),
    and given a per-kanji breakdown (:func:`jlpt.get_kanji_breakdown`).

    Args:
        request: The screen coordinates the user is hovering over.

    Returns:
        A :class:`HoverResponse`. ``source`` is ``"accessibility"`` when real
        text was read, otherwise ``"placeholder"``.
    """
    detected = get_text_at_position(request.x, request.y) or get_focused_text()
    if detected:
        text, source = detected, "accessibility"
    else:
        text, source = _PLACEHOLDER_TEXT, "placeholder"

    tokens = [TokenOut.from_classified(t) for t in classify_tokens(tokenise(text))]
    kanji = [KanjiOut.from_kanji_info(k) for k in get_kanji_breakdown(text)]
    return HoverResponse(
        text=text,
        source=source,
        request_point=request,
        tokens=tokens,
        kanji=kanji,
    )
