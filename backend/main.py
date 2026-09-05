"""FastAPI application entry point for the Mirume backend.

Run locally with::

    uvicorn main:app --reload --reload-exclude 'venv/*' --port 8123

This starts a server on ``http://127.0.0.1:8123`` (port 8123 to avoid a
conflict with another local project on 8000). Install ``watchfiles`` (in
``requirements.txt``) so ``--reload`` uses the event-based watcher;
``--reload-exclude 'venv/*'`` then keeps it off the 30k-file virtualenv.
Without both, uvicorn's fallback stat-poll reloader pins a CPU core walking
``venv/`` every 250 ms and bounces the worker on any stray mtime change. Note
that the reloader does **not** resurrect a worker that *crashes* (as opposed to
being restarted) — if ``/hover`` ever hangs for every caller, the worker has
died and the server must be stopped and restarted. Routes:

* ``GET  /health`` – liveness check; also reports the database paths and
  whether the JMdict dictionary has been built yet.
* ``POST /hover``   – accepts ``{x, y}`` screen coordinates. Reads the real
  on-screen text at that point via the macOS Accessibility API
  (:mod:`accessibility`), detects whether it's Japanese, English, or a mix
  (:mod:`language`), then routes it through the JLPT classifier
  (:mod:`jlpt`) and/or the DeepL translator (:mod:`translator`) as needed.
  Falls back to a fixed placeholder sentence when no text is found
  (``source`` is ``"accessibility"`` vs ``"placeholder"``).
* ``POST /save/word`` / ``/save/sentence`` / ``/save/grammar`` – save items
  the user wants to remember, to ``mirume.db`` (:mod:`models`).
* ``POST /encounter`` – log a passive hover exposure for progress stats.
* ``GET  /review`` / ``POST /review/grade`` – SM-2 spaced-repetition review
  queue (:mod:`spaced_rep`).
* ``GET  /review/words`` / ``GET /review/flashcard`` /
  ``DELETE /review/words/{id}`` – data for the separate review window (see
  ``frontend/src/ReviewApp.tsx``).
* ``GET  /word/{word}/sentences`` – up to 3 Tatoeba example sentences for a
  word (:mod:`sentences`).
* ``GET  /stats`` – aggregate progress statistics.

Before first use:

* Build the dictionary once::   python jlpt.py build
* Grant Accessibility permission to the terminal / Python (see
  :mod:`accessibility`); without it every hover falls back to the placeholder.
* Add a DeepL API key to ``backend/.env`` (``DEEPL_API_KEY=...``) to enable
  English→Japanese translation; without it, English hover text is detected
  but not translated.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from accessibility import (
    accessibility_permission_granted,
    get_focused_text,
    get_text_with_ocr_fallback,
)
from database import JMDICT_DB_PATH, MIRUME_DB_PATH, get_mirume_session, init_databases
from jlpt import (
    ClassifiedToken,
    KanjiInfo,
    classify_tokens,
    dictionary_ready,
    get_example_sentences,
    get_kanji_breakdown,
)
from language import detect_language
from sentences import example_sentences_ready
from models import (
    JLPT_LEVELS,
    ReviewLog,
    SavedGrammar,
    SavedSentence,
    SavedWord,
    WordEncounter,
)
from spaced_rep import sm2
from tokeniser import tokenise
from translator import TranslatorNotConfiguredError, english_to_japanese

API_VERSION = "0.4.0"

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
    if not example_sentences_ready():
        print(
            "\n[mirume] Tatoeba example sentences are not built yet — "
            "/word/{word}/sentences and the review window's flashcards will "
            "return no example sentences.\n"
            "[mirume] Run:  python sentences.py build\n"
        )
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
    if not os.environ.get("DEEPL_API_KEY", "").strip():
        print(
            "\n[mirume] DEEPL_API_KEY is not set — /hover will detect English text "
            "but skip translation.\n"
            "[mirume] Add it to backend/.env (get a free-tier key at "
            "https://www.deepl.com/pro-api).\n"
        )

    # Start the persistent OCR worker thread and warm up its model: loading
    # PaddleOCR takes several seconds (longer still on a cold weights-download
    # cache), so kick it off now rather than on the first Chrome hover. Runs
    # on a daemon thread so a slow/absent network never blocks startup; the
    # (0, 0) capture yields no Japanese and is discarded — we only want the
    # worker's model resident in memory. The warmup call gets a generous
    # timeout since, unlike a real hover, it's the one call that has to
    # actually wait out the model load rather than give up quickly.
    def _warm_ocr() -> None:
        try:
            from ocr import extract_text_at_position, start_ocr_worker

            start_ocr_worker()
            extract_text_at_position(0, 0, timeout=30)
        except Exception as exc:  # pragma: no cover - model/deps optional
            print(f"[mirume] OCR warmup failed ({exc}); Chrome hover will be "
                  "slow on first use or unavailable.")

    threading.Thread(target=_warm_ocr, name="ocr-warmup", daemon=True).start()

    yield


app = FastAPI(
    title="Mirume",
    description="Local backend for the Mirume Japanese-learning overlay.",
    version=API_VERSION,
    lifespan=lifespan,
)

# The overlay's webview runs on http://localhost:1420 (Tauri dev) and calls this
# API on 127.0.0.1:8123 — a cross-origin pair, so WKWebView requires CORS headers
# or it blocks the frontend from reading the response (the request still reaches
# the server and logs 200, but `fetch` rejects). Everything here is loopback-only,
# so a wildcard origin is fine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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


class TranslationOut(BaseModel):
    """An English→Japanese translation of one segment of the detected text."""

    source_text: str = Field(description="The original English segment that was translated.")
    translation: str = Field(description="Japanese translation.")
    reading: str | None = Field(description="Hiragana reading of the translation.")
    jlpt_level: str = Field(description="Hardest JLPT level in the translation, or 'unknown'.")
    jlpt_name: str = Field(description="Same as jlpt_level; kept for API symmetry.")
    grammar_patterns: list[dict[str, str]] = Field(
        description="Common JLPT grammar patterns detected in the translation."
    )
    alternatives: list[str] = Field(
        description="Alternative phrasings, roughly casual (N5) to formal (N2)."
    )

    @classmethod
    def from_result(cls, source_text: str, result: dict) -> "TranslationOut":
        """Build a :class:`TranslationOut` from :func:`translator.english_to_japanese`.

        Args:
            source_text: The English segment that was translated.
            result: The dict returned by :func:`translator.english_to_japanese`.

        Returns:
            The corresponding response model.
        """
        return cls(source_text=source_text, **result)


class HoverResponse(BaseModel):
    """Body returned by ``POST /hover``."""

    text: str = Field(description="Detected text under the cursor.")
    source: str = Field(
        description="How the text was obtained: 'accessibility' (AX tree text), "
        "'ocr' (screenshot + manga-ocr, used for sandboxed content like Chrome "
        "web pages), or 'placeholder' (nothing detected / no permission)."
    )
    language: str = Field(
        description="Detected language of the text: 'ja', 'en', or 'mixed'."
    )
    request_point: HoverRequest = Field(description="Echo of the requested coordinates.")
    tokens: list[TokenOut] = Field(
        description="Classified Japanese words, in reading order — populated for "
        "'ja' and 'mixed' text, empty for pure 'en' text."
    )
    kanji: list[KanjiOut] = Field(
        description="Per-kanji breakdown for every distinct kanji in the text."
    )
    translations: list[TranslationOut] = Field(
        description="English→Japanese translations — one entry for pure 'en' text, "
        "one per English segment for 'mixed' text, empty for pure 'ja' text."
    )


class SaveWordRequest(BaseModel):
    """Body accepted by ``POST /save/word``."""

    word: str = Field(description="Surface form of the word to save.")
    reading: str | None = Field(default=None, description="Kana reading for furigana.")
    meaning: str | None = Field(default=None, description="Short English gloss.")
    jlpt_level: str = Field(default="unknown", description="One of N5..N1 or 'unknown'.")
    context_sentence: str | None = Field(
        default=None, description="Sentence the word was saved from."
    )


class SaveWordResponse(BaseModel):
    """Body returned by ``POST /save/word``."""

    saved: bool
    word_id: int


class SaveSentenceRequest(BaseModel):
    """Body accepted by ``POST /save/sentence``."""

    sentence: str = Field(description="The sentence text.")
    reading: str | None = Field(default=None, description="Furigana reading.")
    translation: str | None = Field(default=None, description="English translation.")
    source: str | None = Field(default=None, description="App the sentence was seen in.")
    jlpt_level: str = Field(
        default="unknown", description="Dominant JLPT level appearing in the sentence."
    )


class SaveSentenceResponse(BaseModel):
    """Body returned by ``POST /save/sentence``."""

    saved: bool
    sentence_id: int


class SaveGrammarRequest(BaseModel):
    """Body accepted by ``POST /save/grammar``."""

    pattern: str = Field(description="Canonical form of the grammar pattern.")
    name: str | None = Field(default=None, description="Short human-readable name.")
    explanation: str | None = Field(default=None, description="What the pattern expresses.")
    example: str | None = Field(default=None, description="An example sentence.")
    jlpt_level: str = Field(default="unknown", description="One of N5..N1 or 'unknown'.")


class SaveGrammarResponse(BaseModel):
    """Body returned by ``POST /save/grammar``."""

    saved: bool
    grammar_id: int


class EncounterRequest(BaseModel):
    """Body accepted by ``POST /encounter``."""

    word: str = Field(description="Surface form of the word encountered.")
    jlpt_level: str = Field(default="unknown", description="One of N5..N1 or 'unknown'.")
    app_context: str | None = Field(
        default=None, description="App the word was hovered in."
    )


class EncounterResponse(BaseModel):
    """Body returned by ``POST /encounter``."""

    logged: bool


class SavedWordOut(BaseModel):
    """A :class:`models.SavedWord` row, serialised for API responses."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    surface: str
    lemma: str | None
    reading: str | None
    meaning: str | None
    jlpt_level: str
    context_sentence: str | None
    times_seen: int
    ease_factor: float
    interval_days: int
    repetitions: int
    due_date: datetime
    last_seen: datetime
    created_at: datetime


class ReviewResponse(BaseModel):
    """Body returned by ``GET /review``."""

    due_count: int
    words: list[SavedWordOut]


class StatsResponse(BaseModel):
    """Body returned by ``GET /stats``."""

    total_saved_words: int
    total_saved_sentences: int
    total_encounters: int
    words_by_level: dict[str, int]
    due_for_review: int
    most_seen_words: list[SavedWordOut]


class GradeRequest(BaseModel):
    """Body accepted by ``POST /review/grade``."""

    word_id: int = Field(description="Id of the SavedWord being reviewed.")
    grade: int = Field(ge=0, le=5, description="Self-graded recall quality, 0-5.")


class GradeResponse(BaseModel):
    """Body returned by ``POST /review/grade``."""

    next_review: date
    interval_days: int


class ExampleSentenceOut(BaseModel):
    """One Tatoeba example sentence for a word."""

    japanese: str = Field(description="The Japanese sentence text.")
    english: str = Field(description="The English translation.")


class WordSentencesResponse(BaseModel):
    """Body returned by ``GET /word/{word}/sentences``."""

    word: str
    sentences: list[ExampleSentenceOut]


class ReviewWordsResponse(BaseModel):
    """Body returned by ``GET /review/words``."""

    total: int = Field(description="Total number of saved words.")
    by_level: dict[str, list[SavedWordOut]] = Field(
        description="Saved words grouped by JLPT level (N1..N5; 'unknown' omitted)."
    )
    due_today: int = Field(description="Number of words currently due for review.")
    streak: int = Field(description="Consecutive days, ending today, with >=1 graded review.")


class FlashcardResponse(BaseModel):
    """Body returned by ``GET /review/flashcard``."""

    word: SavedWordOut | None = Field(
        description="The next word due for review, or null when none are due."
    )
    sentences: list[ExampleSentenceOut] = Field(
        description="Up to 3 Tatoeba example sentences for the word."
    )


class DeleteWordResponse(BaseModel):
    """Body returned by ``DELETE /review/words/{word_id}``."""

    deleted: bool


#: Hiragana, katakana and CJK ideograph ranges — used to split hover text into
#: same-script runs so mixed Japanese/English text can be routed per-segment.
_JAPANESE_SCRIPT_RE = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")

#: Narrower Japanese-character check used to drop OCR noise tokens (page
#: numbers, stray symbols) from /hover's token list — see _is_junk_token.
_JAPANESE_CHAR_RE = re.compile(r"[぀-鿿]")


def _is_junk_token(token: TokenOut) -> bool:
    """Whether a classified token is OCR noise rather than a real word.

    OCR occasionally reads page furniture (view counts, numbering) as text;
    the tokeniser happily turns ``"68"`` into a "word" that then gets a
    (bogus) JLPT classification. Drop anything that's purely numeric or has
    no Japanese characters in it at all.

    Args:
        token: A classified token from the /hover response.

    Returns:
        True if the token should be dropped.
    """
    if token.surface.isnumeric():
        return True
    return not _JAPANESE_CHAR_RE.search(token.surface)


def _segment_by_script(text: str) -> list[tuple[str, str]]:
    """Split ``text`` into contiguous runs of Japanese-script vs Latin-script text.

    Neutral characters (digits, punctuation, whitespace) are attached to
    whichever run they fall inside rather than starting a new one, so
    ``"食べた 2 apples"`` stays as two runs, not four.

    Args:
        text: Arbitrary hover text, possibly mixing Japanese and English.

    Returns:
        A list of ``(kind, segment)`` pairs, ``kind`` being ``"ja"`` or
        ``"en"``, in the order the segments appear in ``text``. Segments that
        are entirely whitespace are dropped.
    """
    segments: list[tuple[str, str]] = []
    buffer = ""
    kind: str | None = None
    for char in text:
        if _JAPANESE_SCRIPT_RE.match(char):
            char_kind = "ja"
        elif _LATIN_LETTER_RE.match(char):
            char_kind = "en"
        else:
            char_kind = None

        if char_kind is not None and kind is not None and char_kind != kind:
            segments.append((kind, buffer))
            buffer = ""
        if char_kind is not None:
            kind = char_kind
        buffer += char

    if buffer.strip():
        segments.append((kind or "en", buffer))
    return [(k, s) for k, s in segments if s.strip()]


def _compute_review_streak(db: Session) -> int:
    """Count consecutive days, ending today, with at least one graded review.

    Walks backwards from today (UTC) counting a day as "reviewed" if
    :class:`ReviewLog` has any row graded on it, stopping at the first gap.

    Args:
        db: ``mirume.db`` session (injected).

    Returns:
        The streak length in days (0 if no review was graded today).
    """
    reviewed_dates = {
        datetime.strptime(row, "%Y-%m-%d").date()
        for row in db.execute(
            select(func.date(ReviewLog.graded_at)).distinct()
        ).scalars()
    }
    streak = 0
    day = datetime.now(timezone.utc).date()
    while day in reviewed_dates:
        streak += 1
        day -= timedelta(days=1)
    return streak


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
async def hover(request: HoverRequest) -> HoverResponse:
    """Detect and explain the text under the cursor (async wrapper).

    The real work is :func:`_hover_sync`, which blocks: it may take a
    screenshot and run manga-ocr over it, and it makes synchronous SQLite
    dictionary lookups. Offloading it to the default executor keeps the event
    loop free and isolates it from FastAPI's shared sync-endpoint thread pool,
    so a burst of slow hovers can't starve the other routes.

    Args:
        request: The screen coordinates the user is hovering over.

    Returns:
        A :class:`HoverResponse`.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _hover_sync, request)


def _hover_sync(request: HoverRequest) -> HoverResponse:
    """Detect the text under the cursor and explain it, in whichever direction fits.

    Reads real on-screen text at ``(x, y)`` via the macOS Accessibility API,
    then the currently-focused element as a fallback. If neither yields text
    (nothing there, or permission not granted) a fixed placeholder sentence is
    used.

    The text is then routed by script composition (:func:`_segment_by_script`):

    * Pure Japanese → tokenised and classified against JMdict + the JLPT
      lists (:func:`jlpt.classify_tokens`), same as before.
    * Pure English → translated to Japanese and classified the same way
      (:func:`translator.english_to_japanese`), so the response always
      carries JLPT-graded content.
    * Mixed text → split into same-script runs, each handled by whichever of
      the above pipelines fits; Japanese runs contribute to ``tokens``,
      English runs each become one ``translations`` entry.

    Args:
        request: The screen coordinates the user is hovering over.

    Returns:
        A :class:`HoverResponse`. ``source`` describes how the text was
        obtained (``"accessibility"`` vs ``"placeholder"``); ``language``
        describes what was found there (``"ja"``, ``"en"`` or ``"mixed"``).
    """
    detected, detection_source = get_text_with_ocr_fallback(request.x, request.y)
    if not detected:
        detected = get_focused_text()
        detection_source = "accessibility" if detected else "none"

    if not detected:
        # Nothing under the cursor from AX, OCR, or the focused element (or no
        # permission). Return an empty response rather than a placeholder
        # sentence, so the frontend keeps the hover card hidden.
        return HoverResponse(
            text="",
            source="placeholder",
            language="unknown",
            request_point=request,
            tokens=[],
            kanji=[],
            translations=[],
        )
    text, source = detected, detection_source

    segments = _segment_by_script(text)
    scripts_present = {kind for kind, segment in segments}

    if scripts_present == {"ja"}:
        language = "ja"
    elif scripts_present == {"en"}:
        language = "en"
    elif scripts_present == {"ja", "en"}:
        language = "mixed"
    else:
        # No segment survived (empty/whitespace-only text) — fall back to the
        # general-purpose detector.
        language = detect_language(text)

    tokens: list[TokenOut] = []
    translations: list[TranslationOut] = []
    for kind, segment in segments:
        if kind == "ja":
            tokens.extend(
                t
                for t in (
                    TokenOut.from_classified(c) for c in classify_tokens(tokenise(segment))
                )
                if not _is_junk_token(t)
            )
        else:
            try:
                result = english_to_japanese(segment)
            except TranslatorNotConfiguredError:
                continue
            translations.append(TranslationOut.from_result(segment, result))

    # get_kanji_breakdown() already dedupes by literal internally; this second
    # pass guarantees the /hover response itself never carries a duplicate
    # literal even if that internal guarantee ever regresses.
    kanji = list(
        {k.literal: k for k in (KanjiOut.from_kanji_info(k) for k in get_kanji_breakdown(text))}.values()
    )
    return HoverResponse(
        text=text,
        source=source,
        language=language,
        request_point=request,
        tokens=tokens,
        kanji=kanji,
        translations=translations,
    )


@app.get("/word/{word}/sentences", response_model=WordSentencesResponse)
def word_sentences(word: str) -> WordSentencesResponse:
    """Return up to 3 Tatoeba example sentences containing ``word``.

    Args:
        word: A surface or dictionary form to look up.

    Returns:
        A :class:`WordSentencesResponse`; ``sentences`` is empty if the
        example-sentence tables haven't been built yet (see
        ``python sentences.py build``) or no sentence contains the word.
    """
    return WordSentencesResponse(
        word=word,
        sentences=[ExampleSentenceOut(**s) for s in get_example_sentences(word, limit=3)],
    )


@app.post("/save/word", response_model=SaveWordResponse)
def save_word(
    request: SaveWordRequest, db: Session = Depends(get_mirume_session)
) -> SaveWordResponse:
    """Save a vocabulary word, or bump its exposure count if already saved.

    Matches existing saves by exact surface form. A repeat save increments
    ``times_seen`` and refreshes ``last_seen`` / ``context_sentence`` rather
    than creating a duplicate row.

    Args:
        request: The word, reading, meaning, JLPT level and optional context.
        db: ``mirume.db`` session (injected).

    Returns:
        A :class:`SaveWordResponse` with the row's id.
    """
    now = datetime.now(timezone.utc)
    existing = db.execute(
        select(SavedWord).where(SavedWord.surface == request.word)
    ).scalar_one_or_none()

    if existing is not None:
        existing.times_seen += 1
        existing.last_seen = now
        if request.context_sentence:
            existing.context_sentence = request.context_sentence
        db.commit()
        return SaveWordResponse(saved=True, word_id=existing.id)

    word = SavedWord(
        surface=request.word,
        reading=request.reading,
        meaning=request.meaning,
        jlpt_level=request.jlpt_level,
        context_sentence=request.context_sentence,
        times_seen=1,
        last_seen=now,
    )
    db.add(word)
    db.commit()
    db.refresh(word)
    return SaveWordResponse(saved=True, word_id=word.id)


@app.post("/save/sentence", response_model=SaveSentenceResponse)
def save_sentence(
    request: SaveSentenceRequest, db: Session = Depends(get_mirume_session)
) -> SaveSentenceResponse:
    """Save a sentence, usually as context for a saved word.

    Args:
        request: The sentence text and optional reading/translation/source.
        db: ``mirume.db`` session (injected).

    Returns:
        A :class:`SaveSentenceResponse` with the new row's id.
    """
    sentence = SavedSentence(
        text=request.sentence,
        reading=request.reading,
        translation=request.translation,
        dominant_jlpt_level=request.jlpt_level,
        source_app=request.source,
    )
    db.add(sentence)
    db.commit()
    db.refresh(sentence)
    return SaveSentenceResponse(saved=True, sentence_id=sentence.id)


@app.post("/save/grammar", response_model=SaveGrammarResponse)
def save_grammar(
    request: SaveGrammarRequest, db: Session = Depends(get_mirume_session)
) -> SaveGrammarResponse:
    """Save a JLPT grammar pattern.

    Args:
        request: The pattern, its name, explanation, example and JLPT level.
        db: ``mirume.db`` session (injected).

    Returns:
        A :class:`SaveGrammarResponse` with the new row's id.
    """
    grammar = SavedGrammar(
        pattern=request.pattern,
        name=request.name,
        meaning=request.explanation,
        example_sentence=request.example,
        jlpt_level=request.jlpt_level,
    )
    db.add(grammar)
    db.commit()
    db.refresh(grammar)
    return SaveGrammarResponse(saved=True, grammar_id=grammar.id)


@app.post("/encounter", response_model=EncounterResponse)
def log_encounter(
    request: EncounterRequest, db: Session = Depends(get_mirume_session)
) -> EncounterResponse:
    """Log one passive exposure to a word (called on every hover).

    Independent of whether the word has been saved; if it has, the encounter
    is linked to the :class:`SavedWord` row via ``saved_word_id``.

    Args:
        request: The word, its JLPT level and the app it appeared in.
        db: ``mirume.db`` session (injected).

    Returns:
        An :class:`EncounterResponse` acknowledging the log.
    """
    saved_word = db.execute(
        select(SavedWord).where(SavedWord.surface == request.word)
    ).scalar_one_or_none()
    encounter = WordEncounter(
        surface=request.word,
        jlpt_level=request.jlpt_level,
        source_app=request.app_context,
        saved_word_id=saved_word.id if saved_word else None,
    )
    db.add(encounter)
    db.commit()
    return EncounterResponse(logged=True)


@app.get("/review", response_model=ReviewResponse)
def get_review(db: Session = Depends(get_mirume_session)) -> ReviewResponse:
    """Return saved words due for review today, per SM-2 scheduling.

    Args:
        db: ``mirume.db`` session (injected).

    Returns:
        A :class:`ReviewResponse` with the due words ordered soonest-due first.
    """
    now = datetime.now(timezone.utc)
    due = (
        db.execute(
            select(SavedWord).where(SavedWord.due_date <= now).order_by(SavedWord.due_date)
        )
        .scalars()
        .all()
    )
    words = [SavedWordOut.model_validate(w) for w in due]
    return ReviewResponse(due_count=len(words), words=words)


@app.get("/review/words", response_model=ReviewWordsResponse)
def get_review_words(db: Session = Depends(get_mirume_session)) -> ReviewWordsResponse:
    """Return every saved word, grouped by JLPT level, for the review window.

    Args:
        db: ``mirume.db`` session (injected).

    Returns:
        A :class:`ReviewWordsResponse`. Words with ``jlpt_level="unknown"``
        are counted in ``total`` but omitted from ``by_level`` (which only
        has N1..N5 buckets).
    """
    all_words = db.execute(select(SavedWord)).scalars().all()

    by_level: dict[str, list[SavedWordOut]] = {level: [] for level in JLPT_LEVELS if level != "unknown"}
    for word in all_words:
        out = SavedWordOut.model_validate(word)
        if word.jlpt_level in by_level:
            by_level[word.jlpt_level].append(out)

    # Compared in SQL (SQLite's own string comparison), not against a Python
    # datetime: SQLite hands back *naive* datetimes for a `DateTime(timezone=
    # True)` column, which can't be compared to an aware `datetime.now(...)`
    # in Python without an explicit strip/attach step.
    now = datetime.now(timezone.utc)
    due_today = db.execute(
        select(func.count()).select_from(SavedWord).where(SavedWord.due_date <= now)
    ).scalar_one()

    return ReviewWordsResponse(
        total=len(all_words),
        by_level=by_level,
        due_today=due_today,
        streak=_compute_review_streak(db),
    )


@app.get("/review/flashcard", response_model=FlashcardResponse)
def get_flashcard(db: Session = Depends(get_mirume_session)) -> FlashcardResponse:
    """Return the next word due for review, per SM-2, with example sentences.

    Args:
        db: ``mirume.db`` session (injected).

    Returns:
        A :class:`FlashcardResponse` with ``word=None`` when nothing is due
        (the review window shows an "All done!" screen in that case).
    """
    now = datetime.now(timezone.utc)
    word = db.execute(
        select(SavedWord).where(SavedWord.due_date <= now).order_by(SavedWord.due_date).limit(1)
    ).scalar_one_or_none()
    if word is None:
        return FlashcardResponse(word=None, sentences=[])

    sentences = get_example_sentences(word.surface, limit=3)
    return FlashcardResponse(
        word=SavedWordOut.model_validate(word),
        sentences=[ExampleSentenceOut(**s) for s in sentences],
    )


@app.delete("/review/words/{word_id}", response_model=DeleteWordResponse)
def delete_review_word(word_id: int, db: Session = Depends(get_mirume_session)) -> DeleteWordResponse:
    """Delete a saved word from the review window's word list.

    Args:
        word_id: Id of the :class:`SavedWord` to delete.
        db: ``mirume.db`` session (injected).

    Returns:
        A :class:`DeleteWordResponse`; ``deleted`` is ``False`` if no word
        with that id existed.

    Raises:
        HTTPException: 404 if no :class:`SavedWord` matches ``word_id``.
    """
    word = db.get(SavedWord, word_id)
    if word is None:
        raise HTTPException(status_code=404, detail="word not found")
    db.delete(word)
    db.commit()
    return DeleteWordResponse(deleted=True)


@app.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_mirume_session)) -> StatsResponse:
    """Return aggregate progress statistics across saved words and encounters.

    Args:
        db: ``mirume.db`` session (injected).

    Returns:
        A :class:`StatsResponse` with totals, a per-level breakdown, the
        current review queue size and the ten most-encountered saved words.
    """
    total_words = db.execute(select(func.count()).select_from(SavedWord)).scalar_one()
    total_sentences = db.execute(
        select(func.count()).select_from(SavedSentence)
    ).scalar_one()
    total_encounters = db.execute(
        select(func.count()).select_from(WordEncounter)
    ).scalar_one()

    level_rows = db.execute(
        select(SavedWord.jlpt_level, func.count()).group_by(SavedWord.jlpt_level)
    ).all()
    level_counts = dict(level_rows)
    words_by_level = {
        level: level_counts.get(level, 0) for level in JLPT_LEVELS if level != "unknown"
    }

    now = datetime.now(timezone.utc)
    due_for_review = db.execute(
        select(func.count()).select_from(SavedWord).where(SavedWord.due_date <= now)
    ).scalar_one()

    most_seen = (
        db.execute(select(SavedWord).order_by(SavedWord.times_seen.desc()).limit(10))
        .scalars()
        .all()
    )

    return StatsResponse(
        total_saved_words=total_words,
        total_saved_sentences=total_sentences,
        total_encounters=total_encounters,
        words_by_level=words_by_level,
        due_for_review=due_for_review,
        most_seen_words=[SavedWordOut.model_validate(w) for w in most_seen],
    )


@app.post("/review/grade", response_model=GradeResponse)
def grade_review(
    request: GradeRequest, db: Session = Depends(get_mirume_session)
) -> GradeResponse:
    """Grade a review and reschedule the word per SM-2.

    Args:
        request: The word id and a 0-5 self-graded recall quality.
        db: ``mirume.db`` session (injected).

    Returns:
        A :class:`GradeResponse` with the new due date and interval.

    Raises:
        HTTPException: 404 if no :class:`SavedWord` matches ``word_id``.
    """
    word = db.get(SavedWord, request.word_id)
    if word is None:
        raise HTTPException(status_code=404, detail="word not found")

    result = sm2(word.ease_factor, word.interval_days, word.repetitions, request.grade)
    word.ease_factor = result.ease_factor
    word.interval_days = result.interval_days
    word.repetitions = result.repetitions
    word.due_date = result.next_review_date
    # Logged purely so _compute_review_streak has history to count — SavedWord
    # itself only ever tracks *current* SM-2 state, not past reviews.
    db.add(ReviewLog(word_id=word.id, grade=request.grade))
    db.commit()

    return GradeResponse(
        next_review=result.next_review_date.date(), interval_days=result.interval_days
    )
