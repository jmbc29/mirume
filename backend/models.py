"""SQLAlchemy models for Mirume's personal database (``mirume.db``).

Four tables back the app's "save and review" workflow:

* :class:`SavedWord`     – individual vocabulary items the user saved.
* :class:`SavedSentence` – full sentences the user saved for context.
* :class:`SavedGrammar`  – JLPT grammar patterns the user saved.
* :class:`WordEncounter` – every time the user hovered a word, for the
  personalisation model and progress statistics.

:class:`SavedWord` and :class:`SavedGrammar` carry SM-2 spaced-repetition
fields (``ease_factor``, ``interval_days``, ``repetitions``, ``due_date``) so
the ``/review`` endpoint can schedule them. Those fields are populated by
:mod:`spaced_rep` and are not touched here beyond their defaults.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base

#: The five JLPT buckets Mirume classifies text into, plus ``"unknown"`` for
#: tokens with no dictionary/JLPT match. Stored as a short string on each row.
JLPT_LEVELS: tuple[str, ...] = ("N5", "N4", "N3", "N2", "N1", "unknown")


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Used as the default factory for ``created_at`` / ``due_date`` columns so new
    rows are stamped consistently regardless of server locale.
    """
    return datetime.now(timezone.utc)


class SavedWord(Base):
    """A single vocabulary item the user saved from a hover card.

    Attributes:
        id: Surrogate primary key.
        surface: The word exactly as it appeared on screen (may be inflected).
        lemma: Dictionary (base) form of the word, when known.
        reading: Kana reading used for furigana display.
        meaning: Short English gloss (typically the first JMdict sense).
        part_of_speech: Coarse part-of-speech tag from the tokeniser.
        jlpt_level: One of :data:`JLPT_LEVELS`.
        jmdict_id: JMdict entry id this word was matched to, if any.
        source_app: Name of the app the word was hovered in (e.g. ``"Safari"``).
        context_sentence: The sentence the word was saved from, if given.
        times_seen: Number of times the word has been saved/re-saved —
            incremented by ``POST /save/word`` on a repeat save.
        last_seen: When the word was last saved or re-encountered via save.
        ease_factor: SM-2 ease factor; starts at 2.5.
        interval_days: SM-2 current inter-review interval in days.
        repetitions: SM-2 count of consecutive successful reviews.
        due_date: When this word is next due for review.
        created_at: When the user saved the word.
    """

    __tablename__ = "saved_words"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    surface: Mapped[str] = mapped_column(String(128), index=True)
    lemma: Mapped[str | None] = mapped_column(String(128), default=None)
    reading: Mapped[str | None] = mapped_column(String(128), default=None)
    meaning: Mapped[str | None] = mapped_column(Text, default=None)
    part_of_speech: Mapped[str | None] = mapped_column(String(64), default=None)
    jlpt_level: Mapped[str] = mapped_column(String(8), default="unknown", index=True)
    jmdict_id: Mapped[int | None] = mapped_column(Integer, default=None, index=True)
    source_app: Mapped[str | None] = mapped_column(String(128), default=None)
    context_sentence: Mapped[str | None] = mapped_column(Text, default=None)
    times_seen: Mapped[int] = mapped_column(Integer, default=1)
    last_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    # --- SM-2 spaced-repetition state -------------------------------------- #
    ease_factor: Mapped[float] = mapped_column(Float, default=2.5)
    interval_days: Mapped[int] = mapped_column(Integer, default=0)
    repetitions: Mapped[int] = mapped_column(Integer, default=0)
    due_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    def __repr__(self) -> str:
        """Return a concise developer-facing representation."""
        return (
            f"SavedWord(id={self.id!r}, surface={self.surface!r}, "
            f"jlpt_level={self.jlpt_level!r})"
        )


class SavedSentence(Base):
    """A full sentence the user saved, usually as context for a saved word.

    Attributes:
        id: Surrogate primary key.
        text: The sentence exactly as detected on screen.
        reading: Optional fully-furigana'd reading of the sentence.
        translation: Optional English translation.
        difficulty: Continuous difficulty score from :mod:`difficulty`
            (roughly 0 = trivial, 1 = very hard), if scored.
        dominant_jlpt_level: The hardest JLPT level appearing in the sentence.
        source_app: Name of the app the sentence was detected in.
        created_at: When the user saved the sentence.
    """

    __tablename__ = "saved_sentences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    text: Mapped[str] = mapped_column(Text)
    reading: Mapped[str | None] = mapped_column(Text, default=None)
    translation: Mapped[str | None] = mapped_column(Text, default=None)
    difficulty: Mapped[float | None] = mapped_column(Float, default=None)
    dominant_jlpt_level: Mapped[str] = mapped_column(String(8), default="unknown")
    source_app: Mapped[str | None] = mapped_column(String(128), default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    def __repr__(self) -> str:
        """Return a concise developer-facing representation."""
        preview = (self.text[:20] + "…") if len(self.text) > 20 else self.text
        return f"SavedSentence(id={self.id!r}, text={preview!r})"


class SavedGrammar(Base):
    """A JLPT grammar pattern the user saved from a hover card.

    Attributes:
        id: Surrogate primary key.
        pattern: Canonical form of the pattern (e.g. ``"~なければならない"``).
        name: Short human-readable name for the pattern.
        meaning: Explanation of what the pattern expresses.
        jlpt_level: One of :data:`JLPT_LEVELS`.
        example_sentence: The sentence the pattern was detected in.
        notes: Free-form user notes.
        ease_factor: SM-2 ease factor; starts at 2.5.
        interval_days: SM-2 current inter-review interval in days.
        repetitions: SM-2 count of consecutive successful reviews.
        due_date: When this pattern is next due for review.
        created_at: When the user saved the pattern.
    """

    __tablename__ = "saved_grammar"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    pattern: Mapped[str] = mapped_column(String(128), index=True)
    name: Mapped[str | None] = mapped_column(String(128), default=None)
    meaning: Mapped[str | None] = mapped_column(Text, default=None)
    jlpt_level: Mapped[str] = mapped_column(String(8), default="unknown", index=True)
    example_sentence: Mapped[str | None] = mapped_column(Text, default=None)
    notes: Mapped[str | None] = mapped_column(Text, default=None)

    # --- SM-2 spaced-repetition state -------------------------------------- #
    ease_factor: Mapped[float] = mapped_column(Float, default=2.5)
    interval_days: Mapped[int] = mapped_column(Integer, default=0)
    repetitions: Mapped[int] = mapped_column(Integer, default=0)
    due_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    def __repr__(self) -> str:
        """Return a concise developer-facing representation."""
        return (
            f"SavedGrammar(id={self.id!r}, pattern={self.pattern!r}, "
            f"jlpt_level={self.jlpt_level!r})"
        )


class WordEncounter(Base):
    """One logged instance of the user hovering over a word.

    Every hover is recorded here (independent of whether the user saved the
    word) so the personalisation model can estimate what the user already
    knows, and so ``/stats`` can chart exposure over time.

    Attributes:
        id: Surrogate primary key.
        surface: The word as it appeared on screen.
        lemma: Dictionary (base) form, when known.
        jlpt_level: One of :data:`JLPT_LEVELS`.
        jmdict_id: JMdict entry id, if matched.
        source_app: Name of the app the word was hovered in.
        saved_word_id: Link to :class:`SavedWord` if the user later saved it.
        encountered_at: When the hover happened.
    """

    __tablename__ = "word_encounters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    surface: Mapped[str] = mapped_column(String(128), index=True)
    lemma: Mapped[str | None] = mapped_column(String(128), default=None, index=True)
    jlpt_level: Mapped[str] = mapped_column(String(8), default="unknown", index=True)
    jmdict_id: Mapped[int | None] = mapped_column(Integer, default=None, index=True)
    source_app: Mapped[str | None] = mapped_column(String(128), default=None)

    saved_word_id: Mapped[int | None] = mapped_column(
        ForeignKey("saved_words.id", ondelete="SET NULL"), default=None
    )
    saved_word: Mapped[SavedWord | None] = relationship("SavedWord")

    encountered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), index=True
    )

    def __repr__(self) -> str:
        """Return a concise developer-facing representation."""
        return (
            f"WordEncounter(id={self.id!r}, surface={self.surface!r}, "
            f"encountered_at={self.encountered_at!r})"
        )
