"""Japanese tokenisation for Mirume, backed by fugashi + MeCab (unidic-lite).

Japanese is written without spaces, so before any dictionary lookup or JLPT
classification the raw text has to be split into words. :func:`tokenise` does
that and returns, for each token, its surface form, kana reading, dictionary
(lemma) form and a coarse part-of-speech tag.

The MeCab ``Tagger`` is expensive to construct (it memory-maps the dictionary),
so a single module-level instance is created lazily on first use and reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import fugashi

# Offset between a katakana code point and its hiragana counterpart in Unicode.
_KATAKANA_TO_HIRAGANA_OFFSET = ord("あ") - ord("ア")
# Katakana block that has a matching hiragana character (ァ..ヶ).
_KATAKANA_START = ord("ァ")
_KATAKANA_END = ord("ヶ")


@dataclass(frozen=True, slots=True)
class Token:
    """A single tokenised unit of Japanese text.

    Attributes:
        surface: The token exactly as it appeared in the input (possibly
            inflected, e.g. ``"食べた"`` → surface ``"食べ"`` for the stem).
        reading: Hiragana reading of the surface form, for furigana. ``None``
            when MeCab has no reading (e.g. Latin text, punctuation).
        lemma: Dictionary/base form used for lookups (e.g. ``"食べる"``).
            Falls back to the surface form when unavailable.
        part_of_speech: Coarse Japanese POS tag from unidic (``pos1``), e.g.
            ``"名詞"`` (noun), ``"動詞"`` (verb), ``"助詞"`` (particle).
        is_content_word: ``True`` for nouns, verbs, adjectives, adverbs — the
            tokens worth classifying and offering to save. ``False`` for
            particles, symbols, whitespace, auxiliary verbs.
    """

    surface: str
    reading: str | None
    lemma: str
    part_of_speech: str | None
    is_content_word: bool


#: unidic ``pos1`` values considered "content" words worth classifying/saving.
_CONTENT_POS: frozenset[str] = frozenset(
    {"名詞", "動詞", "形容詞", "形状詞", "副詞", "連体詞", "感動詞"}
)


def katakana_to_hiragana(text: str) -> str:
    """Convert every katakana character in ``text`` to hiragana.

    MeCab returns readings in katakana, but furigana is conventionally rendered
    in hiragana. Characters outside the katakana block (kanji already stripped
    by MeCab, the長音 mark ``ー``, punctuation) are passed through unchanged.

    Args:
        text: A string that may contain katakana.

    Returns:
        The same string with katakana mapped to hiragana.
    """
    return "".join(
        chr(ord(ch) + _KATAKANA_TO_HIRAGANA_OFFSET)
        if _KATAKANA_START <= ord(ch) <= _KATAKANA_END
        else ch
        for ch in text
    )


@lru_cache(maxsize=1)
def _get_tagger() -> fugashi.Tagger:
    """Build (once) and return the shared fugashi ``Tagger``.

    fugashi auto-discovers the ``unidic-lite`` dictionary that ships as a Python
    package, so no dictionary path needs to be configured. Cached so the
    dictionary is memory-mapped only once per process.

    Returns:
        A ready-to-use :class:`fugashi.Tagger`.
    """
    return fugashi.Tagger()


def _feature_value(word: fugashi.UnidicNode, *names: str) -> str | None:
    """Return the first present, non-empty feature value from ``word``.

    unidic-lite and full unidic expose slightly different feature attribute
    names; MeCab also uses ``"*"`` to mean "no value". This helper tries each
    candidate attribute in order and normalises ``"*"`` / empty to ``None``.

    Args:
        word: A node yielded by iterating the tagger.
        *names: Candidate feature attribute names, tried in order.

    Returns:
        The first usable string value, or ``None`` if none apply.
    """
    for name in names:
        value = getattr(word.feature, name, None)
        if value and value != "*":
            return value
    return None


def _clean_lemma(lemma: str) -> str:
    """Strip unidic's ``カタカナ-english`` loanword annotation from a lemma.

    unidic records loanword lemmas as e.g. ``"ホルダー-holder"``; only the part
    before the hyphen is the Japanese dictionary form. Native lemmas rarely
    contain an ASCII segment, so this only trims when the tail is ASCII.

    Args:
        lemma: A raw lemma string from a MeCab feature.

    Returns:
        The lemma with any trailing ``-<ascii>`` annotation removed.
    """
    head, sep, tail = lemma.partition("-")
    if sep and head and tail.isascii():
        return head
    return lemma


def tokenise(text: str) -> list[Token]:
    """Split Japanese ``text`` into tokens with reading, lemma and POS.

    Whitespace-only tokens are dropped. Everything else MeCab emits is kept,
    including particles and punctuation, with ``is_content_word`` marking which
    tokens are meaningful vocabulary.

    Args:
        text: Raw text, typically a sentence or fragment detected on screen.
            May contain non-Japanese characters; those come back as tokens with
            ``reading=None``.

    Returns:
        A list of :class:`Token` in document order. Empty if ``text`` is empty
        or whitespace only.
    """
    stripped = text.strip()
    if not stripped:
        return []

    tokens: list[Token] = []
    for word in _get_tagger()(stripped):
        if not word.surface.strip():
            continue

        raw_reading = _feature_value(word, "kana", "pron", "kanaBase", "pronBase")
        reading = katakana_to_hiragana(raw_reading) if raw_reading else None

        raw_lemma = _feature_value(word, "lemma", "orthBase", "orth")
        lemma = _clean_lemma(raw_lemma) if raw_lemma else word.surface
        pos1 = _feature_value(word, "pos1")

        tokens.append(
            Token(
                surface=word.surface,
                reading=reading,
                lemma=lemma,
                part_of_speech=pos1,
                is_content_word=pos1 in _CONTENT_POS,
            )
        )
    return tokens
