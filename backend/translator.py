"""English → Japanese translation for Mirume, backed by the DeepL API.

Mirume's ``/hover`` pipeline is bidirectional: Japanese text gets explained in
English (:mod:`jlpt`), English text gets translated to Japanese and *then*
explained the same way, so the learner always lands on a JLPT-graded
breakdown regardless of which language they hovered. :func:`english_to_japanese`
is that second path — one DeepL call for the primary translation, three more
for a casual → formal spread of alternative phrasings, and a run through
:func:`jlpt.classify_tokens` so the result carries the same JLPT metadata as
native Japanese text.

Requires a DeepL API key in the environment as ``DEEPL_API_KEY`` (loaded from
``backend/.env`` via :mod:`dotenv`). Get a free-tier key at
https://www.deepl.com/pro-api.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import deepl
from dotenv import load_dotenv

from jlpt import classify_tokens
from jlpt import jlpt_name as _jlpt_name
from tokeniser import tokenise

load_dotenv(Path(__file__).resolve().parent / ".env")

#: Formality variants requested for the "alternatives" spread, roughly
#: casual (N5) -> polite (N4/N3) -> formal (N2). DeepL's `formality` param
#: only takes two states, so the middle entry re-requests default formality.
_ALTERNATIVE_FORMALITY: tuple[str, ...] = ("less", "default", "more")

#: A small set of common JLPT grammar patterns, detected by simple substring/
#: regex matching against DeepL's output. Not exhaustive — a proper grammar
#: parser is future work — but enough to surface the most common patterns a
#: beginner-to-intermediate translation is likely to contain.
_GRAMMAR_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("〜たい", re.compile(r"[ぁ-ゖ]たい"), "N5"),
    ("〜てください", re.compile(r"て(ください|下さい)"), "N5"),
    ("〜ことができる", re.compile(r"ことができ"), "N4"),
    ("〜なければならない", re.compile(r"なければならない|なきゃいけない"), "N4"),
    ("〜てもいい", re.compile(r"てもいい"), "N4"),
    ("〜てはいけない", re.compile(r"てはいけない"), "N4"),
    ("〜ように", re.compile(r"ように"), "N3"),
    ("〜そうだ（伝聞）", re.compile(r"そうです|そうだ"), "N3"),
    ("〜てしまう", re.compile(r"てしまう|ちゃう|じゃう"), "N3"),
    ("〜させる（使役）", re.compile(r"させ(る|て|た)"), "N3"),
    ("〜れる／られる（受身）", re.compile(r"[らわ]れ(る|て|た)"), "N3"),
)


def find_grammar_patterns(text: str) -> list[dict[str, str]]:
    """Detect common JLPT grammar patterns appearing in ``text`` by regex match.

    Heuristic, not a real parser — intended to surface a handful of common
    beginner/intermediate constructs in a DeepL translation, not to be a
    complete grammar analyzer.

    Args:
        text: Japanese text to scan.

    Returns:
        A list of ``{"pattern": ..., "jlpt_level": ...}`` dicts, one per
        distinct pattern matched, in the order defined in
        :data:`_GRAMMAR_PATTERNS`.
    """
    return [
        {"pattern": pattern, "jlpt_level": level}
        for pattern, regex, level in _GRAMMAR_PATTERNS
        if regex.search(text)
    ]


class TranslatorNotConfiguredError(RuntimeError):
    """Raised when no ``DEEPL_API_KEY`` is set."""


@lru_cache(maxsize=1)
def _get_client() -> deepl.Translator:
    """Return a cached :class:`deepl.Translator` built from ``DEEPL_API_KEY``.

    Returns:
        A configured DeepL client.

    Raises:
        TranslatorNotConfiguredError: If ``DEEPL_API_KEY`` is unset/empty.
    """
    api_key = os.environ.get("DEEPL_API_KEY", "").strip()
    if not api_key:
        raise TranslatorNotConfiguredError(
            "DEEPL_API_KEY is not set — add it to backend/.env. "
            "Get a free-tier key at https://www.deepl.com/pro-api."
        )
    return deepl.Translator(api_key)


def _translate(text: str, *, formality: str = "default") -> str:
    """Translate English ``text`` to Japanese via DeepL.

    Args:
        text: English source text.
        formality: DeepL formality hint — ``"less"``, ``"default"`` or ``"more"``.
            Japanese is a formality-supporting target language for DeepL.

    Returns:
        The Japanese translation.
    """
    client = _get_client()
    result = client.translate_text(
        text, source_lang="EN", target_lang="JA", formality=formality
    )
    return result.text


def english_to_japanese(text: str) -> dict:
    """Translate English text to Japanese and classify the result by JLPT level.

    Args:
        text: English source text, e.g. ``"I want to eat sushi tonight."``.

    Returns:
        A dict with keys:

        * ``translation`` – the primary Japanese translation.
        * ``reading`` – hiragana reading of the translation (furigana source).
        * ``jlpt_level`` – hardest JLPT level among the translation's content
          words (``"N5"``..``"N1"``, or ``"unknown"``).
        * ``jlpt_name`` – same as ``jlpt_level`` (kept for symmetry with the
          Japanese-input pipeline's per-token ``jlpt_name`` field).
        * ``grammar_patterns`` – grammar patterns detected in the translation
          (:func:`find_grammar_patterns`).
        * ``alternatives`` – three rephrasings, casual -> formal.

    Raises:
        TranslatorNotConfiguredError: If ``DEEPL_API_KEY`` is unset/empty.
    """
    translation = _translate(text)
    tokens = tokenise(translation)
    classified = classify_tokens(tokens)
    reading = "".join(t.reading or t.surface for t in tokens)

    content_levels = [t.jlpt_level for t in classified if t.is_content_word and t.jlpt_level]
    # Numerically lower = harder (N1=1 .. N5=5); the sentence is only as easy
    # as its hardest word.
    hardest_level = min(content_levels) if content_levels else None

    alternatives = [_translate(text, formality=f) for f in _ALTERNATIVE_FORMALITY]
    # Deduplicate while preserving order — DeepL sometimes returns identical
    # text for "less"/"default"/"more" when a sentence has no formality axis.
    alternatives = list(dict.fromkeys(alternatives))

    return {
        "translation": translation,
        "reading": reading,
        "jlpt_level": _jlpt_name(hardest_level),
        "jlpt_name": _jlpt_name(hardest_level),
        "grammar_patterns": find_grammar_patterns(translation),
        "alternatives": alternatives,
    }


if __name__ == "__main__":
    for sample in (
        "I want to eat sushi tonight.",
        "Could you help me with my homework?",
    ):
        try:
            result = english_to_japanese(sample)
        except TranslatorNotConfiguredError as exc:
            print(exc)
            break
        print(f"EN: {sample}")
        print(f"JA: {result['translation']}  ({result['reading']})")
        print(f"Level: {result['jlpt_level']}")
        print(f"Alternatives: {result['alternatives']}")
        print(f"Grammar: {result['grammar_patterns']}")
        print()
