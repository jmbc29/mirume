"""Translation for Mirume, backed by DeepL with a Claude fallback.

Mirume's ``/hover`` pipeline is bidirectional: Japanese text gets explained in
English (:mod:`jlpt`), English text gets translated to Japanese and *then*
explained the same way, so the learner always lands on a JLPT-graded
breakdown regardless of which language they hovered. :func:`english_to_japanese`
is that second path — one DeepL call for the primary translation, three more
for a casual → formal spread of alternative phrasings, and a run through
:func:`jlpt.classify_tokens` so the result carries the same JLPT metadata as
native Japanese text.

:func:`japanese_to_english` is the other direction — a plain-English rendering
of the detected sentence for the hover card. It prefers DeepL and falls back to
the Anthropic API (``claude-sonnet-4-6``) when ``DEEPL_API_KEY`` is not set, so
a real full-sentence translation is available with either credential.

Configure via ``backend/.env`` (loaded here with :mod:`dotenv`):
``DEEPL_API_KEY`` (free tier at https://www.deepl.com/pro-api) and/or
``ANTHROPIC_API_KEY``. With neither, :func:`japanese_to_english` returns
``None`` and the card omits the translation line.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

import deepl
from dotenv import load_dotenv

from jlpt import classify_tokens
from jlpt import jlpt_name as _jlpt_name
from paths import ENV_FILES
from tokeniser import tokenise

# Load whichever ``.env`` exists — the Application Support copy in a packaged
# app, or the one next to this source file in development (see paths.ENV_FILES).
for _env_file in ENV_FILES:
    if _env_file.is_file():
        load_dotenv(_env_file)
        break

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


#: Model used for the Claude translation fallback (see japanese_to_english).
#: Only reached when DeepL is not configured.
_CLAUDE_TRANSLATION_MODEL = "claude-sonnet-4-6"

_CLAUDE_TRANSLATION_SYSTEM = (
    "You are a translation engine. Translate the user's Japanese text into "
    "natural, fluent English. Output only the English translation — no notes, "
    "no preamble, no quotation marks, no romaji."
)

#: Successful translations keyed by source text. /hover fires this on every
#: Japanese hover; the same sentence is hovered repeatedly (re-reading, OCR
#: retries), so a small cache spares both APIs the repeat calls and the user
#: the added latency. Failures are never cached — a missing key added later,
#: or a transient network error, should not stick.
_JA_EN_CACHE: dict[str, str] = {}
_JA_EN_CACHE_MAX = 256


def _anthropic_configured() -> bool:
    """Whether a Claude credential is present in the environment.

    Checked before attempting the fallback so a backend with neither key set
    doesn't pay a doomed API round-trip (~0.5 s) on every Japanese hover.
    """
    return bool(
        os.environ.get("ANTHROPIC_API_KEY", "").strip()
        or os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
    )


@lru_cache(maxsize=1)
def _get_anthropic_client():
    """Return a cached ``anthropic.Anthropic`` client.

    Imported lazily so the module still loads if the package is absent. The
    client reads ``ANTHROPIC_API_KEY`` (put it in ``backend/.env``).
    """
    import anthropic

    # This sits in the /hover request path — keep it snappy: one retry, a
    # 10 s ceiling, then japanese_to_english returns None and the card shows
    # without the translation line rather than stalling.
    return anthropic.Anthropic(max_retries=1, timeout=10.0)


def _claude_japanese_to_english(text: str) -> str:
    """Translate Japanese ``text`` to English with Claude.

    The DeepL fallback: one non-streaming Messages call, response text only.

    Args:
        text: Japanese source text.

    Returns:
        The English translation (stripped).
    """
    client = _get_anthropic_client()
    response = client.messages.create(
        model=_CLAUDE_TRANSLATION_MODEL,
        max_tokens=1024,
        system=_CLAUDE_TRANSLATION_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def japanese_to_english(text: str) -> str | None:
    """Translate Japanese ``text`` to English for the hover card.

    DeepL when ``DEEPL_API_KEY`` is set; otherwise Claude
    (``claude-sonnet-4-6``). Both return a real full-sentence translation.

    Args:
        text: Japanese source text.

    Returns:
        The English translation, or ``None`` when no translator is configured
        or the call fails — the card just omits the translation line then.
    """
    text = text.strip()
    if not text:
        return None
    if text in _JA_EN_CACHE:
        return _JA_EN_CACHE[text]

    result: str | None = None
    try:
        client = _get_client()
    except TranslatorNotConfiguredError:
        client = None
    except Exception:
        client = None
    if client is not None:
        try:
            # DeepL rejects a bare "EN" target — it wants a regional variant.
            result = client.translate_text(
                text, source_lang="JA", target_lang="EN-US"
            ).text
        except Exception:
            result = None

    if not result and _anthropic_configured():
        try:
            result = _claude_japanese_to_english(text) or None
        except Exception:
            result = None

    if result:
        try:
            if len(_JA_EN_CACHE) >= _JA_EN_CACHE_MAX:
                _JA_EN_CACHE.pop(next(iter(_JA_EN_CACHE)))
        except (KeyError, StopIteration):
            pass  # raced with another hover thread — the cache is best-effort
        _JA_EN_CACHE[text] = result
    return result


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
