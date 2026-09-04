"""JMdict + kanjidic2 dictionary store and JLPT classification engine.

This module owns everything to do with turning raw Japanese words into
JLPT-graded dictionary information:

* It downloads the source data (JMdict English edition, kanjidic2, plus
  supplementary JLPT vocab/kanji lists) into ``data/`` and parses it into the
  ``jmdict.db`` SQLite database, into two tables:

  - ``Entry`` – one row per written form of a JMdict entry, with columns
    ``id`` (JMdict ``ent_seq``), ``kanji``, ``reading``, ``meaning`` and
    ``jlpt_level`` (integer 1–5, where 5 = N5 easiest and 1 = N1 hardest;
    ``NULL`` when the word is not on any JLPT list).
  - ``Kanji`` – one row per kanji character, with columns ``literal``,
    ``meaning``, ``reading_on``, ``reading_kun`` and ``jlpt_level``.

* It exposes the lookup API used by the rest of the backend:

  - :func:`lookup` – surface word → reading / meaning / JLPT level.
  - :func:`classify_tokens` – annotate :class:`tokeniser.Token` objects.
  - :func:`get_kanji_breakdown` – per-kanji JLPT level and meaning for a word.

JMdict has no native JLPT tags, so word-level levels are derived by matching
JMdict writings/readings against the (unofficial, pre-2010, community-standard)
Tanos JLPT vocabulary lists. Kanji levels come from the estimated *new* JLPT
kanji list in ``davidluzgouveia/kanji-data``, falling back to kanjidic2's own
(old, 4-level) ``<jlpt>`` tag.

Build the database once with::

    python jlpt.py build          # or  python jlpt.py build --force
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import requests
from sqlalchemy import Boolean, Integer, String, Text, delete, insert, or_, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from database import DATA_DIR, JMdictBase, JMdictSession, jmdict_engine
from tokeniser import Token, tokenise

# --------------------------------------------------------------------------- #
# Source data locations
# --------------------------------------------------------------------------- #

#: JMdict, English glosses only, gzip-compressed XML.
JMDICT_URL = "http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz"
#: kanjidic2, gzip-compressed XML.
KANJIDIC2_URL = "http://www.edrdg.org/kanjidic/kanjidic2.xml.gz"
#: Tanos JLPT vocabulary lists (``{level}`` is 1–5), CSV: expression,reading,meaning,tags.
JLPT_VOCAB_URL = "https://raw.githubusercontent.com/elzup/jlpt-word-list/master/src/n{level}.csv"
#: Per-kanji data including an estimated *new* JLPT level (``jlpt_new``).
KANJI_DATA_URL = "https://raw.githubusercontent.com/davidluzgouveia/kanji-data/master/kanji.json"

JMDICT_XML: Path = DATA_DIR / "jmdict_e.xml"
KANJIDIC2_XML: Path = DATA_DIR / "kanjidic2.xml"
JLPT_VOCAB_DIR: Path = DATA_DIR / "jlpt_vocab"
KANJI_DATA_JSON: Path = DATA_DIR / "kanji_data.json"

#: JMdict priority codes that mark a writing/reading as "common".
_COMMON_PRI: frozenset[str] = frozenset(
    {"news1", "ichi1", "spec1", "spec2", "gai1"}
)

#: Maps the internal integer level to its JLPT name.
JLPT_INT_TO_NAME: dict[int, str] = {5: "N5", 4: "N4", 3: "N3", 2: "N2", 1: "N1"}

#: Rough kanjidic2 (old 4-level) → new 5-level fallback, used only when the
#: estimated new-JLPT list has no entry for a kanji.
_OLD_JLPT_TO_NEW: dict[int, int] = {4: 5, 3: 4, 2: 3, 1: 2}

#: Hiragana/katakana/CJK range used to detect whether a token surface contains
#: any Japanese characters at all — OCR noise (page numbers, stray Latin/ASCII
#: symbols) tokenises into "words" with no Japanese in them whatsoever.
_JAPANESE_CHAR_RE = re.compile(r"[぀-鿿]")

#: unidic ``pos1`` values for function words / symbols. These are structural,
#: not vocabulary to learn, so :func:`classify_tokens` does not attach a
#: dictionary meaning or JLPT level to them (grammar patterns are handled
#: separately by ``grammar.py``). Without this filter, homographs such as the
#: topic particle は match unrelated content words (歯, "tooth").
_FUNCTION_POS: frozenset[str] = frozenset(
    {"助詞", "助動詞", "補助記号", "記号", "空白", "フィラー"}
)


def jlpt_name(level: int | None) -> str:
    """Return the JLPT name (``"N5"``…``"N1"``) for an integer level.

    Args:
        level: Internal level integer (5 = N5 … 1 = N1), or ``None``.

    Returns:
        ``"N5"``–``"N1"`` for a known level, otherwise ``"unknown"``.
    """
    return JLPT_INT_TO_NAME.get(level, "unknown") if level is not None else "unknown"


def _is_kanji(char: str) -> bool:
    """Return ``True`` if ``char`` is a CJK ideograph (a kanji).

    Covers the CJK Unified Ideographs block and Extension A, which together
    include every kanji in kanjidic2.

    Args:
        char: A single-character string.

    Returns:
        Whether the character is a kanji.
    """
    return "一" <= char <= "鿿" or "㐀" <= char <= "䶿"


# --------------------------------------------------------------------------- #
# ORM models (jmdict.db)
# --------------------------------------------------------------------------- #


class Entry(JMdictBase):
    """One written form of a JMdict dictionary entry.

    A JMdict entry may have several kanji writings and several kana readings;
    this table stores one row per kanji writing (or a single row with
    ``kanji = NULL`` for kana-only entries), which keeps surface-form lookups a
    simple indexed equality check. ``id`` is therefore **not** unique — all
    rows sharing a JMdict ``ent_seq`` describe the same entry.

    Attributes:
        pk: Surrogate autoincrement primary key (implementation detail).
        id: JMdict ``ent_seq`` — the stable dictionary entry id.
        kanji: A single kanji writing of the entry, or ``None`` for kana-only
            entries.
        reading: The entry's primary kana reading (used for furigana).
        meaning: English glosses, first few senses joined with ``"; "``.
        jlpt_level: 5 (N5) … 1 (N1), or ``None`` if the word is on no JLPT list.
        common: ``True`` if any writing/reading carries a JMdict "common"
            priority code — used to rank lookup matches.
    """

    __tablename__ = "Entry"

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[int] = mapped_column(Integer, index=True)
    kanji: Mapped[str | None] = mapped_column(String(64), index=True, default=None)
    reading: Mapped[str] = mapped_column(String(64), index=True)
    meaning: Mapped[str] = mapped_column(Text, default="")
    jlpt_level: Mapped[int | None] = mapped_column(Integer, index=True, default=None)
    common: Mapped[bool] = mapped_column(Boolean, default=False)

    def __repr__(self) -> str:
        """Return a concise developer-facing representation."""
        return (
            f"Entry(id={self.id!r}, kanji={self.kanji!r}, "
            f"reading={self.reading!r}, jlpt_level={self.jlpt_level!r})"
        )


class Kanji(JMdictBase):
    """One kanji character and its readings, meanings and JLPT level.

    Attributes:
        literal: The kanji itself (primary key).
        meaning: English meanings joined with ``"; "``.
        reading_on: On'yomi readings joined with ``"、"``.
        reading_kun: Kun'yomi readings joined with ``"、"``.
        jlpt_level: Estimated *new* JLPT level, 5 (N5) … 1 (N1), or ``None``.
    """

    __tablename__ = "Kanji"

    literal: Mapped[str] = mapped_column(String(4), primary_key=True)
    meaning: Mapped[str] = mapped_column(Text, default="")
    reading_on: Mapped[str] = mapped_column(Text, default="")
    reading_kun: Mapped[str] = mapped_column(Text, default="")
    jlpt_level: Mapped[int | None] = mapped_column(Integer, index=True, default=None)

    def __repr__(self) -> str:
        """Return a concise developer-facing representation."""
        return f"Kanji(literal={self.literal!r}, jlpt_level={self.jlpt_level!r})"


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ClassifiedToken:
    """A :class:`tokeniser.Token` annotated with dictionary + JLPT data.

    Attributes:
        surface: Word as it appeared on screen.
        reading: Hiragana reading from the tokeniser (context-specific).
        lemma: Dictionary (base) form from the tokeniser.
        part_of_speech: Coarse Japanese POS tag.
        is_content_word: Whether this is a noun/verb/adjective/adverb.
        jlpt_level: 5 (N5) … 1 (N1), or ``None`` if unknown / not on a list.
        jlpt_name: ``"N5"``…``"N1"`` or ``"unknown"``.
        jlpt_estimated: ``True`` when ``jlpt_level`` was not taken from a JLPT
            vocab list but estimated from the word's hardest kanji.
        meaning: English glosses for the matched dictionary entry, or ``None``.
        dictionary_reading: The matched entry's primary reading, or ``None``.
        jmdict_id: JMdict ``ent_seq`` of the matched entry, or ``None``.
    """

    surface: str
    reading: str | None
    lemma: str
    part_of_speech: str | None
    is_content_word: bool
    jlpt_level: int | None
    jlpt_name: str
    jlpt_estimated: bool
    meaning: str | None
    dictionary_reading: str | None
    jmdict_id: int | None


@dataclass(frozen=True, slots=True)
class KanjiInfo:
    """JLPT level and readings/meaning for a single kanji character.

    Attributes:
        literal: The kanji character.
        meaning: English meanings joined with ``"; "`` (``""`` if unknown).
        reading_on: On'yomi readings joined with ``"、"``.
        reading_kun: Kun'yomi readings joined with ``"、"``.
        jlpt_level: Estimated new JLPT level, 5 … 1, or ``None``.
        jlpt_name: ``"N5"``…``"N1"`` or ``"unknown"``.
    """

    literal: str
    meaning: str
    reading_on: str
    reading_kun: str
    jlpt_level: int | None
    jlpt_name: str


# --------------------------------------------------------------------------- #
# Downloading source data
# --------------------------------------------------------------------------- #


def _download(url: str, dest: Path, *, gunzip: bool = False, force: bool = False) -> Path:
    """Download ``url`` to ``dest`` unless it already exists.

    Args:
        url: HTTP(S) URL to fetch.
        dest: Local path to write.
        gunzip: If ``True``, gzip-decompress the response before writing.
        force: Re-download even if ``dest`` exists.

    Returns:
        ``dest``.
    """
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    payload = gzip.decompress(response.content) if gunzip else response.content
    dest.write_bytes(payload)
    return dest


def download_sources(force: bool = False) -> None:
    """Ensure every raw data file is present in ``data/``.

    Downloads JMdict, kanjidic2, the five Tanos JLPT vocab CSVs and the kanji
    data JSON. Files that already exist are left alone unless ``force`` is set.

    Args:
        force: Re-download everything even if already present.
    """
    _download(JMDICT_URL, JMDICT_XML, gunzip=True, force=force)
    _download(KANJIDIC2_URL, KANJIDIC2_XML, gunzip=True, force=force)
    _download(KANJI_DATA_URL, KANJI_DATA_JSON, force=force)
    for level in (5, 4, 3, 2, 1):
        _download(
            JLPT_VOCAB_URL.format(level=level),
            JLPT_VOCAB_DIR / f"n{level}.csv",
            force=force,
        )


# --------------------------------------------------------------------------- #
# Building the JLPT lookup maps from supplementary lists
# --------------------------------------------------------------------------- #


def _load_jlpt_vocab() -> tuple[dict[str, int], dict[str, int]]:
    """Load the Tanos JLPT vocab lists into surface/reading → level maps.

    Levels are processed hardest (N1) to easiest (N5) so that, when a word
    appears on more than one list, the **easiest** level wins — the app cares
    about "the learner has probably met this by level X".

    Returns:
        A ``(by_expression, by_reading)`` pair of dicts mapping a written form
        or a kana reading to its integer JLPT level (5 … 1).
    """
    by_expression: dict[str, int] = {}
    by_reading: dict[str, int] = {}
    for level in (1, 2, 3, 4, 5):
        path = JLPT_VOCAB_DIR / f"n{level}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                expression = (row.get("expression") or "").strip()
                reading = (row.get("reading") or "").strip()
                if expression:
                    by_expression[expression] = level
                if reading:
                    by_reading[reading] = level
    return by_expression, by_reading


def _load_kanji_jlpt() -> dict[str, int]:
    """Load the estimated *new* JLPT level for each kanji from the kanji JSON.

    Returns:
        A dict mapping a kanji character to its integer JLPT level (5 … 1).
        Kanji without a ``jlpt_new`` value are omitted.
    """
    data: dict[str, dict] = json.loads(KANJI_DATA_JSON.read_text(encoding="utf-8"))
    return {
        literal: info["jlpt_new"]
        for literal, info in data.items()
        if isinstance(info.get("jlpt_new"), int)
    }


# --------------------------------------------------------------------------- #
# Parsing the XML sources
# --------------------------------------------------------------------------- #


def parse_jmdict(
    xml_path: Path,
    by_expression: dict[str, int],
    by_reading: dict[str, int],
) -> Iterator[dict]:
    """Stream JMdict XML into row dicts for the :class:`Entry` table.

    ``xml.etree.ElementTree`` expands JMdict's internal DTD entities (``&v1;``
    etc.) automatically, so the file is parsed directly with ``iterparse`` and
    each ``<entry>`` element is cleared after use to keep memory flat.

    Each entry yields one dict per kanji writing (or a single ``kanji=None``
    dict for kana-only entries). ``jlpt_level`` is the easiest level found
    across *all* of the entry's writings and readings.

    Args:
        xml_path: Path to the decompressed ``jmdict_e.xml``.
        by_expression: Written-form → JLPT level map from :func:`_load_jlpt_vocab`.
        by_reading: Reading → JLPT level map from :func:`_load_jlpt_vocab`.

    Yields:
        Dicts with keys ``id``, ``kanji``, ``reading``, ``meaning``,
        ``jlpt_level``, ``common``.
    """
    for _event, element in ET.iterparse(str(xml_path), events=("end",)):
        if element.tag != "entry":
            continue

        ent_seq = int(element.findtext("ent_seq", "0"))
        kanji_forms = [k.text for k in element.findall("k_ele/keb") if k.text]
        readings = [r.text for r in element.findall("r_ele/reb") if r.text]
        primary_reading = readings[0] if readings else ""

        sense_glosses: list[str] = []
        for sense in element.findall("sense")[:3]:
            glosses = [g.text for g in sense.findall("gloss") if g.text]
            if glosses:
                sense_glosses.append(", ".join(glosses))
        meaning = "; ".join(sense_glosses)[:300]

        found_levels = [
            level
            for form in (*kanji_forms, *readings)
            for level in (by_expression.get(form),)
            if level is not None
        ]
        found_levels += [
            level for r in readings for level in (by_reading.get(r),) if level is not None
        ]
        jlpt_level = max(found_levels) if found_levels else None

        priorities = [
            p.text
            for p in (*element.findall("k_ele/ke_pri"), *element.findall("r_ele/re_pri"))
        ]
        common = any(p in _COMMON_PRI for p in priorities)

        element.clear()

        if kanji_forms:
            for form in kanji_forms:
                yield {
                    "id": ent_seq,
                    "kanji": form,
                    "reading": primary_reading,
                    "meaning": meaning,
                    "jlpt_level": jlpt_level,
                    "common": common,
                }
        else:
            yield {
                "id": ent_seq,
                "kanji": None,
                "reading": primary_reading,
                "meaning": meaning,
                "jlpt_level": jlpt_level,
                "common": common,
            }


def parse_kanjidic(xml_path: Path, kanji_jlpt: dict[str, int]) -> Iterator[dict]:
    """Stream kanjidic2 XML into row dicts for the :class:`Kanji` table.

    Args:
        xml_path: Path to the decompressed ``kanjidic2.xml``.
        kanji_jlpt: Kanji → estimated new JLPT level map from
            :func:`_load_kanji_jlpt`.

    Yields:
        Dicts with keys ``literal``, ``meaning``, ``reading_on``,
        ``reading_kun``, ``jlpt_level``.
    """
    for _event, element in ET.iterparse(str(xml_path), events=("end",)):
        if element.tag != "character":
            continue

        literal = element.findtext("literal") or ""
        on_readings: list[str] = []
        kun_readings: list[str] = []
        meanings: list[str] = []
        for rmgroup in element.findall("reading_meaning/rmgroup"):
            for reading in rmgroup.findall("reading"):
                if reading.text is None:
                    continue
                if reading.get("r_type") == "ja_on":
                    on_readings.append(reading.text)
                elif reading.get("r_type") == "ja_kun":
                    kun_readings.append(reading.text)
            for meaning in rmgroup.findall("meaning"):
                if meaning.get("m_lang") is None and meaning.text:
                    meanings.append(meaning.text)

        old_jlpt = element.findtext("misc/jlpt")
        element.clear()

        level = kanji_jlpt.get(literal)
        if level is None and old_jlpt:
            level = _OLD_JLPT_TO_NEW.get(int(old_jlpt))

        if not literal:
            continue
        yield {
            "literal": literal,
            "meaning": "; ".join(meanings),
            "reading_on": "、".join(on_readings),
            "reading_kun": "、".join(kun_readings),
            "jlpt_level": level,
        }


# --------------------------------------------------------------------------- #
# Building the database
# --------------------------------------------------------------------------- #


def _bulk_insert(session: Session, model: type, rows: Iterable[dict], batch: int = 5000) -> int:
    """Insert ``rows`` into ``model``'s table in batches.

    Args:
        session: An open ``jmdict.db`` session.
        model: The ORM model class to insert into.
        rows: An iterable of column-name → value dicts.
        batch: Number of rows per ``executemany`` call.

    Returns:
        The total number of rows inserted.
    """
    pending: list[dict] = []
    total = 0
    for row in rows:
        pending.append(row)
        if len(pending) >= batch:
            session.execute(insert(model), pending)
            total += len(pending)
            pending.clear()
    if pending:
        session.execute(insert(model), pending)
        total += len(pending)
    return total


def build_database(force: bool = False) -> None:
    """Download the source data and (re)build ``Entry`` and ``Kanji``.

    Existing rows in both tables are deleted first, so this is safe to re-run.
    Takes on the order of a minute on first run (≈12 MB download + parsing
    ~200k JMdict entries and ~13k kanji).

    Args:
        force: Also re-download the raw source files.
    """
    print("Building jmdict.db …")
    download_sources(force=force)

    by_expression, by_reading = _load_jlpt_vocab()
    kanji_jlpt = _load_kanji_jlpt()
    print(
        f"  JLPT vocab entries: {len(by_expression)} written forms, "
        f"{len(by_reading)} readings; kanji with new-JLPT level: {len(kanji_jlpt)}"
    )

    JMdictBase.metadata.create_all(bind=jmdict_engine)
    with JMdictSession() as session:
        session.execute(delete(Entry))
        session.execute(delete(Kanji))
        session.commit()

        entries = _bulk_insert(
            session, Entry, parse_jmdict(JMDICT_XML, by_expression, by_reading)
        )
        session.commit()
        kanji = _bulk_insert(
            session, Kanji, parse_kanjidic(KANJIDIC2_XML, kanji_jlpt)
        )
        session.commit()

    print(f"  inserted {entries} Entry rows and {kanji} Kanji rows")


def dictionary_ready() -> bool:
    """Return ``True`` if the ``Entry`` table exists and has at least one row.

    Used by the server to decide whether to warn that :func:`build_database`
    still needs to be run.
    """
    try:
        with JMdictSession() as session:
            return session.execute(select(Entry.pk).limit(1)).first() is not None
    except Exception:  # noqa: BLE001 — table missing, db locked, etc. all mean "not ready"
        return False


# --------------------------------------------------------------------------- #
# Lookup API
# --------------------------------------------------------------------------- #


def _estimate_level_from_kanji(word: str) -> int | None:
    """Estimate a word's JLPT level from the levels of its kanji.

    Used as a fallback for words that are not on any JLPT vocab list (e.g.
    ``日本語``, which the community lists treat as ``日本`` + ``語``). A word is
    taken to be at least as hard as its hardest constituent kanji, so this
    returns the hardest (lowest-numbered) known kanji level.

    Args:
        word: A surface or dictionary form.

    Returns:
        The estimated integer level (5 … 1), or ``None`` if the word has no
        kanji or none of them have a known level.
    """
    kanji_chars = [c for c in dict.fromkeys(word) if _is_kanji(c)]
    if not kanji_chars:
        return None
    with JMdictSession() as session:
        levels = [
            level
            for (level,) in session.execute(
                select(Kanji.jlpt_level).where(Kanji.literal.in_(kanji_chars))
            )
            if level is not None
        ]
    return min(levels) if levels else None


def _rank(entry: Entry, matched_on_kanji: bool) -> tuple[int, int, int]:
    """Return a sort key for choosing the best :class:`Entry` among matches.

    Preference order: matched on a kanji writing > matched on reading;
    common word > rare word; has a JLPT level > has none.

    Args:
        entry: A candidate entry.
        matched_on_kanji: Whether the search key equalled ``entry.kanji``.

    Returns:
        A tuple that sorts higher for better matches.
    """
    return (
        1 if matched_on_kanji else 0,
        1 if entry.common else 0,
        1 if entry.jlpt_level is not None else 0,
    )


def lookup(word: str) -> dict:
    """Look up a single surface word in JMdict.

    Tries an exact match on a kanji writing first, then on a kana reading, so
    both ``食べる`` and ``たべる`` resolve. Among multiple matches the "best" is
    chosen by :func:`_rank`.

    Args:
        word: A surface or dictionary form, e.g. ``"日本語"`` or ``"難しい"``.

    Returns:
        A dict with keys ``word``, ``found`` (bool), ``kanji``, ``reading``,
        ``meaning``, ``jlpt_level`` (int 5…1 or ``None``), ``jlpt_name``,
        ``jlpt_estimated`` (bool — level came from the word's kanji, not a vocab
        list), ``common`` and ``id`` (JMdict ``ent_seq`` or ``None``). When the
        word is not in the dictionary, ``found`` is ``False`` and the data
        fields are ``None`` / ``"unknown"``.
    """
    with JMdictSession() as session:
        rows = session.execute(select(Entry).where(Entry.kanji == word)).scalars().all()
        matched_on_kanji = bool(rows)
        if not rows:
            rows = (
                session.execute(select(Entry).where(Entry.reading == word))
                .scalars()
                .all()
            )

    if not rows:
        return {
            "word": word,
            "found": False,
            "kanji": None,
            "reading": None,
            "meaning": None,
            "jlpt_level": None,
            "jlpt_name": "unknown",
            "jlpt_estimated": False,
            "common": False,
            "id": None,
        }

    best = max(rows, key=lambda r: _rank(r, matched_on_kanji))
    level = best.jlpt_level
    estimated = False
    if level is None:
        level = _estimate_level_from_kanji(best.kanji or word)
        estimated = level is not None
    return {
        "word": word,
        "found": True,
        "kanji": best.kanji,
        "reading": best.reading,
        "meaning": best.meaning or None,
        "jlpt_level": level,
        "jlpt_name": jlpt_name(level),
        "jlpt_estimated": estimated,
        "common": best.common,
        "id": best.id,
    }


def classify_tokens(tokens: list[Token]) -> list[ClassifiedToken]:
    """Annotate tokeniser output with JMdict meaning and JLPT level.

    Performs a single batched query for every surface form and lemma in
    ``tokens``, then attaches the best matching entry to each token. Content
    words with no JLPT-list level fall back to an estimate from their kanji
    (``jlpt_estimated=True``); particles, punctuation and unknown words come
    back with ``jlpt_level=None`` and ``jlpt_name="unknown"``.

    Args:
        tokens: The list returned by :func:`tokeniser.tokenise`.

    Returns:
        A list of :class:`ClassifiedToken` in the same order as ``tokens``.
    """
    def _enrich(token: Token) -> bool:
        """Whether this token should get dictionary / JLPT data attached."""
        if token.part_of_speech in _FUNCTION_POS:
            return False
        return bool(_JAPANESE_CHAR_RE.search(token.surface))

    keys: set[str] = set()
    for token in tokens:
        if not _enrich(token):
            continue
        keys.add(token.surface)
        keys.add(token.lemma)
    keys.discard("")

    best_by_key: dict[str, tuple[tuple[int, int, int], Entry]] = {}
    if keys:
        with JMdictSession() as session:
            rows = (
                session.execute(
                    select(Entry).where(
                        or_(Entry.kanji.in_(keys), Entry.reading.in_(keys))
                    )
                )
                .scalars()
                .all()
            )
        for entry in rows:
            for key, on_kanji in ((entry.kanji, True), (entry.reading, False)):
                if not key or key not in keys:
                    continue
                candidate = (_rank(entry, on_kanji), entry)
                current = best_by_key.get(key)
                if current is None or candidate[0] > current[0]:
                    best_by_key[key] = candidate

    classified: list[ClassifiedToken] = []
    for token in tokens:
        match = (
            (best_by_key.get(token.lemma) or best_by_key.get(token.surface))
            if _enrich(token)
            else None
        )
        entry = match[1] if match else None

        level = entry.jlpt_level if entry else None
        estimated = False
        if level is None and token.is_content_word:
            level = _estimate_level_from_kanji(token.lemma) or _estimate_level_from_kanji(
                token.surface
            )
            estimated = level is not None

        classified.append(
            ClassifiedToken(
                surface=token.surface,
                reading=token.reading,
                lemma=token.lemma,
                part_of_speech=token.part_of_speech,
                is_content_word=token.is_content_word,
                jlpt_level=level,
                jlpt_name=jlpt_name(level),
                jlpt_estimated=estimated,
                meaning=(entry.meaning or None) if entry else None,
                dictionary_reading=entry.reading if entry else None,
                jmdict_id=entry.id if entry else None,
            )
        )
    return classified


def get_kanji_breakdown(word: str) -> list[KanjiInfo]:
    """Return per-kanji JLPT level and meaning for each kanji in ``word``.

    Non-kanji characters (kana, punctuation, Latin) are skipped. Repeated
    kanji appear once, in order of first occurrence.

    Args:
        word: Any string; typically a single word or short phrase.

    Returns:
        A list of :class:`KanjiInfo`, one per distinct kanji. Empty if ``word``
        contains no kanji.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for char in word:
        if _is_kanji(char) and char not in seen:
            seen.add(char)
            ordered.append(char)
    if not ordered:
        return []

    with JMdictSession() as session:
        by_literal = {
            k.literal: k
            for k in session.execute(
                select(Kanji).where(Kanji.literal.in_(ordered))
            ).scalars()
        }

    breakdown: list[KanjiInfo] = []
    for char in ordered:
        record = by_literal.get(char)
        if record is None:
            breakdown.append(KanjiInfo(char, "", "", "", None, "unknown"))
        else:
            breakdown.append(
                KanjiInfo(
                    literal=record.literal,
                    meaning=record.meaning,
                    reading_on=record.reading_on,
                    reading_kun=record.reading_kun,
                    jlpt_level=record.jlpt_level,
                    jlpt_name=jlpt_name(record.jlpt_level),
                )
            )
    return breakdown


# --------------------------------------------------------------------------- #
# Manual verification
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_database(force="--force" in sys.argv)
    elif not dictionary_ready():
        print("Dictionary database is empty — running a one-time build.\n")
        build_database()

    print("\n--- lookup() ---")
    for test_word in ("日本語", "食べる", "難しい"):
        info = lookup(test_word)
        gloss = (info["meaning"] or "")[:45]
        flag = " (est.)" if info["jlpt_estimated"] else ""
        print(
            f"{test_word:6}  {info['jlpt_name']:8}{flag:7} (level {info['jlpt_level']})  "
            f"{info['reading']}  —  {gloss}"
        )

    print("\n--- classify_tokens() ---")
    for ctoken in classify_tokens(tokenise("日本語の勉強は難しいです")):
        if ctoken.is_content_word:
            flag = " (est.)" if ctoken.jlpt_estimated else ""
            print(f"{ctoken.surface:6}  {ctoken.jlpt_name:4}{flag:7}  {ctoken.meaning}")

    print("\n--- get_kanji_breakdown() ---")
    for test_word in ("日本語", "食べる", "難しい"):
        parts = get_kanji_breakdown(test_word)
        rendered = ", ".join(f"{p.literal}={p.jlpt_name}" for p in parts)
        print(f"{test_word:6}  {rendered}")
