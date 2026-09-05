"""Tatoeba example-sentence store, in ``jmdict.db``.

Loads Japanese sentences that have an English translation into two tables so
:func:`jlpt.get_example_sentences` can look up "sentences containing this
word" without re-tokenising anything at request time:

* ``example_sentences`` – one row per Japanese/English sentence pair.
* ``word_sentences`` – ``(word, sentence_id)`` pairs, one per word that
  appears in that sentence, built from Tatoeba's own word-segmented index
  rather than running the sentence back through :mod:`tokeniser`.

Source data and a deliberate deviation from the literal spec
--------------------------------------------------------------
The two files originally asked for — the top-level ``sentences.tar.bz2``
(every language Tatoeba has, ~750 MB decompressed) and ``jpn_indices.tar.bz2``
— cannot actually be turned into Japanese/English *pairs* on their own:
inspecting the real ``jpn_indices.csv`` shows its second column is a legacy
Tanaka-Corpus "meaning id", not a Tatoeba sentence id, so it carries no link
to an English sentence at all. Reconstructing pairs from the two originally
named files would mean downloading and scanning the full multi-language
``sentences.tar.bz2`` for a translation link that isn't there.

Tatoeba separately publishes the exact per-language slices this needs, at a
fraction of the size:

* :data:`JPN_ENG_LINKS_URL` (~1.4 MB) – ``(japanese_id, english_id)`` pairs.
* :data:`JPN_SENTENCES_URL` (~3.4 MB) – ``jpn`` sentence text, id-keyed.
* :data:`ENG_SENTENCES_URL` (~25 MB) – ``eng`` sentence text, id-keyed.

:data:`JPN_INDICES_URL` (the originally-named ``jpn_indices.tar.bz2``, ~2.8 MB)
is still used, for its per-sentence word segmentation.

Build once with::

    python sentences.py build          # or  python sentences.py build --force
"""

from __future__ import annotations

import bz2
import io
import re
import sys
import tarfile
from pathlib import Path
from typing import Iterable

import requests
from sqlalchemy import ForeignKey, Integer, String, Text, delete, insert, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from database import DATA_DIR, JMdictBase, JMdictSession, jmdict_engine

# --------------------------------------------------------------------------- #
# Source data locations
# --------------------------------------------------------------------------- #

#: (japanese_sentence_id, english_sentence_id) pairs, tab-separated, no header.
JPN_ENG_LINKS_URL = "https://downloads.tatoeba.org/exports/per_language/jpn/jpn-eng_links.tsv.bz2"
#: Japanese sentences only: id, lang ("jpn"), text.
JPN_SENTENCES_URL = "https://downloads.tatoeba.org/exports/per_language/jpn/jpn_sentences.tsv.bz2"
#: English sentences only: id, lang ("eng"), text.
ENG_SENTENCES_URL = "https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2"
#: Per-sentence word segmentation for Japanese sentences (Tanaka Corpus
#: format): id, legacy meaning-id (unused), tagged text — see
#: :func:`_index_token_words` for the tag syntax.
JPN_INDICES_URL = "https://downloads.tatoeba.org/exports/jpn_indices.tar.bz2"

SENTENCES_DIR: Path = DATA_DIR / "tatoeba"
JPN_ENG_LINKS_TSV: Path = SENTENCES_DIR / "jpn-eng_links.tsv"
JPN_SENTENCES_TSV: Path = SENTENCES_DIR / "jpn_sentences.tsv"
ENG_SENTENCES_TSV: Path = SENTENCES_DIR / "eng_sentences.tsv"
JPN_INDICES_CSV: Path = SENTENCES_DIR / "jpn_indices.csv"

#: Hiragana/katakana/CJK range, used to reject markup fragments (reference
#: numbers, empty groups) that aren't actually words.
_JAPANESE_CHAR_RE = re.compile(r"[぀-鿿]")

#: Tanaka-Corpus index token syntax: ``Base(Reading)[Sense]{ActualForm}~``,
#: every group but the base optional. E.g. ``為る(する){して}`` (base 為る,
#: alternate reading する, actual inflected form して) or ``直ぐに{すぐに}``
#: (base 直ぐに, actual form すぐに). ``[...]`` is a sense/reference number
#: (e.g. ``[01]``, or ``(#2028930)`` for a cross-reference) and carries no
#: word text, so it's matched but discarded. A trailing ``~`` marks a
#: compound-word fragment and is likewise stripped, not captured.
_INDEX_TOKEN_RE = re.compile(
    r"^(?P<base>[^(){}\[\]~]*)"
    r"(?:\((?P<paren>[^)]*)\))?"
    r"(?:\[[^\]]*\])?"
    r"(?:\{(?P<curly>[^}]*)\})?"
    r"~?$"
)


# --------------------------------------------------------------------------- #
# ORM models (jmdict.db)
# --------------------------------------------------------------------------- #


class ExampleSentence(JMdictBase):
    """One Tatoeba Japanese sentence with its English translation.

    Attributes:
        id: The Japanese sentence's own Tatoeba sentence id (reused directly
            rather than a surrogate key, since :class:`WordSentence` rows —
            built from Tatoeba's own per-sentence word index — already refer
            to sentences by this same id).
        japanese: The Japanese sentence text.
        english: The linked English translation text.
        source: Always ``"tatoeba"`` for now; a plain column rather than a
            constant so a future manually-curated sentence can share the
            table.
    """

    __tablename__ = "example_sentences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    japanese: Mapped[str] = mapped_column(Text)
    english: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(32), default="tatoeba")

    def __repr__(self) -> str:
        """Return a concise developer-facing representation."""
        preview = (self.japanese[:20] + "…") if len(self.japanese) > 20 else self.japanese
        return f"ExampleSentence(id={self.id!r}, japanese={preview!r})"


class WordSentence(JMdictBase):
    """One (word, sentence) pairing used to look up example sentences by word.

    Attributes:
        word: A word form (base or inflected) that appears in the sentence.
        sentence_id: The :class:`ExampleSentence` it appears in.
    """

    __tablename__ = "word_sentences"

    word: Mapped[str] = mapped_column(String(64), primary_key=True)
    sentence_id: Mapped[int] = mapped_column(
        ForeignKey("example_sentences.id"), primary_key=True
    )


# --------------------------------------------------------------------------- #
# Downloading source data
# --------------------------------------------------------------------------- #


def _download_bz2(url: str, dest: Path, *, force: bool = False) -> Path:
    """Download a ``.bz2``-compressed text file and decompress it to ``dest``.

    Args:
        url: HTTP(S) URL of the ``.bz2`` file.
        dest: Local path to write the decompressed content to.
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
    dest.write_bytes(bz2.decompress(response.content))
    return dest


def _download_tar_bz2_member(
    url: str, member_name: str, dest: Path, *, force: bool = False
) -> Path:
    """Download a ``.tar.bz2`` archive and extract one member to ``dest``.

    Args:
        url: HTTP(S) URL of the ``.tar.bz2`` archive.
        member_name: Name of the file inside the archive to extract.
        dest: Local path to write the extracted member to.
        force: Re-download even if ``dest`` exists.

    Returns:
        ``dest``.

    Raises:
        RuntimeError: ``member_name`` is not present in the archive.
    """
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:bz2") as tar:
        extracted = tar.extractfile(member_name)
        if extracted is None:
            raise RuntimeError(f"{member_name!r} not found in {url}")
        dest.write_bytes(extracted.read())
    return dest


def download_sources(force: bool = False) -> None:
    """Ensure every raw Tatoeba export is present in ``data/tatoeba/``.

    Args:
        force: Re-download everything even if already present.
    """
    _download_bz2(JPN_ENG_LINKS_URL, JPN_ENG_LINKS_TSV, force=force)
    _download_bz2(JPN_SENTENCES_URL, JPN_SENTENCES_TSV, force=force)
    _download_bz2(ENG_SENTENCES_URL, ENG_SENTENCES_TSV, force=force)
    _download_tar_bz2_member(JPN_INDICES_URL, "jpn_indices.csv", JPN_INDICES_CSV, force=force)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _index_token_words(token: str) -> list[str]:
    """Extract plain word forms from one Tanaka-Corpus index token.

    See :data:`_INDEX_TOKEN_RE` for the tag syntax. A token typically yields
    one word (its base form) but can yield two — e.g. ``為る(する){して}``
    contributes both ``為る`` (dictionary form) and ``する`` (the reading/
    common form in parens) — so lookups by either the lemma or the everyday
    written form succeed. A reference marker such as ``(#2028930)`` is
    recognised and dropped rather than indexed as a word.

    Args:
        token: One whitespace-separated token from a ``jpn_indices.csv`` line.

    Returns:
        Zero or more distinct word forms, each containing at least one
        Japanese character.
    """
    match = _INDEX_TOKEN_RE.match(token)
    if not match:
        return [token] if _JAPANESE_CHAR_RE.search(token) else []

    words: list[str] = []
    base = match.group("base")
    paren = match.group("paren")
    curly = match.group("curly")
    if base and _JAPANESE_CHAR_RE.search(base):
        words.append(base)
    if paren and not paren.startswith("#") and _JAPANESE_CHAR_RE.search(paren):
        words.append(paren)
    if curly and _JAPANESE_CHAR_RE.search(curly):
        words.append(curly)
    return words


def _load_tsv_text_map(path: Path, *, keep_ids: set[int] | None = None) -> dict[int, str]:
    """Load a Tatoeba ``id\\tlang\\ttext`` file into an ``id -> text`` map.

    Args:
        path: Path to the decompressed ``.tsv`` file.
        keep_ids: If given, only rows whose id is in this set are kept
            (used to avoid holding every English sentence in memory when
            only ~15% of them are ever linked to a Japanese one).

    Returns:
        A dict mapping sentence id to its text.
    """
    result: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3:
                continue
            sentence_id = int(parts[0])
            if keep_ids is not None and sentence_id not in keep_ids:
                continue
            result[sentence_id] = parts[2]
    return result


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


def build_example_sentences_tables(force: bool = False) -> None:
    """Download the Tatoeba exports and (re)build ``example_sentences`` / ``word_sentences``.

    Existing rows in both tables are deleted first, so this is safe to re-run.
    Only Japanese sentences that have an English translation are kept; a
    Japanese sentence with several translations keeps just the first.

    Args:
        force: Also re-download the raw source files.
    """
    print("Building example_sentences / word_sentences …")
    download_sources(force=force)

    jpn_to_eng: dict[int, int] = {}
    with JPN_ENG_LINKS_TSV.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            jpn_id, eng_id = int(parts[0]), int(parts[1])
            jpn_to_eng.setdefault(jpn_id, eng_id)

    jpn_text = _load_tsv_text_map(JPN_SENTENCES_TSV, keep_ids=set(jpn_to_eng))
    eng_text = _load_tsv_text_map(ENG_SENTENCES_TSV, keep_ids=set(jpn_to_eng.values()))
    print(
        f"  {len(jpn_to_eng)} jpn->eng links, {len(jpn_text)} jpn texts, "
        f"{len(eng_text)} eng texts loaded"
    )

    JMdictBase.metadata.create_all(bind=jmdict_engine)
    with JMdictSession() as session:
        session.execute(delete(WordSentence))
        session.execute(delete(ExampleSentence))
        session.commit()

        valid_ids: set[int] = set()
        sentence_rows: list[dict] = []
        for jpn_id, eng_id in jpn_to_eng.items():
            japanese = jpn_text.get(jpn_id)
            english = eng_text.get(eng_id)
            if not japanese or not english:
                continue
            valid_ids.add(jpn_id)
            sentence_rows.append(
                {"id": jpn_id, "japanese": japanese, "english": english, "source": "tatoeba"}
            )
        inserted_sentences = _bulk_insert(session, ExampleSentence, sentence_rows)
        session.commit()
        print(f"  inserted {inserted_sentences} example_sentences rows")

        word_pairs: set[tuple[str, int]] = set()
        with JPN_INDICES_CSV.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t", 2)
                if len(parts) != 3:
                    continue
                sentence_id = int(parts[0])
                if sentence_id not in valid_ids:
                    continue
                for token in parts[2].split():
                    for word in _index_token_words(token):
                        word_pairs.add((word, sentence_id))

        word_rows = ({"word": word, "sentence_id": sid} for word, sid in word_pairs)
        inserted_words = _bulk_insert(session, WordSentence, word_rows)
        session.commit()
        print(f"  inserted {inserted_words} word_sentences rows")


def example_sentences_ready() -> bool:
    """Return ``True`` if ``example_sentences`` exists and has at least one row.

    Used by the server to decide whether to warn that
    :func:`build_example_sentences_tables` still needs to be run.
    """
    try:
        with JMdictSession() as session:
            return session.execute(select(ExampleSentence.id).limit(1)).first() is not None
    except Exception:  # noqa: BLE001 — table missing, db locked, etc. all mean "not ready"
        return False


# --------------------------------------------------------------------------- #
# Manual verification
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_example_sentences_tables(force="--force" in sys.argv)
    elif not example_sentences_ready():
        print("example_sentences table is empty — running a one-time build.\n")
        build_example_sentences_tables()

    # Queries ExampleSentence/WordSentence directly (rather than importing
    # jlpt.get_example_sentences) because this file is running as __main__:
    # `import jlpt` would import *this* module a second time under the name
    # "sentences", and SQLAlchemy rejects the resulting duplicate table
    # definitions.
    print("\n--- example sentence lookup ---")
    with JMdictSession() as session:
        for test_word in ("日本語", "食べる", "難しい", "勉強"):
            rows = session.execute(
                select(ExampleSentence.japanese, ExampleSentence.english)
                .join(WordSentence, WordSentence.sentence_id == ExampleSentence.id)
                .where(WordSentence.word == test_word)
                .limit(3)
            ).all()
            for japanese, english in rows:
                print(f"{test_word}: {japanese}  —  {english}")
