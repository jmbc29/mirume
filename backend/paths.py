"""Filesystem locations for Mirume's data, models and configuration.

The backend runs in two very different layouts and this module is the single
place that knows which one it is in:

* **Development** — run straight from the source tree
  (``uvicorn main:app ...``). Data lives in the repo's own ``data/`` and
  ``models/`` directories, exactly as before.
* **Packaged** — run as the PyInstaller binary bundled inside
  ``Mirume.app``. The app bundle is read-only and code-signed, so nothing
  can be written next to the executable. The Tauri launcher points the
  backend at a writable per-user directory via the ``MIRUME_DATA_DIR``
  environment variable (``~/Library/Application Support/com.mirume.app``),
  and at the read-only seed files shipped inside the bundle via
  ``MIRUME_BUNDLED_RES``.

Resolution order for the data root:

1. ``$MIRUME_DATA_DIR`` if set (the packaged app always sets it).
2. ``~/Library/Application Support/com.mirume.app`` when running frozen
   (a PyInstaller build) without the variable set — a sensible default so a
   manually-launched frozen binary still works.
3. ``<repo>/data`` and ``<repo>/models`` otherwise (development).

:func:`seed_from_bundle` copies the shipped dictionary and language-id model
into the writable data root on first run, so the packaged app works offline
immediately instead of downloading ~80 MB of sources and building the
database (:func:`jlpt.build_database`, ~1-2 min) on the user's machine.
"""

from __future__ import annotations

import gzip
import os
import shutil
import sys
from pathlib import Path

#: ``com.mirume.app`` — matches ``tauri.conf.json``'s ``identifier`` and the
#: Application Support subdirectory the packaged app writes to.
BUNDLE_IDENTIFIER = "com.mirume.app"

#: Repo root when running from source (``mirume/``), used for the development
#: layout only.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _running_frozen() -> bool:
    """Whether this is a PyInstaller (or similar) frozen build."""
    return bool(getattr(sys, "frozen", False))


def _resolve_data_root() -> Path:
    """Return the writable directory holding ``data/`` and ``models/``."""
    override = os.environ.get("MIRUME_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if _running_frozen():
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / BUNDLE_IDENTIFIER
        )
    return _REPO_ROOT


#: Writable root under which ``data/`` and ``models/`` live.
DATA_ROOT: Path = _resolve_data_root()

#: Generated SQLite databases + downloaded dictionary sources.
DATA_DIR: Path = DATA_ROOT / "data"
#: fastText language-id model cache.
MODELS_DIR: Path = DATA_ROOT / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

#: ``.env`` with the optional DeepL / Anthropic keys. In a packaged app the
#: user drops this into the Application Support directory; in development it
#: stays next to the backend source as before. Both are tried, in this order.
ENV_FILES: tuple[Path, ...] = (
    DATA_ROOT / ".env",
    Path(__file__).resolve().parent / ".env",
)

#: Directory of read-only seed files shipped inside the app bundle
#: (``Mirume.app/Contents/Resources/backend-data``). Set by the Tauri
#: launcher; ``None`` in development.
_bundled_res = os.environ.get("MIRUME_BUNDLED_RES", "").strip()
BUNDLED_RES_DIR: Path | None = Path(_bundled_res).resolve() if _bundled_res else None


def seed_from_bundle() -> None:
    """Populate the writable data root from the bundle's seed files, once.

    Copies the pre-built ``jmdict.db`` (shipped gzip-compressed to roughly a
    third of its 145 MB) and the fastText ``lid.176.ftz`` model into place if
    they are not already there. A no-op when :data:`BUNDLED_RES_DIR` is unset
    (development) or every target already exists.
    """
    if BUNDLED_RES_DIR is None or not BUNDLED_RES_DIR.is_dir():
        return

    jmdict_db = DATA_DIR / "jmdict.db"
    jmdict_gz = BUNDLED_RES_DIR / "jmdict.db.gz"
    if not jmdict_db.exists() and jmdict_gz.exists():
        print(f"[mirume] seeding dictionary database -> {jmdict_db}")
        tmp = jmdict_db.with_suffix(".db.partial")
        with gzip.open(jmdict_gz, "rb") as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        tmp.replace(jmdict_db)

    lid_model = MODELS_DIR / "lid.176.ftz"
    lid_seed = BUNDLED_RES_DIR / "lid.176.ftz"
    if not lid_model.exists() and lid_seed.exists():
        print(f"[mirume] seeding language-id model -> {lid_model}")
        shutil.copy2(lid_seed, lid_model)
