"""Language identification for Mirume, backed by fastText's lid.176 model.

Mirume's ``/hover`` endpoint has to route text down one of two very different
pipelines depending on whether it is Japanese (classify it) or English
(translate it). :func:`detect_language` answers that question using
fastText's compressed 176-language identification model, which is downloaded
into ``models/`` on first use.
"""

from __future__ import annotations

import fasttext
import requests

from paths import MODELS_DIR

#: Compressed (~917 KB) 176-language fastText identification model.
LID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

#: Local cache location — resolved by :mod:`paths` (``mirume/models/`` in
#: development, a writable per-user directory in the packaged app, where it is
#: also seeded from the bundle so no download is needed).
LID_MODEL_PATH = MODELS_DIR / "lid.176.ftz"

_model: fasttext.FastText._FastText | None = None


def _download_model() -> None:
    """Download the fastText language-id model to :data:`LID_MODEL_PATH`.

    No-ops if the file is already present.
    """
    if LID_MODEL_PATH.exists():
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[mirume] downloading language-id model to {LID_MODEL_PATH} …")
    response = requests.get(LID_MODEL_URL, timeout=180)
    response.raise_for_status()
    LID_MODEL_PATH.write_bytes(response.content)


def _get_model() -> fasttext.FastText._FastText:
    """Return the module-level fastText model, loading (and downloading) it lazily.

    Returns:
        The loaded fastText identification model, cached after first call.
    """
    global _model
    if _model is None:
        _download_model()
        # fastText's C++ loader prints a harmless deprecation warning to
        # stderr on load; suppressing it is not worth the extra dependency.
        _model = fasttext.load_model(str(LID_MODEL_PATH))
    return _model


def detect_language(text: str) -> str:
    """Detect the dominant language of ``text``.

    Args:
        text: Any non-empty string. Newlines are stripped since fastText's
            predict only accepts single-line input.

    Returns:
        A two-letter ISO 639-1 language code (e.g. ``"ja"``, ``"en"``), or
        ``"unknown"`` if ``text`` is empty/whitespace-only.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return "unknown"

    model = _get_model()
    labels, _confidence = model.predict(cleaned, k=1)
    # Labels come back as "__label__<code>".
    return labels[0].removeprefix("__label__")


if __name__ == "__main__":
    for sample in (
        "日本語の勉強は難しいですが、面白いです",
        "Studying Japanese is hard, but it's fun.",
        "これは日本語で、This part is English.",
    ):
        print(f"{detect_language(sample):6}  {sample}")
