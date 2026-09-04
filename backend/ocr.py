"""Screenshot OCR fallback for Mirume.

The macOS Accessibility API (:mod:`accessibility`) cannot see inside a
sandboxed renderer — Chrome, Electron apps, canvas/WebGL content, video — so
for that content ``/hover`` falls back to grabbing a small screenshot around
the cursor and running `manga-ocr <https://github.com/kha-white/manga-ocr>`_
over it. manga-ocr is a Japanese-specific recognition model (trained on manga
pages) and does well on the short, mixed-font runs typical of web text.

Public API:

* :func:`capture_region` – screenshot a rectangle of the screen as a
  :class:`PIL.Image.Image`, via the ``screencapture`` CLI. It grabs the
  *composited* display exactly as the user sees it, so Mirume's own
  transparent full-screen overlay contributes nothing and the web content
  underneath shows through (``CGWindowListCreateImage`` composited the empty
  overlay on top instead, yielding the desktop wallpaper).
* :func:`extract_text_at_position` – OCR a region centred on a screen
  coordinate; returns the text only when it actually contains Japanese.

The model weights (~400 MB) download once on first use and are then cached by
``huggingface_hub``. Loading them into memory takes 2-3 s, so the first call is
slow; :func:`extract_text_at_position` loads lazily and caches the model for
every call after. The backend calls it once during startup warmup so the first
real hover is fast.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

# manga-ocr / huggingface_hub default to the Xet transfer protocol, which has
# been seen to stall indefinitely on the first weights download behind some
# networks. Fall back to plain HTTPS range downloads unless the operator opts
# back in. Must be set before `manga_ocr`/`huggingface_hub` are imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# --------------------------------------------------------------------------- #
# Optional-dependency guard. manga-ocr pulls in torch/transformers; the module
# must still import (degrading to "no OCR") when it or Pillow is missing so the
# rest of the backend keeps working.
# --------------------------------------------------------------------------- #

try:
    from PIL import Image, ImageGrab
except Exception as exc:  # pragma: no cover - dependency missing
    Image = None  # type: ignore[assignment]
    ImageGrab = None  # type: ignore[assignment]
    _PIL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Region OCR'd by :func:`extract_text_at_position`, centred on the cursor.
_OCR_REGION_WIDTH = 400
_OCR_REGION_HEIGHT = 100

#: Codepoint range that counts as "Japanese" — U+3040..U+9FFF spans hiragana,
#: katakana, the CJK symbols/punctuation block and the CJK unified ideographs
#: (kanji). If an OCR result contains none of these it is noise (manga-ocr
#: hallucinates short Latin strings on empty or dark backgrounds) and we drop it.
_JAPANESE_RE = re.compile(r"[぀-鿿]")

#: Set once we've warned that ``screencapture`` is failing (almost always a
#: missing Screen Recording grant), so the log isn't spammed on every hover.
_capture_warned = False

#: manga-ocr always returns *some* string, and on a near-uniform region (a
#: blank wall of colour, an empty margin) that string is a plausible-looking
#: Japanese hallucination. Skip OCR entirely when the captured region has too
#: little tonal contrast to contain rendered text — measured as the spread
#: between the 5th and 95th percentile of greyscale pixel values (0-255).
_MIN_CONTRAST_RANGE = 40


# --------------------------------------------------------------------------- #
# Lazy model loader
# --------------------------------------------------------------------------- #

_mocr = None  # cached manga_ocr.MangaOcr instance
_mocr_failed = False


def _get_model():
    """Return a cached ``MangaOcr`` instance, loading it on first call.

    Returns:
        The ``manga_ocr.MangaOcr`` callable, or ``None`` if manga-ocr is not
        installed or the model could not be loaded (logged once).
    """
    global _mocr, _mocr_failed
    if _mocr is not None:
        return _mocr
    if _mocr_failed:
        return None
    try:
        from manga_ocr import MangaOcr
    except Exception as exc:  # pragma: no cover - dependency missing
        print(f"[mirume] manga-ocr not available ({exc}); OCR fallback disabled.")
        _mocr_failed = True
        return None
    try:
        print("[mirume] loading manga-ocr model (first use, ~2-3s)...", file=sys.stderr)
        _mocr = MangaOcr()
    except Exception as exc:  # pragma: no cover - runtime/model failure
        print(f"[mirume] failed to load manga-ocr model ({exc}); OCR disabled.")
        _mocr_failed = True
        return None
    return _mocr


# --------------------------------------------------------------------------- #
# Screenshot capture
# --------------------------------------------------------------------------- #


def _screencapture_region(left: int, top: int, width: int, height: int) -> "Image.Image | None":
    """Run ``screencapture -x -R`` for one rectangle and load the PNG it writes.

    ``screencapture`` captures the composited framebuffer — what is actually on
    screen — so a transparent window (Mirume's overlay) contributes nothing and
    the app underneath is what gets grabbed.
    """
    global _capture_warned
    if Image is None:
        return None
    fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="mirume-ocr-")
    os.close(fd)
    try:
        proc = subprocess.run(
            ["screencapture", "-x", "-R", f"{left},{top},{width},{height}", tmp_path],
            capture_output=True,
            timeout=5,
        )
        if proc.returncode != 0:
            if not _capture_warned:
                _capture_warned = True
                err = proc.stderr.decode("utf-8", "replace").strip()
                print(
                    f"[mirume] screencapture failed ({err or 'unknown error'}). "
                    "OCR fallback needs Screen Recording permission — grant it in "
                    "System Settings > Privacy & Security > Screen Recording for "
                    "the app that launches the backend (Terminal/iTerm), then "
                    "fully quit and reopen it."
                )
            return None
        with Image.open(tmp_path) as image:
            image.load()
            return image.copy()
    except (subprocess.SubprocessError, OSError, ValueError):
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def capture_region(x: float, y: float, width: int = 300, height: int = 80) -> "Image.Image | None":
    """Screenshot a ``width`` x ``height`` rectangle whose top-left is ``(x, y)``.

    Coordinates are top-left-origin screen points (the same convention as the
    Accessibility API and the cursor poller). Uses the ``screencapture`` CLI —
    which grabs the composited display, seeing past Mirume's transparent
    overlay — and falls back to :func:`PIL.ImageGrab.grab`.

    Args:
        x: Left edge of the region, in screen points.
        y: Top edge of the region, in screen points.
        width: Region width in points.
        height: Region height in points.

    Returns:
        The captured image (RGB), or ``None`` if screen capture failed or
        Screen Recording permission has not been granted.
    """
    left, top = int(round(x)), int(round(y))
    w, h = max(1, int(round(width))), max(1, int(round(height)))

    image = _screencapture_region(left, top, w, h)
    if image is None and ImageGrab is not None:
        try:
            image = ImageGrab.grab(bbox=(left, top, left + w, top + h))
        except Exception:
            image = None
    if image is None:
        return None

    if image.mode != "RGB":
        image = image.convert("RGB")
    # Retina displays hand back a 2x pixel buffer for a point-sized region;
    # normalise to the requested point size so OCR sees a consistent scale.
    if image.width > w or image.height > h:
        image = image.resize((w, h), Image.LANCZOS)
    return image


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _has_text_contrast(image: "Image.Image") -> bool:
    """Return whether ``image`` has enough tonal range to contain rendered text.

    A near-uniform region (blank background, empty margin) makes manga-ocr
    hallucinate; we'd rather return nothing there.
    """
    histogram = image.convert("L").histogram()
    total = sum(histogram)
    if total == 0:
        return False
    lo_cut, hi_cut = total * 0.05, total * 0.95
    running = 0
    lo = hi = 0
    for value, count in enumerate(histogram):
        running += count
        if running <= lo_cut:
            lo = value
        if running <= hi_cut:
            hi = value
    return (hi - lo) >= _MIN_CONTRAST_RANGE


def extract_text_at_position(x: float, y: float) -> str | None:
    """OCR a 400x100 region centred on ``(x, y)`` and return any Japanese text.

    Args:
        x: Horizontal screen coordinate (top-left origin, points).
        y: Vertical screen coordinate.

    Returns:
        The recognised text, stripped, or ``None`` when nothing was captured,
        the model is unavailable, the region is too low-contrast to hold text,
        or the result contains no Japanese characters (hiragana / katakana /
        kanji). manga-ocr always emits *something* — a Japanese-free or
        low-contrast result is treated as "no text here".
    """
    model = _get_model()
    if model is None:
        return None

    left = x - _OCR_REGION_WIDTH / 2
    top = y - _OCR_REGION_HEIGHT / 2
    image = capture_region(left, top, _OCR_REGION_WIDTH, _OCR_REGION_HEIGHT)
    if image is None or not _has_text_contrast(image):
        return None

    try:
        text = model(image)
    except Exception:
        return None

    text = (text or "").strip()
    if not text or not _JAPANESE_RE.search(text):
        return None
    return text


# --------------------------------------------------------------------------- #
# Manual verification
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR the screen around a point.")
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument(
        "--save", metavar="PATH", help="also write the captured region to PATH"
    )
    args = parser.parse_args()

    if args.save:
        region = capture_region(
            args.x - _OCR_REGION_WIDTH / 2,
            args.y - _OCR_REGION_HEIGHT / 2,
            _OCR_REGION_WIDTH,
            _OCR_REGION_HEIGHT,
        )
        if region is not None:
            region.save(args.save)
            print(f"saved capture to {args.save}")
        else:
            print("capture failed (Screen Recording permission?)")

    result = extract_text_at_position(args.x, args.y)
    print("ocr result:", repr(result))
