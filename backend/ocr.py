"""Screenshot OCR fallback for Mirume.

The macOS Accessibility API (:mod:`accessibility`) cannot see inside a
sandboxed renderer — Chrome, Electron apps, canvas/WebGL content, video — so
for that content ``/hover`` falls back to grabbing a small screenshot around
the cursor and running `manga-ocr <https://github.com/kha-white/manga-ocr>`_
over it. manga-ocr is a Japanese-specific recognition model (trained on manga
pages) and does well on the short, mixed-font runs typical of web text.

Public API:

* :func:`capture_region` – screenshot a rectangle of the screen as a
  :class:`PIL.Image.Image`.
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
import sys

# manga-ocr / huggingface_hub default to the Xet transfer protocol, which has
# been seen to stall indefinitely on the first weights download behind some
# networks. Fall back to plain HTTPS range downloads unless the operator opts
# back in. Must be set before `manga_ocr`/`huggingface_hub` are imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# --------------------------------------------------------------------------- #
# Optional-dependency guards. manga-ocr pulls in torch/transformers and Quartz
# is macOS-only; the module must still import (degrading to "no OCR") when
# either is missing so the rest of the backend keeps working.
# --------------------------------------------------------------------------- #

_QUARTZ_AVAILABLE = True
try:  # pragma: no cover - platform dependent
    from Quartz import (
        CGMainDisplayID,
        CGRectMake,
        CGWindowListCreateImage,
        kCGNullWindowID,
        kCGWindowImageDefault,
        kCGWindowListOptionOnScreenOnly,
    )
    from Quartz.CoreGraphics import (
        CGDataProviderCopyData,
        CGImageGetDataProvider,
        CGImageGetHeight,
        CGImageGetWidth,
        CGImageGetBytesPerRow,
    )
except Exception as exc:  # pragma: no cover - platform dependent
    _QUARTZ_AVAILABLE = False
    _QUARTZ_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    from PIL import Image, ImageGrab
except Exception as exc:  # pragma: no cover - dependency missing
    Image = None  # type: ignore[assignment]
    ImageGrab = None  # type: ignore[assignment]
    _PIL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Default capture size for :func:`capture_region` (width, height) in points.
_DEFAULT_REGION = (300, 80)

#: Region OCR'd by :func:`extract_text_at_position`, centred on the cursor.
_OCR_REGION_WIDTH = 400
_OCR_REGION_HEIGHT = 100

#: Codepoint range that counts as "Japanese" — U+3040..U+9FFF spans hiragana,
#: katakana, the CJK symbols/punctuation block and the CJK unified ideographs
#: (kanji). If an OCR result contains none of these it is noise (manga-ocr
#: hallucinates short Latin strings on empty or dark backgrounds) and we drop it.
_JAPANESE_RE = re.compile(r"[぀-鿿]")

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


def _cgimage_to_pil(cg_image) -> "Image.Image | None":
    """Convert a ``CGImageRef`` to a :class:`PIL.Image.Image` (RGB)."""
    if Image is None:
        return None
    width = CGImageGetWidth(cg_image)
    height = CGImageGetHeight(cg_image)
    if width == 0 or height == 0:
        return None
    bytes_per_row = CGImageGetBytesPerRow(cg_image)
    provider = CGImageGetDataProvider(cg_image)
    data = CGDataProviderCopyData(provider)
    buffer = bytes(data)
    # CGWindowListCreateImage hands back premultiplied BGRA, one row every
    # ``bytes_per_row`` bytes (which may be padded past ``width * 4``).
    image = Image.frombuffer(
        "RGBA", (width, height), buffer, "raw", "BGRA", bytes_per_row, 1
    )
    return image.convert("RGB")


def capture_region(x: float, y: float, width: int = 300, height: int = 80) -> "Image.Image | None":
    """Screenshot a ``width`` x ``height`` rectangle whose top-left is ``(x, y)``.

    Coordinates are top-left-origin screen points (the same convention as the
    Accessibility API and the cursor poller). Tries the Quartz
    ``CGWindowListCreateImage`` path first — it captures on-screen window
    content without stealing focus — and falls back to :func:`PIL.ImageGrab.grab`.

    Args:
        x: Left edge of the region, in screen points.
        y: Top edge of the region, in screen points.
        width: Region width in points.
        height: Region height in points.

    Returns:
        The captured image (RGB), or ``None`` if screen capture is unavailable
        or permission (Screen Recording) has not been granted.
    """
    left, top = int(x), int(y)
    right, bottom = left + int(width), top + int(height)

    if _QUARTZ_AVAILABLE:
        try:
            rect = CGRectMake(left, top, int(width), int(height))
            cg_image = CGWindowListCreateImage(
                rect,
                kCGWindowListOptionOnScreenOnly,
                kCGNullWindowID,
                kCGWindowImageDefault,
            )
            if cg_image is not None:
                pil = _cgimage_to_pil(cg_image)
                if pil is not None:
                    # Retina displays hand back a 2x pixel buffer for the same
                    # point rect; normalise back to the requested point size so
                    # OCR sees a consistent scale.
                    if pil.width > int(width) or pil.height > int(height):
                        pil = pil.resize((int(width), int(height)), Image.LANCZOS)
                    return pil
        except Exception:
            pass  # fall through to ImageGrab

    if ImageGrab is not None:
        try:
            return ImageGrab.grab(bbox=(left, top, right, bottom)).convert("RGB")
        except Exception:
            return None
    return None


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
