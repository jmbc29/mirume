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
  coordinate; returns the text only when it actually contains Japanese and
  the frontmost app isn't a dev tool (see :data:`_OCR_BLOCKED_APPS`).
* :func:`get_frontmost_app` – name of the active application, used to gate
  OCR off code editors/terminals whose own UI text isn't meant to be read.

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
import threading

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

# NOTE: the CoreGraphics in-process capture APIs (``CGDisplayCreateImage`` /
# ``CGDisplayCreateImageForRect``) are *not* used here. They are deprecated as
# of macOS 14 and on macOS 15+ they block for ~5 s on a TCC round-trip and then
# return ``None`` when called from a non-GUI process like the uvicorn worker,
# even with Screen Recording granted — worse than the ``screencapture`` CLI,
# which is a separate signed Apple binary and keeps working. A ScreenCaptureKit
# path would be the modern replacement but needs an async delegate + run loop.


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Region OCR'd by :func:`extract_text_at_position`, centred on the cursor.
_OCR_REGION_WIDTH = 400
_OCR_REGION_HEIGHT = 100

#: Hard ceiling on the ``screencapture`` CLI. It normally returns in well under
#: 200 ms; anything longer means it is wedged (a permission prompt, a stuck
#: WindowServer round-trip inside the uvicorn worker) and we abandon it.
_SCREENCAPTURE_TIMEOUT_S = 2.0

#: How long :func:`extract_text_at_position` waits for the shared inference
#: slot before giving up on this hover. Capture (~200 ms, or the 2 s
#: ``screencapture`` timeout) plus inference (~0.5 s) is the normal hold time,
#: so a wait longer than this means calls are stacking up faster than manga-ocr
#: can drain them and it is better to skip.
_OCR_LOCK_WAIT_S = 2.5

#: Codepoint range that counts as "Japanese" — U+3040..U+9FFF spans hiragana,
#: katakana, the CJK symbols/punctuation block and the CJK unified ideographs
#: (kanji). Used both to reject results with none of these (manga-ocr
#: hallucinates short Latin strings on empty or dark backgrounds) and to check
#: what fraction of a result is actually Japanese (see _MIN_JAPANESE_DENSITY).
_JAPANESE_RE = re.compile(r"[぀-鿿]")

#: Set once we've warned that ``screencapture`` is failing (almost always a
#: missing Screen Recording grant), so the log isn't spammed on every hover.
_capture_warned = False

#: manga-ocr always returns *some* string, and on a near-uniform region (a
#: blank wall of colour, an empty margin) that string is a plausible-looking
#: Japanese hallucination. Skip OCR entirely when the captured region has too
#: little tonal contrast to contain rendered text — measured as the spread
#: between the 5th and 95th percentile of greyscale pixel values (0-255).
_MIN_CONTRAST_RANGE = 70

#: Minimum fraction of non-whitespace characters in an OCR result that must be
#: Japanese for the result to be trusted. manga-ocr sometimes locks onto a
#: sliver of nearby English/UI text at the edge of the capture region; a
#: result that's mostly non-Japanese is more likely noise than a real word.
_MIN_JAPANESE_DENSITY = 0.4

#: App process names (as reported by System Events, i.e. what
#: :func:`get_frontmost_app` returns) where OCR should never run: dev tools
#: and terminals the user is likely to have Mirume's own backend open in, plus
#: Claude itself. A blocklist rather than an allowlist of browsers, so OCR
#: still works in any browser / PDF viewer / other app we haven't thought to
#: list — only the apps below are actively excluded.
_OCR_BLOCKED_APPS: frozenset[str] = frozenset(
    {
        "Code",  # VS Code
        "Terminal",
        "iTerm2",
        "Xcode",
        "cursor",
        "Cursor",
        "claude",
        "Claude",
    }
)


# --------------------------------------------------------------------------- #
# Lazy model loader
# --------------------------------------------------------------------------- #

_mocr = None  # cached manga_ocr.MangaOcr instance
_mocr_failed = False
_mocr_lock = threading.Lock()  # serialise the one-time load across threads

#: Held for the duration of every capture + manga-ocr inference. torch-MPS
#: inference is not reentrant — concurrent calls abort the worker process — so
#: OCR is strictly serialised process-wide. See :func:`extract_text_at_position`.
_infer_lock = threading.Lock()


def _get_model():
    """Return a cached ``MangaOcr`` instance, loading it on first call.

    Thread-safe: the startup warmup thread and a racing first request would
    otherwise both build a ``MangaOcr`` (each a ~5 s, ~440 MB load). The lock
    lets the first caller do the load while the rest wait for it.

    Returns:
        The ``manga_ocr.MangaOcr`` callable, or ``None`` if manga-ocr is not
        installed or the model could not be loaded (logged once).
    """
    global _mocr, _mocr_failed
    if _mocr is not None:
        return _mocr
    if _mocr_failed:
        return None
    with _mocr_lock:
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
            print(
                "[mirume] loading manga-ocr model (first use, ~2-3s)...",
                file=sys.stderr,
            )
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

    The primary capture path. ``screencapture`` grabs the composited
    framebuffer — what is actually on screen — so a transparent window (Mirume's
    overlay) contributes nothing and the app underneath is what gets grabbed.
    Bounded by :data:`_SCREENCAPTURE_TIMEOUT_S` so a wedged subprocess is
    abandoned rather than hanging the request.
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
            timeout=_SCREENCAPTURE_TIMEOUT_S,
            # Inherit the launching shell's full environment; a stripped-down
            # PATH/CFFIXED_USER_HOME has been seen to make the CLI stall.
            env=os.environ.copy(),
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
    except subprocess.TimeoutExpired:
        if not _capture_warned:
            _capture_warned = True
            print(
                f"[mirume] screencapture did not return within "
                f"{_SCREENCAPTURE_TIMEOUT_S:g}s — abandoning it; this hover yields "
                "no OCR text."
            )
        return None
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
    Accessibility API and the cursor poller). Uses the ``screencapture`` CLI
    (:func:`_screencapture_region`), falling back to :func:`PIL.ImageGrab.grab`.
    Both grab the composited display, so Mirume's transparent overlay is
    invisible to them.

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
# Frontmost-app gating
# --------------------------------------------------------------------------- #


def get_frontmost_app() -> str:
    """Return the name of the frontmost (active) application.

    Used to keep OCR off the user's own dev tools and this backend's
    terminal — see :data:`_OCR_BLOCKED_APPS` — since the AX API alone can't
    tell OCR-worthy sandboxed web content (Chrome) apart from a code editor
    that also happens to render its own text in a way the AX tree misses.
    Shells out to System Events via ``osascript``; bounded by a 1 s timeout so
    a slow/wedged AppleScript round-trip never turns into a hung hover.

    Returns:
        The frontmost app's process name (e.g. ``"Google Chrome"``), or
        ``""`` if it could not be determined.
    """
    script = """
    tell application "System Events"
        set frontApp to name of first application process whose frontmost is true
        return frontApp
    end tell
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=1,
        )
        return result.stdout.strip()
    except Exception:
        return ""


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
        the model is unavailable or busy, the frontmost app is a blocked dev
        tool (:data:`_OCR_BLOCKED_APPS`), the region is too low-contrast to
        hold text, the result is shorter than 2 characters, or fewer than
        :data:`_MIN_JAPANESE_DENSITY` of the result's characters are Japanese
        (hiragana / katakana / kanji). manga-ocr always emits *something* — a
        low-contrast capture or a result that's mostly non-Japanese noise is
        treated as "no text here".

    Concurrency: manga-ocr / torch-MPS inference is **not** reentrant — two
    overlapping calls crash the whole worker process (silently, with no
    traceback), which is what leaves ``/hover`` hanging for every later caller.
    ``FastAPI`` dispatches ``/hover`` across a thread pool and the cursor poller
    fires every 200 ms, so overlap is the norm. This function therefore takes a
    process-wide lock for the capture + inference; if it can't get the lock
    quickly it returns ``None`` (skip OCR for this hover) rather than queue —
    the cursor has usually moved on anyway. The one-time model load happens
    before the lock, is slow (~5 s) but bounded, and runs once (normally on the
    startup warmup thread).
    """
    model = _get_model()
    if model is None:
        return None

    # Gated on the frontmost app, not just cursor position: dev tools and
    # this backend's own terminal are never a valid OCR target (VS Code text
    # the AX tree misses still isn't web content). Checked here rather than
    # before the model load so the startup warmup call still loads the model
    # into memory regardless of what's frontmost at that moment.
    if get_frontmost_app() in _OCR_BLOCKED_APPS:
        return None

    if not _infer_lock.acquire(timeout=_OCR_LOCK_WAIT_S):
        return None
    try:
        left = x - _OCR_REGION_WIDTH / 2
        top = y - _OCR_REGION_HEIGHT / 2
        image = capture_region(left, top, _OCR_REGION_WIDTH, _OCR_REGION_HEIGHT)
        if image is None or not _has_text_contrast(image):
            return None
        try:
            text = model(image)
        except Exception:
            return None
    finally:
        _infer_lock.release()

    text = (text or "").strip()
    if len(text) < 2:
        return None
    japanese_chars = len(_JAPANESE_RE.findall(text))
    if japanese_chars / len(text) < _MIN_JAPANESE_DENSITY:
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
