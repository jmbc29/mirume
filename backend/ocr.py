"""Screenshot OCR fallback for Mirume.

The macOS Accessibility API (:mod:`accessibility`) cannot see inside a
sandboxed renderer — Chrome, Electron apps, canvas/WebGL content, video — so
for that content ``/hover`` falls back to grabbing a small screenshot around
the cursor and running `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR>`_
over it. PaddleOCR runs as a two-stage pipeline — a detection model that finds
each line of text's exact bounding box, then a recognition model that reads
only the pixels inside each box — which is what this module is built around:
of everything detected in the captured region, only the box closest to the
cursor is returned. This replaced `manga-ocr <https://github.com/kha-white/manga-ocr>`_,
which read the captured region as a single blob of text and would blend
together unrelated nearby UI elements on a dense page (its output looked like
a hallucination but was really just several fragments run together).

Public API:

* :func:`capture_region` – screenshot a rectangle of the screen as a
  :class:`PIL.Image.Image`, via the ``screencapture`` CLI. It grabs the
  *composited* display exactly as the user sees it, so Mirume's own
  transparent full-screen overlay contributes nothing and the web content
  underneath shows through (``CGWindowListCreateImage`` composited the empty
  overlay on top instead, yielding the desktop wallpaper).
* :func:`extract_text_at_position` – OCR a region centred on a screen
  coordinate; returns the text of whichever detected line sits closest to the
  cursor, when it actually contains Japanese, the frontmost app isn't a dev
  tool (see :data:`_OCR_BLOCKED_APPS`), and — when the frontmost app is
  Chrome — the active tab isn't an AI chat site (see :data:`_BLOCKED_DOMAINS`).
* :func:`get_frontmost_app` – name of the active application, used to gate
  OCR off code editors/terminals whose own UI text isn't meant to be read.
* :func:`get_chrome_url` – active tab URL when Chrome is frontmost, used to
  tell an actual webpage apart from an AI chat site running in the browser.

The detection + recognition model weights (~150 MB total) download once on
first use into ``~/.paddlex/official_models`` and are cached there by
``paddlex``. Loading them into memory is slow (several seconds on a cold
cache, longer the very first time while the weights download), so
:func:`extract_text_at_position` loads lazily and caches the model for every
call after. The backend calls it once during startup warmup so the first real
hover is fast.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import threading

import numpy as np

# --------------------------------------------------------------------------- #
# Optional-dependency guard. paddleocr pulls in paddlepaddle, a full ML
# runtime; the module must still import (degrading to "no OCR") when it or
# Pillow is missing so the rest of the backend keeps working.
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
#: Taller than the old manga-ocr region (400x100) since PaddleOCR's detector
#: needs enough vertical room to isolate the line the cursor is on from the
#: lines immediately above/below it, rather than reading everything at once.
_OCR_REGION_WIDTH = 600
_OCR_REGION_HEIGHT = 300

#: Hard ceiling on the ``screencapture`` CLI. It normally returns in well under
#: 200 ms; anything longer means it is wedged (a permission prompt, a stuck
#: WindowServer round-trip inside the uvicorn worker) and we abandon it.
_SCREENCAPTURE_TIMEOUT_S = 2.0

#: How long :func:`extract_text_at_position` waits for the shared inference
#: slot before giving up on this hover. Capture (~200 ms, or the 2 s
#: ``screencapture`` timeout) plus inference (~300 ms) is the normal hold
#: time, so a wait longer than this means calls are stacking up faster than
#: PaddleOCR can drain them and it is better to skip.
_OCR_LOCK_WAIT_S = 2.5

#: Codepoint range that counts as "Japanese" — U+3040..U+9FFF spans hiragana,
#: katakana, the CJK symbols/punctuation block and the CJK unified ideographs
#: (kanji). Used to reject a detected line with none of these — PaddleOCR's
#: recognition model still emits its best guess for a box the detector found,
#: even when that box turns out to be non-Japanese UI text.
_JAPANESE_RE = re.compile(r"[぀-鿿]")

#: Set once we've warned that ``screencapture`` is failing (almost always a
#: missing Screen Recording grant), so the log isn't spammed on every hover.
_capture_warned = False

#: Skip OCR entirely when the captured region has too little tonal contrast to
#: contain rendered text — measured as the spread between the 5th and 95th
#: percentile of greyscale pixel values (0-255). Detection-based OCR won't
#: hallucinate text on a blank region the way manga-ocr did, but running the
#: full detect+recognise pipeline over an empty margin is still wasted work.
_MIN_CONTRAST_RANGE = 70

#: Minimum recognition confidence (0-1) for a detected line to be trusted.
_MIN_OCR_CONFIDENCE = 0.7

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

#: Substrings matched against the active Chrome tab's URL (lowercased) to
#: block OCR on — AI chat sites where Mirume would otherwise treat the
#: assistant's Japanese-language replies as text to classify, plus localhost,
#: which is almost always this project's own dev servers.
_BLOCKED_DOMAINS: frozenset[str] = frozenset(
    {
        "claude.ai",
        "chat.openai.com",
        "chatgpt.com",
        "localhost",
        "127.0.0.1",
    }
)


# --------------------------------------------------------------------------- #
# Lazy model loader
# --------------------------------------------------------------------------- #

_ocr_instance = None  # cached paddleocr.PaddleOCR instance
_ocr_load_failed = False
_model_lock = threading.Lock()  # serialise the one-time load across threads

#: Last OCR result, keyed by the cursor position it was computed for. A hover
#: that lands within :data:`_OCR_CACHE_RADIUS` px of the last call reuses this
#: instead of re-running capture + inference (~2-3s), since a cursor that has
#: only nudged a few pixels is almost always still over the same line of text.
_last_ocr_cache: dict = {"x": -9999.0, "y": -9999.0, "text": None}
_OCR_CACHE_RADIUS = 100

#: Held for the duration of every capture + PaddleOCR inference. Concurrent
#: PaddleOCR calls have not shown the crash-on-overlap behaviour manga-ocr's
#: torch-MPS backend had, but the lock is kept anyway: it's cheap (inference
#: is ~300 ms) and only one OCR pass is ever useful per hover — the cursor has
#: usually moved on by the time a second concurrent call would finish.
_infer_lock = threading.Lock()


def _get_model():
    """Return a cached ``PaddleOCR`` instance, loading it on first call.

    Thread-safe: the startup warmup thread and a racing first request would
    otherwise both build a ``PaddleOCR`` instance (each a multi-second load,
    longer still on the very first run while weights download). The lock lets
    the first caller do the load while the rest wait for it.

    Returns:
        The ``paddleocr.PaddleOCR`` instance, or ``None`` if paddleocr is not
        installed or the model could not be loaded (logged once).
    """
    global _ocr_instance, _ocr_load_failed
    if _ocr_instance is not None:
        return _ocr_instance
    if _ocr_load_failed:
        return None
    with _model_lock:
        if _ocr_instance is not None:
            return _ocr_instance
        if _ocr_load_failed:
            return None
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # pragma: no cover - dependency missing
            print(f"[mirume] paddleocr not available ({exc}); OCR fallback disabled.")
            _ocr_load_failed = True
            return None
        try:
            print(
                "[mirume] loading PaddleOCR model (first use, several seconds; "
                "longer on first run while weights download)...",
                file=sys.stderr,
            )
            _ocr_instance = PaddleOCR(
                lang="japan",
                use_textline_orientation=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        except Exception as exc:  # pragma: no cover - runtime/model failure
            print(f"[mirume] failed to load PaddleOCR model ({exc}); OCR disabled.")
            _ocr_load_failed = True
            return None
    return _ocr_instance


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


def get_chrome_url() -> str:
    """Return the URL of Chrome's active tab, lowercased.

    Used to keep OCR off AI chat sites (:data:`_BLOCKED_DOMAINS`) — the
    frontmost-app check alone sees "Google Chrome" for every website, so it
    can't tell NHK news apart from a chat with Claude. Shells out to Chrome
    via ``osascript``; bounded by a 1 s timeout for the same reason as
    :func:`get_frontmost_app`. Only meaningful (and only ever called) while
    Chrome is frontmost — asking a backgrounded Chrome for its URL still
    works, but there's no reason to pay the round-trip when it isn't.

    Returns:
        The active tab's URL, lowercased, or ``""`` if it could not be
        determined (Chrome isn't running, AppleScript is disabled for it, or
        the call times out).
    """
    script = """
    tell application "Google Chrome"
        return URL of active tab of front window
    end tell
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=1,
        )
        return result.stdout.strip().lower()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _has_text_contrast(image: "Image.Image") -> bool:
    """Return whether ``image`` has enough tonal range to contain rendered text.

    A near-uniform region (blank background, empty margin) is never worth
    running the full detect+recognise pipeline over.
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
    """OCR a 400x200 region centred on ``(x, y)`` and return the line nearest the cursor.

    PaddleOCR's detector finds every line of text in the captured region as a
    separate bounding box; of those, only the one whose box centre is closest
    to the cursor (the exact centre of the capture) is returned. This is what
    keeps a dense page — several unrelated lines of text within 200px of the
    cursor — from blending into one hallucinated string the way manga-ocr's
    single whole-region pass did.

    Args:
        x: Horizontal screen coordinate (top-left origin, points).
        y: Vertical screen coordinate.

    Returns:
        The recognised text of the closest line, stripped, or ``None`` when
        nothing was captured, the model is unavailable or busy, the frontmost
        app is a blocked dev tool (:data:`_OCR_BLOCKED_APPS`), Chrome's active
        tab is a blocked AI chat site (:data:`_BLOCKED_DOMAINS`), the region
        is too low-contrast to hold text, or no detected line clears both
        :data:`_MIN_OCR_CONFIDENCE` and having at least one Japanese
        character.

    Concurrency: a process-wide lock serialises capture + inference — see
    :data:`_infer_lock`. If it can't be acquired quickly this returns ``None``
    (skip OCR for this hover) rather than queue — the cursor has usually moved
    on anyway. The one-time model load happens before the lock, is slow but
    bounded, and runs once (normally on the startup warmup thread).

    Caching: a hover within :data:`_OCR_CACHE_RADIUS` px of the last call
    returns the cached result immediately instead of re-running capture +
    inference — see :data:`_last_ocr_cache`.
    """
    global _last_ocr_cache
    dx = x - _last_ocr_cache["x"]
    dy = y - _last_ocr_cache["y"]
    if (dx * dx + dy * dy) < _OCR_CACHE_RADIUS**2:
        return _last_ocr_cache["text"]

    model = _get_model()
    if model is None:
        return None

    # Gated on the frontmost app, not just cursor position: dev tools and
    # this backend's own terminal are never a valid OCR target (VS Code text
    # the AX tree misses still isn't web content). Checked here rather than
    # before the model load so the startup warmup call still loads the model
    # into memory regardless of what's frontmost at that moment.
    frontmost = get_frontmost_app()
    if frontmost in _OCR_BLOCKED_APPS:
        return None

    # The app-level check above sees "Google Chrome" for every website, so
    # AI chat sites need a second, URL-based gate — otherwise Mirume reads
    # Claude's own Japanese-language replies as text to classify.
    if frontmost == "Google Chrome":
        url = get_chrome_url()
        if any(blocked in url for blocked in _BLOCKED_DOMAINS):
            return None

    if not _infer_lock.acquire(timeout=_OCR_LOCK_WAIT_S):
        return None
    try:
        left = x - _OCR_REGION_WIDTH / 2
        top = y - _OCR_REGION_HEIGHT / 2
        image = capture_region(left, top, _OCR_REGION_WIDTH, _OCR_REGION_HEIGHT)
        if image is None or not _has_text_contrast(image):
            _last_ocr_cache = {"x": x, "y": y, "text": None}
            return None
        try:
            results = model.predict(np.array(image))
        except Exception:
            _last_ocr_cache = {"x": x, "y": y, "text": None}
            return None
    finally:
        _infer_lock.release()

    if not results:
        _last_ocr_cache = {"x": x, "y": y, "text": None}
        return None
    result = results[0]
    texts = result.get("rec_texts") or []
    scores = result.get("rec_scores") or []
    boxes = result.get("rec_boxes")
    if boxes is None or len(texts) == 0:
        _last_ocr_cache = {"x": x, "y": y, "text": None}
        return None

    # The capture region is centred on the cursor, so the cursor sits at the
    # exact centre of the image regardless of x/y — the closest detected line
    # to that point is the one the user is actually pointing at.
    cursor_in_image = (_OCR_REGION_WIDTH / 2, _OCR_REGION_HEIGHT / 2)
    best_text: str | None = None
    best_distance = float("inf")
    for text, score, box in zip(texts, scores, boxes):
        if score < _MIN_OCR_CONFIDENCE:
            continue
        if not _JAPANESE_RE.search(text):
            continue
        box_x1, box_y1, box_x2, box_y2 = (float(v) for v in box[:4])
        box_center = ((box_x1 + box_x2) / 2, (box_y1 + box_y2) / 2)
        distance = (
            (box_center[0] - cursor_in_image[0]) ** 2 + (box_center[1] - cursor_in_image[1]) ** 2
        ) ** 0.5
        if distance < best_distance:
            best_distance = distance
            best_text = text

    if best_text is None:
        _last_ocr_cache = {"x": x, "y": y, "text": None}
        return None
    text = best_text.strip()
    if len(text) < 2:
        _last_ocr_cache = {"x": x, "y": y, "text": None}
        return None
    _last_ocr_cache = {"x": x, "y": y, "text": text}
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
