"""Screenshot OCR fallback for Mirume.

The macOS Accessibility API (:mod:`accessibility`) cannot see inside a
sandboxed renderer — Chrome, Electron apps, canvas/WebGL content, video — so
for that content ``/hover`` falls back to grabbing a small screenshot around
the cursor and running `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR>`_
over it. PaddleOCR runs as a two-stage pipeline — a detection model that finds
each line of text's exact bounding box, then a recognition model that reads
only the pixels inside each box — which is what this module is built around:
of everything detected in the captured region, only the single box closest to
the cursor is returned (never more than one — see :data:`_OCR_MAX_BOX_DISTANCE_PX`
for the cutoff that keeps a cursor sitting between two blocks from matching
whichever is barely nearer). This replaced `manga-ocr <https://github.com/kha-white/manga-ocr>`_,
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
* :func:`start_ocr_worker` – start the persistent background thread that owns
  the model and every inference call (see :func:`_ocr_worker`); called from
  ``main.py``'s startup warmup and, idempotently, from every
  :func:`extract_text_at_position` call in case it hasn't been already.
* :func:`get_frontmost_app` – name of the active application, used to gate
  OCR off code editors/terminals whose own UI text isn't meant to be read.
* :func:`get_chrome_url` – active tab URL when Chrome is frontmost, used to
  tell an actual webpage apart from an AI chat site running in the browser.

The detection + recognition model weights (~150 MB total) download once on
first use into ``~/.paddlex/official_models`` and are cached there by
``paddlex``. Loading them into memory is slow (several seconds on a cold
cache, longer the very first time while the weights download), so the model
is loaded once by :func:`_ocr_worker` when that thread starts, not per
request. The backend starts the worker (and warms it up) during startup so
the first real hover is fast.
"""

from __future__ import annotations

import os
import queue
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

#: How long :func:`extract_text_at_position` waits for the background OCR
#: worker (see :func:`_ocr_worker`) to answer before giving up on this hover.
#: Capture (~200 ms, or the 2 s ``screencapture`` timeout) plus inference
#: (~300 ms), possibly doubled by the one retry pass, is the normal turnaround
#: time; a wait longer than that means the worker is still busy with a
#: previous (now stale) request and it's better to skip.
_OCR_WORKER_TIMEOUT_S = 2.5

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
#: Raised from 0.5 back to 0.7 — a lower bar let more low-quality detections
#: through, including boxes where the detector had merged two adjacent
#: pieces of text (e.g. a word and its furigana reading) into one box that
#: the recognizer then read as a single garbled string.
_MIN_OCR_CONFIDENCE = 0.7

#: Maximum distance (px, in the *captured image's* coordinate space — the
#: cursor always sits at the image centre, see :func:`_run_ocr_pass`) a
#: detected box's centre may be from the cursor to still count as "the line
#: under the cursor". Without this, the closest box was returned no matter
#: how far away it actually was, so a cursor sitting in the whitespace
#: between two unrelated text blocks would still match whichever was
#: (barely) nearer — which looked like text from the wrong block, or two
#: blocks' text blended together when the cursor moved between them.
_OCR_MAX_BOX_DISTANCE_PX = 80

#: Vertical shift (screen points) applied to the capture region on a retry
#: pass — see :func:`extract_text_at_position`. Catches text that sat right
#: at the edge of (and so was clipped or missed by) the first pass's region.
_OCR_RETRY_Y_SHIFT = 30

#: Matches a short (<=6 char) unit immediately repeated 3+ times in a row —
#: a known PaddleOCR recognition failure on an ambiguous/small crop, where
#: the decoder gets stuck reproducing the same span (e.g. a box that should
#: read "鹿児島" comes back "鹿児島鹿児島鹿児島鹿児島"). Requires *3 or more*
#: repeats, not 2, so legitimate Japanese reduplication is never touched —
#: words like 色々, 我々, 時々 and onomatopoeia like わくわく double, they
#: don't triple.
_REPEATED_UNIT_RE = re.compile(r"(.{1,6}?)\1{2,}")

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
#: assistant's Japanese-language replies as text to classify; GitHub, where
#: this project's own README (which has Japanese text in it) would otherwise
#: get read while browsing the repo; and localhost, almost always this
#: project's own dev servers.
_BLOCKED_DOMAINS: frozenset[str] = frozenset(
    {
        "claude.ai",
        "claude.com",
        "chat.openai.com",
        "chatgpt.com",
        "github.com",
        "github.io",
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

#: OCR results, keyed by a coarse (:data:`_OCR_CACHE_GRID_PX`-rounded) cursor
#: position. Several buckets are kept (unlike a single "last position" slot)
#: so a cursor moving between a few nearby lines — e.g. reading down a
#: paragraph — hits cache on all of them, not just the most recent. Only ever
#: touched from :func:`_ocr_worker`, so it needs no lock.
_ocr_result_cache: dict[tuple[int, int], str | None] = {}
_OCR_CACHE_GRID_PX = 50
_OCR_CACHE_MAX_ENTRIES = 50

#: Single-slot request queue feeding :func:`_ocr_worker`. ``maxsize=1`` plus
#: the drop-then-put in :func:`extract_text_at_position` means a burst of
#: hovers never queues up work for stale positions — only the newest request
#: is ever waiting, since the cursor has usually moved on by the time an
#: older one would be processed anyway.
_ocr_request_queue: "queue.Queue[tuple[float, float, threading.Event]]" = queue.Queue(maxsize=1)
_ocr_worker_started = False
_ocr_worker_start_lock = threading.Lock()


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
        url = result.stdout.strip().lower()
        if not url:
            print(
                f"[mirume] get_chrome_url: empty result "
                f"(stderr: {result.stderr.strip()!r})"
            )
        return url
    except Exception as exc:
        print(f"[mirume] get_chrome_url failed: {exc}")
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


def _collapse_repeated_text(text: str) -> str:
    """Collapse a run of the same short unit repeated 3+ times to one copy.

    See :data:`_REPEATED_UNIT_RE`. PaddleOCR's recognizer occasionally gets
    stuck on an ambiguous or small crop and reproduces the same short span
    several times in a row instead of emitting an end token — this is a
    single detected box misbehaving, not multiple boxes being combined
    (:func:`_run_ocr_pass` only ever returns one box's text to begin with).
    Applied to the winning box's text before it's returned, so a garbled
    "鹿児島鹿児島鹿児島鹿児島" becomes "鹿児島" instead of being shown as-is.
    """
    return _REPEATED_UNIT_RE.sub(r"\1", text)


def _run_ocr_pass(model, x: float, y: float) -> str | None:
    """Capture + recognise a single :data:`_OCR_REGION_WIDTH` x
    :data:`_OCR_REGION_HEIGHT` region centred on ``(x, y)``.

    Of everything PaddleOCR's detector finds in that region, only the
    recognised text of the box whose centre sits closest to ``(x, y)`` — the
    exact centre of the capture — is returned. This is what keeps a dense
    page — several unrelated lines of text within the capture region — from
    blending into one hallucinated string the way manga-ocr's single
    whole-region pass did.

    Only ever called from :func:`_ocr_worker`, the single thread that ever
    touches the model — no lock needed, unlike the old direct-call design.

    Returns:
        The recognised text of the closest line, stripped, or ``None`` when
        nothing was captured, the region is too low-contrast to hold text, no
        detected line clears both :data:`_MIN_OCR_CONFIDENCE` and having at
        least one Japanese character, or the closest qualifying line is
        farther than :data:`_OCR_MAX_BOX_DISTANCE_PX` from the cursor (it's
        whitespace between blocks, not a line the cursor is actually on).
    """
    left = x - _OCR_REGION_WIDTH / 2
    top = y - _OCR_REGION_HEIGHT / 2
    image = capture_region(left, top, _OCR_REGION_WIDTH, _OCR_REGION_HEIGHT)
    if image is None or not _has_text_contrast(image):
        return None
    try:
        results = model.predict(np.array(image))
    except Exception:
        return None

    if not results:
        return None
    result = results[0]
    texts = result.get("rec_texts") or []
    scores = result.get("rec_scores") or []
    boxes = result.get("rec_boxes")
    if boxes is None or len(texts) == 0:
        return None

    # The capture region is centred on (x, y), so that point sits at the
    # exact centre of the image regardless of its value — the closest
    # detected line to that point is the one the user is actually pointing at.
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
        return None
    if best_distance > _OCR_MAX_BOX_DISTANCE_PX:
        return None
    text = _collapse_repeated_text(best_text.strip())
    if len(text) < 2:
        return None
    return text


def _ocr_cache_key(x: float, y: float) -> tuple[int, int]:
    """Quantise ``(x, y)`` onto a :data:`_OCR_CACHE_GRID_PX` grid.

    Used as the :data:`_ocr_result_cache` key so nearby-but-not-identical
    positions (the cursor rarely settles on the exact same point twice) still
    land in the same bucket.
    """
    return (
        round(x / _OCR_CACHE_GRID_PX) * _OCR_CACHE_GRID_PX,
        round(y / _OCR_CACHE_GRID_PX) * _OCR_CACHE_GRID_PX,
    )


def _warm_up_inference(model) -> None:
    """Run one throwaway inference pass before the worker starts serving requests.

    Measured directly (2026-09): a fresh ``model.predict()`` call on a
    600x300 image (the real capture size) costs ~0.5s whether or not an
    earlier call already ran one — there's no separate "first-call is much
    slower" tax to amortize in this setup, so this isn't what was behind a
    multi-second stall (see :func:`extract_text_at_position`'s
    ``_ocr_instance`` check for the actual fix for that). Kept anyway as a
    cheap, harmless sanity check that the loaded model can really run
    inference before the worker advertises itself as ready — and the old
    version of this warmup call (through :func:`extract_text_at_position`,
    with frontmost-app gating still active) would return at the gate before
    ever reaching ``model.predict()`` at all, since the backend's own
    launching terminal is almost always frontmost at startup. This bypasses
    all gating by construction: called directly on a blank image, never
    through :func:`extract_text_at_position`.
    """
    try:
        blank = np.zeros((100, 300, 3), dtype=np.uint8)
        model.predict(blank)
    except Exception as exc:  # pragma: no cover - warmup is best-effort
        print(f"[mirume] OCR inference warmup failed ({exc}); first real hover may be slow.")


def _ocr_worker() -> None:
    """Background thread owning the PaddleOCR model and every inference call.

    Loads the model once and runs one warmup inference (blocking this thread
    only — real requests just wait on it, see :func:`extract_text_at_position`),
    then loops pulling ``(x, y, result_event)`` off :data:`_ocr_request_queue`
    forever. For each: checks :data:`_ocr_result_cache` first; otherwise runs
    :func:`_run_ocr_pass` at ``(x, y)`` and, if that misses, once more with
    the region shifted up by :data:`_OCR_RETRY_Y_SHIFT` px (catches a line
    clipped at the first pass's edge) — then caches and stores the result on
    ``result_event`` before setting it, waking whichever request thread is
    waiting.

    Being the sole caller of :func:`_run_ocr_pass`, this thread is also what
    makes that function's capture+inference lock-free and safe: only one
    inference ever runs at a time by construction, not by a shared lock.

    Runs for the lifetime of the process; started once by
    :func:`start_ocr_worker`.
    """
    model = _get_model()
    if model is not None:
        _warm_up_inference(model)
    while True:
        try:
            x, y, result_event = _ocr_request_queue.get(timeout=1)
        except queue.Empty:
            continue

        cache_key = _ocr_cache_key(x, y)
        if cache_key in _ocr_result_cache:
            result_event.result = _ocr_result_cache[cache_key]  # type: ignore[attr-defined]
            result_event.set()
            continue

        result: str | None = None
        if model is not None:
            result = _run_ocr_pass(model, x, y)
            if result is None:
                result = _run_ocr_pass(model, x, y - _OCR_RETRY_Y_SHIFT)

        _ocr_result_cache[cache_key] = result
        if len(_ocr_result_cache) > _OCR_CACHE_MAX_ENTRIES:
            # Plain dict preserves insertion order in Python 3.7+, so the
            # first key really is the oldest entry — good enough for a soft
            # perf cache; not a strict LRU (a re-hit doesn't move it to the
            # back), which is an acceptable simplification here.
            oldest = next(iter(_ocr_result_cache))
            del _ocr_result_cache[oldest]

        result_event.result = result  # type: ignore[attr-defined]
        result_event.set()


def start_ocr_worker() -> None:
    """Start the background OCR worker thread, if it isn't already running.

    Idempotent and safe to call from multiple places — both
    ``main.py``'s startup warmup and every :func:`extract_text_at_position`
    call do — the underlying thread is only ever created once.
    """
    global _ocr_worker_started
    with _ocr_worker_start_lock:
        if _ocr_worker_started:
            return
        _ocr_worker_started = True
        threading.Thread(target=_ocr_worker, name="ocr-worker", daemon=True).start()


def extract_text_at_position(x: float, y: float, timeout: float = _OCR_WORKER_TIMEOUT_S) -> str | None:
    """OCR the region around ``(x, y)`` and return the line nearest the cursor.

    Delegates the actual capture + inference to the persistent
    :func:`_ocr_worker` thread via :data:`_ocr_request_queue`, rather than
    running it inline — a burst of hovers this way never backs up behind
    each other, since a still-queued (and by now stale) request is dropped
    the moment a newer one arrives.

    Args:
        x: Horizontal screen coordinate (top-left origin, points).
        y: Vertical screen coordinate.
        timeout: How long to wait for the worker to answer before giving up
            (seconds). The startup warmup call passes a much longer value,
            since the worker's first job is the one-time model load.

    Returns:
        The recognised text of the closest line, stripped, or ``None`` when
        nothing was captured on either pass, the model is unavailable, the
        worker didn't answer within ``timeout``, the frontmost app is a
        blocked dev tool (:data:`_OCR_BLOCKED_APPS`), Chrome's active tab is
        a blocked AI chat site (:data:`_BLOCKED_DOMAINS`), the region is too
        low-contrast to hold text, or no detected line clears both
        :data:`_MIN_OCR_CONFIDENCE` and having at least one Japanese
        character.

    Gating (frontmost app / Chrome URL) runs *before* the cache lookup, not
    after. It used to run after, which meant a hover that landed on an
    already-cached screen position skipped gating entirely — so switching
    from an allowed app/tab to a blocked one (e.g. an NHK Chrome tab to a
    claude.ai tab) without moving the cursor would still return the
    *previous* tab's cached text for that same screen coordinate, since the
    cache was consulted first and gating never got a chance to say no.
    ``get_frontmost_app()`` costs ~10-50 ms, trivial next to the OCR pass a
    cache hit exists to save, so paying it on every call (not just cache
    misses) is a small, worthwhile price for correctness.

    Checks ``_ocr_instance`` directly rather than calling :func:`_get_model`
    here — :func:`_get_model` blocks on ``_model_lock`` with no timeout
    while the model loads (several seconds, longer on a cold weights-
    download), and a hover landing during that window would otherwise
    block this whole request for however much of the load remains. The
    worker thread (see :func:`_ocr_worker`) is the one place that should
    ever pay that cost; every other caller just treats "not loaded yet" the
    same as "not available" and returns fast — the next hover, once loading
    finishes, proceeds normally.
    """
    if _ocr_instance is None:
        return None

    # Gated on the frontmost app, not just cursor position: dev tools and
    # this backend's own terminal are never a valid OCR target (VS Code text
    # the AX tree misses still isn't web content).
    frontmost = get_frontmost_app()
    blocked = frontmost in _OCR_BLOCKED_APPS

    # The app-level check above sees "Google Chrome" for every website, so
    # AI chat sites need a second, URL-based gate — otherwise Mirume reads
    # Claude's own Japanese-language replies as text to classify.
    if not blocked and frontmost == "Google Chrome":
        url = get_chrome_url()
        blocked = any(domain in url for domain in _BLOCKED_DOMAINS)

    if blocked:
        # Drop anything still queued for the worker too — it was enqueued
        # for a screen position that's no longer valid to OCR (the app/tab
        # changed since), so there's no point letting it run.
        try:
            _ocr_request_queue.get_nowait()
        except queue.Empty:
            pass
        return None

    cache_key = _ocr_cache_key(x, y)
    if cache_key in _ocr_result_cache:
        return _ocr_result_cache[cache_key]

    start_ocr_worker()

    result_event = threading.Event()
    result_event.result = None  # type: ignore[attr-defined]

    # Drop a previous, now-stale request rather than let it (or this one)
    # queue up behind it — only the newest cursor position is ever worth an
    # OCR pass; the cursor has usually moved on by the time an older request
    # would be processed anyway.
    try:
        _ocr_request_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        _ocr_request_queue.put_nowait((x, y, result_event))
    except queue.Full:
        return None

    result_event.wait(timeout=timeout)
    return result_event.result  # type: ignore[attr-defined]


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
