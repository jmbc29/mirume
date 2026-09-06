"""Screenshot OCR fallback for Mirume, backed by the macOS Vision framework.

The macOS Accessibility API (:mod:`accessibility`) cannot see inside a
sandboxed renderer — Chrome, Electron apps, canvas/WebGL content, video — so
for that content ``/hover`` falls back to grabbing a small screenshot around
the cursor and running text recognition over it. Recognition uses
``VNRecognizeTextRequest`` from Apple's `Vision
<https://developer.apple.com/documentation/vision>`_ framework: it is built
into macOS (no model download, no multi-hundred-MB dependency), has native
Japanese + English support from macOS 13 on, and returns each line of text
with its own bounding box. Of everything found in the captured region, only
the single line whose box sits closest to the cursor is returned (never more
than one — see :data:`_OCR_MAX_BOX_DISTANCE_PX` for the cutoff that keeps a
cursor sitting between two blocks from matching whichever is barely nearer).

This replaced a PaddleOCR (``paddlepaddle`` + ``paddleocr``) detect+recognise
pipeline: equivalent per-line results, but ~1 GB less to bundle, nothing to
download on first run, and no native-library fragility in the packaged app.
An earlier iteration before that used `manga-ocr
<https://github.com/kha-white/manga-ocr>`_, which read the captured region as
a single blob and blended unrelated nearby UI elements into one hallucinated
string on a dense page.

Public API:

* :func:`capture_region` – screenshot a rectangle of the screen as a
  :class:`PIL.Image.Image`, via the ``screencapture`` CLI. It grabs the
  *composited* display exactly as the user sees it, so Mirume's own
  transparent full-screen overlay contributes nothing and the web content
  underneath shows through (``CGWindowListCreateImage`` composited the empty
  overlay on top instead, yielding the desktop wallpaper).
* :func:`extract_text_at_position` – OCR a region centred on a screen
  coordinate; returns the text of whichever recognised line sits closest to
  the cursor, when it actually contains Japanese, the frontmost app isn't a
  dev tool (see :data:`_OCR_BLOCKED_APPS`), and — when the frontmost app is
  Chrome — the active tab isn't an AI chat site (see :data:`_BLOCKED_DOMAINS`).
* :func:`start_ocr_worker` – start the persistent background thread that owns
  every recognition call (see :func:`_ocr_worker`); called from ``main.py``'s
  startup warmup and, idempotently, from every
  :func:`extract_text_at_position` call in case it hasn't been already.
* :func:`get_frontmost_app` – name of the active application, used to gate
  OCR off code editors/terminals whose own UI text isn't meant to be read.
* :func:`get_chrome_url` – active tab URL when Chrome is frontmost, used to
  tell an actual webpage apart from an AI chat site running in the browser.

Vision does its own model management inside the OS; the first request pays a
small one-time warmup which :func:`_ocr_worker` absorbs on the backend's
startup thread so the first real hover is fast.
"""

from __future__ import annotations

import io
import os
import queue
import re
import subprocess
import tempfile
import threading
import time

# --------------------------------------------------------------------------- #
# Optional-dependency guard. Pillow and pyobjc's Vision/Foundation bindings
# must all import for OCR to work; the module still imports (degrading to
# "no OCR") when any of them is missing so the rest of the backend keeps
# working — e.g. on a non-mac box, or a stripped build.
# --------------------------------------------------------------------------- #

try:
    from PIL import Image, ImageGrab
except Exception as exc:  # pragma: no cover - dependency missing
    Image = None  # type: ignore[assignment]
    ImageGrab = None  # type: ignore[assignment]
    _PIL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:  # pragma: no cover - platform dependent
    import Foundation
    import Vision

    _VISION_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - dependency missing / non-mac
    Foundation = None  # type: ignore[assignment]
    Vision = None  # type: ignore[assignment]
    _VISION_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

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
#: Tall enough for Vision to isolate the line the cursor is on from the lines
#: immediately above/below it rather than returning a whole paragraph.
_OCR_REGION_WIDTH = 600
_OCR_REGION_HEIGHT = 300

#: Hard ceiling on the ``screencapture`` CLI. It normally returns in well under
#: 200 ms; anything longer means it is wedged (a permission prompt, a stuck
#: WindowServer round-trip inside the uvicorn worker) and we abandon it.
_SCREENCAPTURE_TIMEOUT_S = 2.0

#: How long :func:`extract_text_at_position` waits for the background OCR
#: worker (see :func:`_ocr_worker`) to answer before giving up on this hover.
#: Capture (~200 ms, or the 2 s ``screencapture`` timeout) plus Vision
#: recognition (~150-300 ms), possibly doubled by the one retry pass, is the
#: normal turnaround; a wait longer than that means the worker is still busy
#: with a previous (now stale) request and it's better to skip.
_OCR_WORKER_TIMEOUT_S = 2.5

#: Codepoint range that counts as "Japanese" — U+3040..U+9FFF spans hiragana,
#: katakana, the CJK symbols/punctuation block and the CJK unified ideographs
#: (kanji). Used to reject a recognised line with none of these — Vision, told
#: to expect ``["ja", "en"]``, still returns English UI chrome caught in the
#: capture region.
_JAPANESE_RE = re.compile(r"[぀-鿿]")

#: Set once we've warned that ``screencapture`` is failing (almost always a
#: missing Screen Recording grant), so the log isn't spammed on every hover.
_capture_warned = False

#: Skip OCR entirely when the captured region has too little tonal contrast to
#: contain rendered text — measured as the spread between the 5th and 95th
#: percentile of greyscale pixel values (0-255). Running the recognition
#: request over an empty margin is wasted work.
_MIN_CONTRAST_RANGE = 70

#: Minimum recognition confidence (0-1) for a recognised line to be trusted.
#: Vision's ``accurate`` level reports noticeably lower absolute confidences
#: than PaddleOCR did for equally good Japanese reads (~0.4-0.6 is common), so
#: the bar is lower than the old 0.7 — the real filter is
#: :data:`_JAPANESE_RE`, which drops the English UI text that a lenient
#: threshold lets through.
_MIN_OCR_CONFIDENCE = 0.3

#: Maximum distance (px, in the *captured image's* coordinate space — the
#: cursor always sits at the image centre, see :func:`_run_ocr_pass`) a
#: recognised box's centre may be from the cursor to still count as "the line
#: under the cursor". Without this, the closest box was returned no matter how
#: far away it actually was, so a cursor sitting in the whitespace between two
#: unrelated text blocks would still match whichever was (barely) nearer.
_OCR_MAX_BOX_DISTANCE_PX = 80

#: Vertical shift (screen points) applied to the capture region on a retry
#: pass — see :func:`extract_text_at_position`. Catches text that sat right at
#: the edge of (and so was clipped or missed by) the first pass's region.
_OCR_RETRY_Y_SHIFT = 30

#: Matches a short (<=6 char) unit immediately repeated 3+ times in a row — a
#: recognition failure mode on an ambiguous/small crop where the decoder gets
#: stuck reproducing the same span (e.g. "鹿児島" comes back
#: "鹿児島鹿児島鹿児島鹿児島"). Requires *3 or more* repeats, not 2, so
#: legitimate Japanese reduplication is never touched — 色々, 我々, 時々 and
#: onomatopoeia like わくわく double, they don't triple.
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
# Vision availability
# --------------------------------------------------------------------------- #

#: Languages Vision is told to expect, best-effort first. Japanese is the
#: point; English is included so a mixed line isn't mangled.
_OCR_LANGUAGES = ["ja", "en"]

_ocr_unavailable_logged = False


def _ocr_available() -> bool:
    """Whether screenshot OCR can run at all (Pillow + Vision both imported).

    Logs the reason once when it can't, then stays quiet — a missing Vision
    binding or a non-mac host isn't going to fix itself between hovers.
    """
    global _ocr_unavailable_logged
    if Image is not None and Vision is not None and Foundation is not None:
        return True
    if not _ocr_unavailable_logged:
        _ocr_unavailable_logged = True
        reason = _VISION_IMPORT_ERROR or globals().get("_PIL_IMPORT_ERROR") or "unknown"
        print(f"[mirume] screenshot OCR unavailable ({reason}); Chrome/web hover disabled.")
    return False


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
                    "System Settings > Privacy & Security > Screen Recording to "
                    "Mirume (in a dev run, to the terminal launching the backend), "
                    "then fully quit and reopen it."
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


#: Cache for :func:`_frontmost_context` — ``(monotonic_deadline, (app, url))``.
#: ``/hover`` fires in bursts as the cursor settles and the frontend retries, so
#: without this every one of those would pay a fresh ``osascript`` round-trip
#: (~250 ms) just to re-learn the app hasn't changed. A short TTL keeps the gate
#: honest — you would have to switch to a blocked app *and* hover within the
#: window for a stale "not blocked" to slip through.
_CONTEXT_TTL_S = 0.5
_context_cache: tuple[float, tuple[str, str]] | None = None
_context_lock = threading.Lock()


def _frontmost_context() -> tuple[str, str]:
    """Return ``(frontmost app name, active Chrome tab URL)`` in one round-trip.

    A single ``osascript`` call gets the frontmost process name and, only when
    that is Chrome, its active tab URL (lowercased) — half the subprocess spawn
    cost of calling :func:`get_frontmost_app` and :func:`get_chrome_url`
    separately. The result is cached for :data:`_CONTEXT_TTL_S`.

    Returns:
        ``(app, url)``; ``url`` is ``""`` when Chrome isn't frontmost or its
        URL couldn't be read. ``("", "")`` if the whole call failed.
    """
    global _context_cache
    now = time.monotonic()
    cached = _context_cache
    if cached is not None and now < cached[0]:
        return cached[1]

    script = """
    tell application "System Events"
        set frontApp to name of first application process whose frontmost is true
    end tell
    set theURL to ""
    if frontApp is "Google Chrome" then
        try
            tell application "Google Chrome" to set theURL to URL of active tab of front window
        end try
    end if
    return frontApp & linefeed & theURL
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        app, _, url = result.stdout.strip().partition("\n")
        context = (app.strip(), url.strip().lower())
    except Exception:
        context = ("", "")

    with _context_lock:
        _context_cache = (now + _CONTEXT_TTL_S, context)
    return context


def reading_blocked_here() -> bool:
    """Whether Mirume should read *nothing* from whatever is frontmost right now.

    This gates **both** text sources — the Accessibility tree and the OCR
    fallback — not just OCR. It used to guard only OCR (inside
    :func:`extract_text_at_position`), but once Mirume's cursor polling has
    nudged Chrome into exposing its renderer accessibility tree, a blocked
    site's text (an AI chat, this repo on GitHub, a localhost dev server)
    flows straight out through :func:`accessibility.get_text_at_position`
    with the OCR gate never consulted. So the check has to happen before the
    AX read too — which is what :func:`accessibility.get_text_with_ocr_fallback`
    now does by calling this first.

    Blocked when the frontmost app is a dev tool / terminal / Claude itself
    (:data:`_OCR_BLOCKED_APPS`), or when it is Chrome and the active tab's URL
    is a blocked domain (:data:`_BLOCKED_DOMAINS`) — *or* Chrome is frontmost
    but its URL could not be read at all. That last case fails **closed** on
    purpose: an ``osascript`` timeout used to make :func:`get_chrome_url`
    return ``""``, which the old ``any(domain in url ...)`` check waved
    through as "not blocked". We know it's a browser; if we can't confirm the
    tab is safe, we don't read it.

    Returns:
        ``True`` if nothing under the cursor should be read right now.
    """
    frontmost, url = _frontmost_context()
    if frontmost in _OCR_BLOCKED_APPS:
        return True
    if frontmost == "Google Chrome":
        if not url or any(domain in url for domain in _BLOCKED_DOMAINS):
            return True
    return False


# --------------------------------------------------------------------------- #
# Vision text recognition
# --------------------------------------------------------------------------- #


def _has_text_contrast(image: "Image.Image") -> bool:
    """Return whether ``image`` has enough tonal range to contain rendered text.

    A near-uniform region (blank background, empty margin) is never worth
    running a recognition request over.
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

    See :data:`_REPEATED_UNIT_RE`. Recognition occasionally gets stuck on an
    ambiguous or small crop and reproduces the same short span several times
    in a row instead of ending the line; a garbled "鹿児島鹿児島鹿児島鹿児島"
    becomes "鹿児島" instead of being shown as-is.
    """
    return _REPEATED_UNIT_RE.sub(r"\1", text)


def _image_to_nsdata(image: "Image.Image"):
    """Encode a PIL image as PNG bytes wrapped in an ``NSData``.

    ``VNImageRequestHandler`` accepts encoded image data directly, so there is
    no need to build a ``CGImage`` by hand.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    return Foundation.NSData.dataWithBytes_length_(payload, len(payload))


def _recognise_lines(image: "Image.Image") -> list[tuple[str, float, tuple[float, float]]]:
    """Run Vision text recognition over ``image``.

    Args:
        image: The captured region (RGB, already normalised to point size).

    Returns:
        One ``(text, confidence, (cx, cy))`` tuple per recognised line, where
        ``(cx, cy)`` is the line's bounding-box centre in **pixel** coordinates
        with a top-left origin (matching the capture's own coordinate space).
        Empty on any failure.
    """
    data = _image_to_nsdata(image)
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, {})
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    try:
        request.setRecognitionLanguages_(_OCR_LANGUAGES)
    except Exception:  # pragma: no cover - older Vision without ja support
        pass

    ok, _err = handler.performRequests_error_([request], None)
    if not ok:
        return []

    width, height = image.width, image.height
    lines: list[tuple[str, float, tuple[float, float]]] = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        candidate = candidates[0]
        text = (candidate.string() or "").strip()
        if not text:
            continue
        confidence = float(candidate.confidence())
        # Vision boxes are normalised [0, 1] with a bottom-left origin; convert
        # the centre to top-left-origin pixels so distances compare against the
        # cursor (which sits at the pixel centre of the capture).
        box = observation.boundingBox()
        cx = (box.origin.x + box.size.width / 2) * width
        cy = (1.0 - (box.origin.y + box.size.height / 2)) * height
        lines.append((text, confidence, (cx, cy)))
    return lines


def _run_ocr_pass(x: float, y: float) -> str | None:
    """Capture a :data:`_OCR_REGION_WIDTH` x :data:`_OCR_REGION_HEIGHT` region
    centred on ``(x, y)`` and recognise the line under the cursor.

    Recognition runs over the whole region (Vision needs the vertical context
    to separate the cursor's line from its neighbours), but **only the line
    whose box is nearest the cursor is returned** — walking outward to the next
    few lines only if the nearest turns out to be non-Japanese UI text. So a
    dense page never blends into one hallucinated string.

    Only ever called from :func:`_ocr_worker`, the single thread that issues
    recognition requests.

    Args:
        x: Horizontal screen coordinate the region is centred on.
        y: Vertical screen coordinate the region is centred on.

    Returns:
        The recognised text of the closest qualifying line, stripped, or
        ``None`` when nothing was captured, the region is too low-contrast to
        hold text, no line's box is within :data:`_OCR_MAX_BOX_DISTANCE_PX` of
        the cursor, or none of the nearest lines clears both
        :data:`_MIN_OCR_CONFIDENCE` and having at least one Japanese character.
    """
    left = x - _OCR_REGION_WIDTH / 2
    top = y - _OCR_REGION_HEIGHT / 2
    image = capture_region(left, top, _OCR_REGION_WIDTH, _OCR_REGION_HEIGHT)
    if image is None or not _has_text_contrast(image):
        return None

    centre_x, centre_y = image.width / 2, image.height / 2
    ranked: list[tuple[float, str, float]] = []
    for text, confidence, (cx, cy) in _recognise_lines(image):
        distance = ((cx - centre_x) ** 2 + (cy - centre_y) ** 2) ** 0.5
        if distance <= _OCR_MAX_BOX_DISTANCE_PX:
            ranked.append((distance, text, confidence))
    ranked.sort(key=lambda item: item[0])

    for _distance, text, confidence in ranked:
        if confidence < _MIN_OCR_CONFIDENCE or not _JAPANESE_RE.search(text):
            continue
        text = _collapse_repeated_text(text)
        if len(text) < 2:
            continue
        return text
    return None


# --------------------------------------------------------------------------- #
# Background worker
# --------------------------------------------------------------------------- #

#: OCR results, keyed by a coarse (:data:`_OCR_CACHE_GRID_PX`-rounded) cursor
#: position. Several buckets are kept (unlike a single "last position" slot) so
#: a cursor moving between a few nearby lines — e.g. reading down a paragraph —
#: hits cache on all of them, not just the most recent. Only ever touched from
#: :func:`_ocr_worker`, so it needs no lock.
_ocr_result_cache: dict[tuple[int, int], str | None] = {}
_OCR_CACHE_GRID_PX = 50
_OCR_CACHE_MAX_ENTRIES = 50

#: Single-slot request queue feeding :func:`_ocr_worker`. ``maxsize=1`` plus
#: the drop-then-put in :func:`extract_text_at_position` means a burst of
#: hovers never queues up work for stale positions — only the newest request
#: is ever waiting, since the cursor has usually moved on by the time an older
#: one would be processed anyway.
_ocr_request_queue: "queue.Queue[tuple[float, float, threading.Event]]" = queue.Queue(maxsize=1)
_ocr_worker_started = False
_ocr_worker_start_lock = threading.Lock()

#: Set true once the worker thread has run its warmup pass and is ready to
#: serve requests. :func:`extract_text_at_position` checks this rather than
#: enqueueing work that would just sit behind the (one-time) Vision warmup.
_ocr_ready = False


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


def _warm_up_recognition() -> None:
    """Issue one throwaway recognition request before the worker serves traffic.

    Vision lazily brings up its recognition stack on the first request (a
    few hundred ms); doing it here, on the worker thread at startup, keeps it
    off the first real hover. Runs on a blank in-memory image and never
    touches gating or the screen.
    """
    if Image is None:
        return
    try:
        _recognise_lines(Image.new("RGB", (240, 48), (255, 255, 255)))
    except Exception as exc:  # pragma: no cover - warmup is best-effort
        print(f"[mirume] Vision OCR warmup failed ({exc}); first real hover may be slow.")


def _ocr_worker() -> None:
    """Background thread owning every Vision recognition call.

    Runs one warmup recognition (blocking this thread only — real requests
    just wait on it, see :func:`extract_text_at_position`), then loops pulling
    ``(x, y, result_event)`` off :data:`_ocr_request_queue` forever. For each:
    checks :data:`_ocr_result_cache` first; otherwise runs :func:`_run_ocr_pass`
    at ``(x, y)`` and, if that misses, once more with the region shifted up by
    :data:`_OCR_RETRY_Y_SHIFT` px (catches a line clipped at the first pass's
    edge) — then caches and stores the result on ``result_event`` before
    setting it, waking whichever request thread is waiting.

    Being the sole issuer of recognition requests, this thread is also what
    keeps :func:`_run_ocr_pass` free of any shared lock.

    Runs for the lifetime of the process; started once by
    :func:`start_ocr_worker`.
    """
    global _ocr_ready
    available = _ocr_available()
    if available:
        _warm_up_recognition()
    _ocr_ready = True

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
        if available:
            result = _run_ocr_pass(x, y)
            if result is None:
                result = _run_ocr_pass(x, y - _OCR_RETRY_Y_SHIFT)

        _ocr_result_cache[cache_key] = result
        if len(_ocr_result_cache) > _OCR_CACHE_MAX_ENTRIES:
            # Plain dict preserves insertion order in Python 3.7+, so the first
            # key really is the oldest entry — good enough for a soft perf
            # cache; not a strict LRU (a re-hit doesn't move it to the back),
            # which is an acceptable simplification here.
            oldest = next(iter(_ocr_result_cache))
            del _ocr_result_cache[oldest]

        result_event.result = result  # type: ignore[attr-defined]
        result_event.set()


def start_ocr_worker() -> None:
    """Start the background OCR worker thread, if it isn't already running.

    Idempotent and safe to call from multiple places — both ``main.py``'s
    startup warmup and every :func:`extract_text_at_position` call do — the
    underlying thread is only ever created once.
    """
    global _ocr_worker_started
    with _ocr_worker_start_lock:
        if _ocr_worker_started:
            return
        _ocr_worker_started = True
        threading.Thread(target=_ocr_worker, name="ocr-worker", daemon=True).start()


def extract_text_at_position(
    x: float,
    y: float,
    timeout: float = _OCR_WORKER_TIMEOUT_S,
    *,
    skip_context_check: bool = False,
) -> str | None:
    """OCR the region around ``(x, y)`` and return the line nearest the cursor.

    Delegates the actual capture + recognition to the persistent
    :func:`_ocr_worker` thread via :data:`_ocr_request_queue`, rather than
    running it inline — a burst of hovers this way never backs up behind each
    other, since a still-queued (and by now stale) request is dropped the
    moment a newer one arrives.

    Args:
        x: Horizontal screen coordinate (top-left origin, points).
        y: Vertical screen coordinate.
        timeout: How long to wait for the worker to answer before giving up
            (seconds). The startup warmup call passes a longer value, since the
            worker's first job is the one-time Vision warmup.
        skip_context_check: Set by :func:`accessibility.get_text_with_ocr_fallback`,
            which has *already* called :func:`reading_blocked_here` before its
            AX read — no point paying the ~0.5 s of ``osascript`` round-trips
            again. Any other caller leaves this ``False`` so the gate still
            runs.

    Returns:
        The recognised text of the closest line, stripped, or ``None`` when
        nothing was captured on either pass, OCR is unavailable, the worker
        didn't answer within ``timeout``, reading is blocked here
        (:func:`reading_blocked_here`), the region is too low-contrast to hold
        text, or no recognised line clears both :data:`_MIN_OCR_CONFIDENCE` and
        having at least one Japanese character.

    Gating (:func:`reading_blocked_here`) runs *before* the cache lookup, not
    after. It used to run after, which meant a hover that landed on an
    already-cached screen position skipped gating entirely — so switching from
    an allowed app/tab to a blocked one (e.g. an NHK Chrome tab to a claude.ai
    tab) without moving the cursor would still return the *previous* tab's
    cached text for that same screen coordinate.

    Checks :data:`_ocr_ready` rather than issuing work before the worker's
    one-time Vision warmup has finished — a hover landing in that window
    returns fast (treated the same as "not available"); the next hover, once
    warmup is done, proceeds normally.
    """
    if not _ocr_ready or not _ocr_available():
        start_ocr_worker()
        return None

    if not skip_context_check and reading_blocked_here():
        # Drop anything still queued for the worker too — it was enqueued for a
        # screen position that's no longer valid to OCR (the app/tab changed
        # since), so there's no point letting it run.
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

    # Drop a previous, now-stale request rather than let it (or this one) queue
    # up behind it — only the newest cursor position is ever worth an OCR pass;
    # the cursor has usually moved on by the time an older request would run.
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

    start_ocr_worker()
    for _ in range(60):
        if _ocr_ready:
            break
        time.sleep(0.5)

    result = extract_text_at_position(args.x, args.y, timeout=10.0)
    print("ocr result:", repr(result))
