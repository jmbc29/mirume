"""macOS Accessibility (AX) API text detection for Mirume.

This reads the text of on-screen UI elements directly from any application via
the system Accessibility API — no OCR, no screenshots. It is the primary text
source for the ``/hover`` endpoint; ``ocr.py`` is the fallback for image/video
content the AX tree does not expose.

Public API:

* :func:`get_text_at_position` – text of the element under a screen coordinate.
* :func:`get_text_with_ocr_fallback` – as above, falling back to screenshot OCR
  (:mod:`ocr`) for content the AX tree does not expose.
* :func:`get_focused_text` – text of the currently focused element (fallback).
* :func:`accessibility_permission_granted` – is AX access allowed for this
  process (optionally triggering the system prompt).
* :func:`require_accessibility_permission` – raise :class:`PermissionError`
  with setup instructions if it is not.

macOS attributes this process's AX permission to the app that launched it
(your terminal, or the Python binary). Grant it once in
**System Settings ▸ Privacy & Security ▸ Accessibility** and restart the
process.

Run ``python accessibility.py`` to print the text under the cursor every two
seconds for ten seconds.
"""

from __future__ import annotations

import re
import time

# --------------------------------------------------------------------------- #
# Framework import (guarded so the module still imports on a non-mac box / when
# pyobjc is missing — every public function then degrades to "no access").
# --------------------------------------------------------------------------- #

_AX_AVAILABLE = True
_AX_IMPORT_ERROR: str | None = None
try:  # pragma: no cover - platform dependent
    from ApplicationServices import (
        AXIsProcessTrusted,
        AXIsProcessTrustedWithOptions,
        AXUIElementCopyAttributeValue,
        AXUIElementCopyElementAtPosition,
        AXUIElementCreateSystemWide,
    )
    from Quartz import CGEventCreate, CGEventGetLocation
except Exception as exc:  # pragma: no cover - platform dependent
    _AX_AVAILABLE = False
    _AX_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: AX error code for a successful call.
_AX_SUCCESS = 0

#: Option key that makes :func:`AXIsProcessTrustedWithOptions` show the system
#: "grant accessibility access" dialog. This is a stable documented string.
_PROMPT_OPTION_KEY = "AXTrustedCheckOptionPrompt"

#: Element attributes checked, in order, for a text value. Stable AX strings.
_TEXT_ATTRIBUTES: tuple[str, ...] = (
    "AXValue",
    "AXSelectedText",
    "AXTitle",
    "AXDescription",
    "AXHelp",
)

_CHILDREN_ATTRIBUTE = "AXChildren"
_FOCUSED_UI_ELEMENT_ATTRIBUTE = "AXFocusedUIElement"

#: How deep to descend the AX subtree looking for text, and how many children
#: to visit per level, when the element under the cursor is a container.
_MAX_DESCEND_DEPTH = 2
_MAX_CHILDREN_PER_LEVEL = 30
#: Stop concatenating child text once this many characters have been collected.
_MAX_TEXT_CHARS = 4000

PERMISSION_INSTRUCTIONS = (
    "Mirume needs macOS Accessibility permission to read on-screen text.\n"
    "  1. Open System Settings ▸ Privacy & Security ▸ Accessibility\n"
    "  2. Enable the app running this process (your terminal, or Python)\n"
    "  3. Restart the process\n"
    "If the app is already listed, toggle it off and on again."
)


# --------------------------------------------------------------------------- #
# Permission handling
# --------------------------------------------------------------------------- #


def accessibility_permission_granted(prompt: bool = False) -> bool:
    """Return whether this process may use the Accessibility API.

    Args:
        prompt: If ``True``, show the one-time macOS dialog asking the user to
            grant access when it is not already granted. The dialog is
            non-blocking; this call still returns the *current* (usually still
            ``False``) state.

    Returns:
        ``True`` if AX access is granted, ``False`` otherwise (including when
        pyobjc/the frameworks are unavailable).
    """
    if not _AX_AVAILABLE:
        return False
    try:
        if prompt:
            return bool(AXIsProcessTrustedWithOptions({_PROMPT_OPTION_KEY: True}))
        return bool(AXIsProcessTrusted())
    except Exception:
        return False


def require_accessibility_permission() -> None:
    """Raise :class:`PermissionError` with instructions if AX access is denied.

    Raises:
        PermissionError: If Accessibility permission is not granted, or the
            Accessibility frameworks could not be imported.
    """
    if _AX_AVAILABLE and accessibility_permission_granted():
        return
    detail = PERMISSION_INSTRUCTIONS
    if not _AX_AVAILABLE:
        detail = (
            f"Accessibility frameworks are unavailable ({_AX_IMPORT_ERROR}).\n"
            + detail
        )
    raise PermissionError(detail)


# --------------------------------------------------------------------------- #
# Low-level AX helpers
# --------------------------------------------------------------------------- #


def _copy_attribute(element: object, attribute: str) -> object | None:
    """Return one AX attribute value for ``element``, or ``None`` on any error.

    Args:
        element: An ``AXUIElementRef``.
        attribute: The AX attribute name (e.g. ``"AXValue"``).

    Returns:
        The bridged attribute value, or ``None`` if the attribute is missing,
        unsupported, or the call fails.
    """
    try:
        error, value = AXUIElementCopyAttributeValue(element, attribute, None)
    except Exception:
        return None
    if error != _AX_SUCCESS:
        return None
    return value


def _clean(text: str) -> str:
    """Collapse runs of spaces/tabs and trim, preserving line breaks.

    Args:
        text: Raw text pulled from an AX attribute.

    Returns:
        The normalised text (may be empty).
    """
    text = re.sub(r"[ \t　]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def _direct_text(element: object) -> str | None:
    """Return the first non-empty text attribute of ``element`` itself.

    Args:
        element: An ``AXUIElementRef``.

    Returns:
        The text, or ``None`` if the element carries no textual attribute.
    """
    for attribute in _TEXT_ATTRIBUTES:
        value = _copy_attribute(element, attribute)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_text(element: object | None, depth: int = _MAX_DESCEND_DEPTH) -> str | None:
    """Extract readable text from ``element``, descending into children if needed.

    Leaf controls (static text, text fields) expose their text directly. For a
    container the immediate children are visited up to ``depth`` levels and
    their text joined.

    Args:
        element: An ``AXUIElementRef`` or ``None``.
        depth: Remaining levels of child descent allowed.

    Returns:
        The extracted text, or ``None`` if nothing textual was found.
    """
    if element is None:
        return None

    direct = _direct_text(element)
    if direct:
        cleaned = _clean(direct)
        if cleaned:
            return cleaned

    if depth <= 0:
        return None

    children = _copy_attribute(element, _CHILDREN_ATTRIBUTE)
    if not children:
        return None

    parts: list[str] = []
    collected = 0
    for child in list(children)[:_MAX_CHILDREN_PER_LEVEL]:
        child_text = _extract_text(child, depth - 1)
        if child_text:
            parts.append(child_text)
            collected += len(child_text)
            if collected >= _MAX_TEXT_CHARS:
                break

    combined = _clean(" ".join(parts))
    return combined or None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def get_text_at_position(x: float, y: float) -> str | None:
    """Return the text of the UI element at screen coordinate ``(x, y)``.

    Coordinates are in screen points with the origin at the top-left of the
    main display (the same convention as the AX API and
    :func:`Quartz.CGEventGetLocation`).

    Args:
        x: Horizontal screen coordinate.
        y: Vertical screen coordinate.

    Returns:
        The element's text, or ``None`` if there is no element there, it has no
        text, permission is not granted, or the API call fails.
    """
    if not accessibility_permission_granted():
        return None
    try:
        system_wide = AXUIElementCreateSystemWide()
        error, element = AXUIElementCopyElementAtPosition(
            system_wide, float(x), float(y), None
        )
        if error != _AX_SUCCESS or element is None:
            return None
        return _extract_text(element)
    except Exception:
        return None


def get_focused_text() -> str | None:
    """Return the text of the currently focused UI element, or ``None``.

    Used as a fallback when :func:`get_text_at_position` finds nothing (for
    example, the cursor is over a decoration but the user is typing in a field).

    Returns:
        The focused element's text, or ``None`` if there is no focused element,
        it has no text, permission is not granted, or the API call fails.
    """
    if not accessibility_permission_granted():
        return None
    try:
        system_wide = AXUIElementCreateSystemWide()
        focused = _copy_attribute(system_wide, _FOCUSED_UI_ELEMENT_ATTRIBUTE)
        if focused is None:
            return None
        return _extract_text(focused)
    except Exception:
        return None


def get_text_with_ocr_fallback(x: float, y: float) -> tuple[str | None, str]:
    """Return the text under ``(x, y)`` and how it was obtained.

    Tries the Accessibility API first (fast, exact, but blind to sandboxed
    renderers like Chrome's web content). If that finds nothing, falls back to
    a screenshot + manga-ocr pass (:func:`ocr.extract_text_at_position`) —
    but only when the frontmost app isn't a dev tool / terminal
    (``ocr._OCR_BLOCKED_APPS``), since OCR has no way to tell "sandboxed web
    content the AX tree can't see" apart from "a code editor's own UI text
    the AX tree also happens to miss".

    Args:
        x: Horizontal screen coordinate (top-left origin, points).
        y: Vertical screen coordinate.

    Returns:
        ``(text, "accessibility")`` if the AX tree had text, ``(text, "ocr")``
        if OCR recovered it, or ``(None, "none")`` if both failed or the
        frontmost app is blocked from OCR.
    """
    ax_text = get_text_at_position(x, y)
    if ax_text:
        return ax_text, "accessibility"

    try:
        from ocr import _OCR_BLOCKED_APPS, extract_text_at_position, get_frontmost_app
    except Exception:
        return None, "none"

    if get_frontmost_app() in _OCR_BLOCKED_APPS:
        return None, "none"

    ocr_text = extract_text_at_position(x, y)
    if ocr_text:
        return ocr_text, "ocr"
    return None, "none"


def current_mouse_position() -> tuple[float, float]:
    """Return the current mouse location in top-left-origin screen points.

    Returns:
        An ``(x, y)`` tuple. ``(0.0, 0.0)`` if the frameworks are unavailable.
    """
    if not _AX_AVAILABLE:
        return (0.0, 0.0)
    location = CGEventGetLocation(CGEventCreate(None))
    return (float(location.x), float(location.y))


# --------------------------------------------------------------------------- #
# Manual verification
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if not accessibility_permission_granted():
        # Trigger the system prompt, then explain what to do.
        accessibility_permission_granted(prompt=True)
        print(PERMISSION_INSTRUCTIONS)
        print("\nWaiting — grant permission and re-run.\n")

    print("Reading text under the cursor for 10 seconds...\n")
    for step in range(5):
        px, py = current_mouse_position()
        snippet = get_text_at_position(px, py)
        if snippet is not None and len(snippet) > 120:
            snippet = snippet[:120] + "…"
        print(f"[{step * 2:>2}s] cursor=({px:6.0f}, {py:6.0f})  text={snippet!r}")
        time.sleep(2)
