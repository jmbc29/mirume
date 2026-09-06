"""Entry point for the packaged Mirume backend.

This is what the PyInstaller build freezes and what ``Mirume.app`` launches as
a child process. It differs from the development ``uvicorn main:app`` command
in two ways:

* It seeds the writable data directory from the read-only files shipped in the
  app bundle (:func:`paths.seed_from_bundle`) *before* importing the app, so
  the dictionary database is in place by the time anything opens it.
* It runs uvicorn programmatically with no ``--reload`` / file watcher — the
  reloader is a development-only convenience and ``watchfiles`` is not bundled.

The host/port match what the frontend hard-codes (``127.0.0.1:8123``). Run it
directly (``python mirume_server.py``) to exercise the packaged code path from
a source checkout.
"""

from __future__ import annotations

import os
import threading
import time

import uvicorn

from paths import seed_from_bundle

#: Loopback address + port the Tauri frontend expects (see
#: ``frontend/src/hooks/useMouseTracker.ts`` and ``ReviewApp.tsx``).
HOST = "127.0.0.1"
PORT = 8123


def _exit_with_parent() -> None:
    """Terminate this process if the launching Tauri app goes away.

    ``frontend/src/lib.rs`` kills the backend on a clean quit, but a crash
    would leave it orphaned and holding port 8123. When the parent dies this
    process is reparented to ``launchd`` (PID 1), so a change of parent PID is
    the signal to exit. No-op when launched directly (no ``MIRUME_PARENT_PID``).
    """
    expected = os.environ.get("MIRUME_PARENT_PID", "").strip()
    if not expected:
        return

    def _watch() -> None:
        while True:
            time.sleep(2)
            if os.getppid() != int(expected):
                os._exit(0)

    threading.Thread(target=_watch, name="parent-watchdog", daemon=True).start()


def _selftest() -> int:
    """Smoke-test the parts most likely to break in a frozen build, then exit.

    Run with ``MIRUME_SELFTEST=1`` (the packaged binary has no other way to
    exercise a submodule). Checks the MeCab tokeniser, the fastText model load
    and — the fragile one — that macOS Vision text recognition works from
    inside the PyInstaller bundle.
    """
    seed_from_bundle()
    ok = True

    try:
        from tokeniser import tokenise

        assert [t.surface for t in tokenise("日本語")], "empty tokenisation"
        print("[selftest] tokeniser OK")
    except Exception as exc:
        ok = False
        print(f"[selftest] tokeniser FAILED: {exc!r}")

    try:
        from PIL import Image, ImageDraw, ImageFont

        import ocr

        img = Image.new("RGB", (360, 90), (255, 255, 255))
        for path in (
            "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
        ):
            try:
                font = ImageFont.truetype(path, 44)
                break
            except OSError:
                font = None
        ImageDraw.Draw(img).text((16, 18), "日本語", font=font, fill=(0, 0, 0))
        lines = ocr._recognise_lines(img)
        assert any("日" in text for text, _c, _p in lines), f"no Japanese recognised: {lines}"
        print(f"[selftest] Vision OCR OK ({lines[0][0]!r})")
    except Exception as exc:
        ok = False
        print(f"[selftest] Vision OCR FAILED: {exc!r}")

    print("[selftest] PASS" if ok else "[selftest] FAIL")
    return 0 if ok else 1


def main() -> None:
    """Seed bundled data, then serve the FastAPI app until the process exits."""
    if os.environ.get("MIRUME_SELFTEST") == "1":
        raise SystemExit(_selftest())

    _exit_with_parent()
    seed_from_bundle()

    # Import only after seeding: main -> database/jlpt/sentences open engines
    # bound to files under the (now populated) data directory.
    from main import app

    log_level = os.environ.get("MIRUME_LOG_LEVEL", "info")
    uvicorn.run(app, host=HOST, port=PORT, log_level=log_level, access_log=False)


if __name__ == "__main__":
    main()
