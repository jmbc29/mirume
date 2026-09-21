"""File logging for the Mirume backend, for diagnosing crashes users report.

The packaged app has no visible terminal, so a print() statement or an
unhandled exception normally vanishes. :func:`setup_logging` attaches a
rotating file handler to the root logger (every module's ``logging.getLogger``
call inherits it) and installs hooks that catch what ``logging`` calls alone
would miss:

* ``sys.excepthook`` — an unhandled exception on the main thread.
* ``threading.excepthook`` — an unhandled exception on a background thread
  (the OCR worker, the parent-process watchdog).

Log file: ``~/Library/Logs/Mirume/mirume.log`` — the standard per-user log
location on macOS, so it survives app updates and is easy to find (Console.app
or ``~/Library/Logs/Mirume/``). Rotated at 5 MB, keeping 3 backups.
"""

from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR: Path = Path.home() / "Library" / "Logs" / "Mirume"
LOG_FILE: Path = LOG_DIR / "mirume.log"

_configured = False


def setup_logging() -> None:
    """Attach the rotating file handler and crash hooks, once per process.

    Safe to call multiple times (from both ``main.py`` and
    ``mirume_server.py``) — only the first call takes effect.
    """
    global _configured
    if _configured:
        return
    _configured = True

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)

    def _log_unhandled(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logging.getLogger("mirume.crash").critical(
            "Unhandled exception on main thread", exc_info=(exc_type, exc_value, exc_tb)
        )
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    def _log_unhandled_thread(args: "threading.ExceptHookArgs") -> None:
        logging.getLogger("mirume.crash").critical(
            "Unhandled exception on thread %r",
            args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _log_unhandled
    threading.excepthook = _log_unhandled_thread

    logging.getLogger("mirume").info("Logging initialised -> %s", LOG_FILE)
