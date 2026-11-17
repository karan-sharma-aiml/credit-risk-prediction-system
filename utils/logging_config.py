"""Logging configuration shared by every entry point."""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once, writing to stdout.

    Safe to call multiple times: existing handlers are replaced rather than
    duplicated, so repeated calls (CLI + tests) never produce double output.

    Args:
        level: Logging level as an ``int`` or a name such as ``"DEBUG"``.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
    # Third-party libraries are chatty at INFO; keep our own logs readable.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
