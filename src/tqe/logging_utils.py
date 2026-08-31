"""Structured logging for the Treasury Quant Engine.

A single ``setup_logging`` entry point is used by the CLI, the live runner and the
API so that every process emits the same format.  Live trading additionally
writes a JSON-lines audit log which is the record of record for order events.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CONFIGURED = False

LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per line - suitable for shipping to a log store."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, default=str)


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    json_file: str | Path | None = None,
) -> logging.Logger:
    """Configure root logging once per process.

    Parameters
    ----------
    level:
        Console log level.
    log_file:
        Optional human readable rotating log file.
    json_file:
        Optional JSON-lines audit log (used by the live trading runner).
    """
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        root.setLevel(LEVELS.get(level.upper(), logging.INFO))
        return logging.getLogger("tqe")

    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(LEVELS.get(level.upper(), logging.INFO))
    console.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s", "%H:%M:%S")
    )
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=20_000_000, backupCount=5)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
        root.addHandler(fh)

    if json_file:
        path = Path(json_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        jh = logging.handlers.RotatingFileHandler(path, maxBytes=50_000_000, backupCount=10)
        jh.setLevel(logging.INFO)
        jh.setFormatter(JsonFormatter())
        root.addHandler(jh)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    _CONFIGURED = True
    return logging.getLogger("tqe")


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger (``tqe.<name>``)."""
    if not name.startswith("tqe"):
        name = f"tqe.{name}"
    return logging.getLogger(name)


def audit(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a structured audit event.

    Every order lifecycle transition flows through here so the JSON audit log can
    be replayed to reconstruct the exact state of the OMS at any point in time.
    """
    logger.info(event, extra={"extra_fields": {"event": event, **fields}})


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable (``1/true/yes/on``)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
