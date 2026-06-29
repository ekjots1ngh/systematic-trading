"""
audit.py
--------
Logging and an append-only trade ledger. Two outputs:

  audit/trader.log     human-readable log of everything the system does
  audit/ledger.jsonl   one JSON record per event, machine-readable, append-only

The ledger is the source of truth for traceability. Every signal decision, every
order, every approval, every fill and every halt is written here with a timestamp
and a run id, so the complete life of any trade can be reconstructed after the fact.
Append-only means records are never edited or deleted, which is what makes it an
audit trail rather than just a log.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone

AUDIT_DIR = os.path.join(os.path.dirname(__file__), "..", "audit")
LEDGER_PATH = os.path.join(AUDIT_DIR, "ledger.jsonl")
LOG_PATH = os.path.join(AUDIT_DIR, "trader.log")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_logger(name: str = "trader") -> logging.Logger:
    os.makedirs(AUDIT_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s")
    fh = logging.FileHandler(LOG_PATH)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class Ledger:
    """Append-only event ledger. Each run gets a unique run_id tying its events together."""

    def __init__(self, run_id: str | None = None):
        os.makedirs(AUDIT_DIR, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:12]

    def record(self, event: str, **fields) -> dict:
        """Write one event record. `event` is a short type tag; fields are free-form."""
        rec = {"ts": _now(), "run_id": self.run_id, "event": event}
        rec.update(fields)
        with open(LEDGER_PATH, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    @staticmethod
    def read_all() -> list[dict]:
        if not os.path.exists(LEDGER_PATH):
            return []
        with open(LEDGER_PATH) as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def history(symbol: str | None = None, limit: int = 50) -> list[dict]:
        """Return recent order/fill events, optionally for one symbol."""
        rows = [r for r in Ledger.read_all()
                if r["event"] in ("order_submitted", "order_pending_approval", "fill")]
        if symbol:
            rows = [r for r in rows if r.get("symbol") == symbol]
        return rows[-limit:]
