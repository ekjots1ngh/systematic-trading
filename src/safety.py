"""
safety.py
---------
The human-in-control layer. Three independent safeguards:

  1) KILL SWITCH. If a file named HALT exists in the project root, the system refuses
     to trade. Creating that file (or running `run_live.py --halt`) stops everything
     instantly, no code change or restart needed. Remove it (or `--resume`) to re-enable.

  2) CIRCUIT BREAKER. If the account has lost more than a configured fraction of equity
     versus the recorded high-water mark, trading halts automatically and an alert fires.
     This caps how bad a single bad day (or a bug) can get before a human looks.

  3) APPROVAL GATE. Any single order whose notional exceeds a configured limit is not
     sent automatically. It is parked in audit/pending_orders.json, the human is alerted,
     and it only executes after explicit approval (`run_live.py --approve`). Small,
     routine orders flow through; only large ones need a human to say yes.
"""

import json
import os

from .audit import AUDIT_DIR

ROOT = os.path.join(os.path.dirname(__file__), "..")
HALT_FILE = os.path.join(ROOT, "HALT")
PENDING_PATH = os.path.join(AUDIT_DIR, "pending_orders.json")
HWM_PATH = os.path.join(AUDIT_DIR, "high_water_mark.json")


# ----- 1) kill switch -----
def is_halted() -> bool:
    return os.path.exists(HALT_FILE)


def set_halt(reason: str = "manual"):
    with open(HALT_FILE, "w") as f:
        f.write(reason + "\n")


def clear_halt():
    if os.path.exists(HALT_FILE):
        os.remove(HALT_FILE)


# ----- 2) circuit breaker -----
def update_high_water_mark(equity: float) -> float:
    hwm = equity
    if os.path.exists(HWM_PATH):
        try:
            hwm = max(hwm, json.load(open(HWM_PATH)).get("hwm", equity))
        except Exception:
            pass
    json.dump({"hwm": hwm}, open(HWM_PATH, "w"))
    return hwm


def circuit_breaker_tripped(equity: float, max_drawdown: float) -> tuple[bool, float]:
    """Returns (tripped, drawdown). drawdown is fraction below the high-water mark."""
    hwm = update_high_water_mark(equity)
    dd = 0.0 if hwm <= 0 else (equity / hwm - 1.0)
    return (dd <= -abs(max_drawdown)), dd


# ----- 3) approval gate -----
def needs_approval(order: dict, approval_threshold: float) -> bool:
    return order.get("notional", 0.0) > approval_threshold


def queue_for_approval(orders: list[dict]):
    existing = load_pending()
    json.dump(existing + orders, open(PENDING_PATH, "w"), indent=2, default=str)


def load_pending() -> list[dict]:
    if not os.path.exists(PENDING_PATH):
        return []
    try:
        return json.load(open(PENDING_PATH))
    except Exception:
        return []


def clear_pending():
    if os.path.exists(PENDING_PATH):
        os.remove(PENDING_PATH)
