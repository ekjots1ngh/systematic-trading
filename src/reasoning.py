"""
reasoning.py
------------
Turns the numbers behind a position into a sentence a human can check.

Nothing here is invented after the fact. The rationale is built from the exact values
the strategy used to size the trade: each sub-strategy's signal, the blended conviction,
the trailing volatility, the vol-target scaler, the resulting target weight, the current
position, and the order that closes the gap. If the explanation and the order ever
disagree, that is a bug you can see, which is the whole point.
"""

import numpy as np


def describe_instrument(symbol: str, comps: dict, vol: float, scaler: float,
                        target_weight: float, current_units: float,
                        order: dict | None) -> str:
    direction = "long" if target_weight > 0 else "short" if target_weight < 0 else "flat"
    parts = []
    # which sub-strategies drove it
    members = {k: v for k, v in comps.items() if k != "combined"}
    desc = ", ".join(f"{name} {v['signal']:+.2f}" for name, v in members.items())
    parts.append(f"{symbol}: target {direction}.")
    parts.append(f"Signals [{desc}] blend to conviction {comps['combined']:+.2f}.")
    parts.append(f"Trailing vol {vol:.0%}, so vol-target scaler {scaler:.2f}x "
                 f"gives target weight {target_weight:+.2%} of capital.")
    if order is None:
        parts.append(f"Current position already on target; no trade.")
    else:
        parts.append(f"Currently {current_units:+.4g} units; "
                     f"{order['side']} {order['qty']:g} (~${order['notional']:,.0f}) to reach target.")
    return " ".join(parts)


def one_line(symbol: str, order: dict, comps: dict) -> str:
    """A compact rationale suitable for an alert or a ledger entry."""
    return (f"{order['side'].upper()} {order['qty']:g} {symbol} "
            f"(~${order['notional']:,.0f}); conviction {comps['combined']:+.2f}")
