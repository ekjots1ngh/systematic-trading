"""
live.py
-------
The bridge from backtest to a real account. It computes today's target positions with
the *exact same* strategy and sizing code used in the backtest, then reconciles a live
or paper Alpaca account towards those targets by placing orders.

Why Alpaca: commission-free, supports fractional shares and crypto, has a clean REST
API and a proper paper-trading sandbox, so you can run the identical system on fake
money first and on real money once it has earned your trust. (For genuine futures you
would swap this adapter for Interactive Brokers; the strategy layer does not change.)

SAFETY RAILS (deliberate, do not remove lightly):
  - Defaults to PAPER trading and DRY-RUN (prints orders, submits nothing).
  - Real-money trading requires BOTH live=True AND execute=True AND the env var
    SYSTEMATIC_TRADER_CONFIRM set to "I_UNDERSTAND_THE_RISK".
  - Per-instrument and gross exposure caps so a bug cannot lever you to the moon.

Setup on your own machine:
    pip install alpaca-py yfinance
    export ALPACA_API_KEY=...        # from alpaca.markets (paper keys to start)
    export ALPACA_SECRET_KEY=...
"""

import os
from dataclasses import dataclass

from .backtest import BTConfig, target_weights_now
from .strategies import Strategy


@dataclass
class LiveConfig:
    live: bool = False          # False -> paper endpoint
    execute: bool = False       # False -> dry run, no orders submitted
    max_gross: float = 1.0      # cap total |weight| summed across names (1.0 = no margin)
    max_per_name: float = 0.5   # cap |weight| on any single instrument
    crypto: bool = False        # set True if trading crypto symbols (BTC/USD etc.)
    whole_shares: bool = True   # round equity orders to whole shares (Alpaca blocks
                                # fractional short sales); set False for crypto


def _confirmed_for_real_money() -> bool:
    return os.environ.get("SYSTEMATIC_TRADER_CONFIRM") == "I_UNDERSTAND_THE_RISK"


def compute_orders(account_equity: float,
                   targets: dict[str, float],
                   current_positions: dict[str, float],
                   prices: dict[str, float],
                   lcfg: LiveConfig) -> list[dict]:
    """
    Turn target weights into a list of orders. Pure function: given the account state
    and target weights, returns the orders needed. Easy to unit-test, no broker needed.
    """
    # apply caps
    capped = {t: max(-lcfg.max_per_name, min(lcfg.max_per_name, w)) for t, w in targets.items()}
    gross = sum(abs(w) for w in capped.values())
    if gross > lcfg.max_gross and gross > 0:
        scale = lcfg.max_gross / gross
        capped = {t: w * scale for t, w in capped.items()}

    orders = []
    for t, w in capped.items():
        px = prices.get(t)
        if not px or px <= 0:
            continue
        target_dollars = w * account_equity
        target_shares = target_dollars / px
        current_shares = current_positions.get(t, 0.0)

        if not lcfg.whole_shares:
            # crypto path: fractional is fine, no short-sale restriction
            delta = target_shares - current_shares
            if abs(delta * px) >= 1.0 and delta != 0:
                orders.append(_order(t, delta, px, w))
            continue

        # equities: trade whole shares so we never submit a fractional short sale.
        target_int = float(int(target_shares))   # truncate toward zero

        if target_int < 0 and current_shares > 0:
            # long -> short. Do it in two legs so neither is a fractional short:
            #   1) sell the (possibly fractional) long down to flat  (allowed, ends >= 0)
            #   2) open the whole-share short from flat               (allowed, whole qty)
            if current_shares * px >= 1.0:
                orders.append(_order(t, -current_shares, px, w))   # flatten long
            orders.append(_order(t, target_int, px, w))            # open whole short
            continue

        delta = target_int - current_shares
        # if we are ending short, the sell must be whole shares; truncate the delta
        if target_int < 0:
            delta = float(int(delta))
        if abs(delta * px) >= 1.0 and delta != 0:
            orders.append(_order(t, delta, px, w))
    return orders


def _order(symbol: str, delta: float, px: float, w: float) -> dict:
    return {
        "symbol": symbol,
        "side": "buy" if delta > 0 else "sell",
        "qty": round(abs(delta), 6),
        "notional": round(abs(delta * px), 2),
        "target_weight": round(w, 4),
    }


class AlpacaTrader:
    """Thin wrapper around alpaca-py. Imported lazily so the rest of the repo runs
    without alpaca installed (e.g. in a sandbox)."""

    def __init__(self, lcfg: LiveConfig):
        self.lcfg = lcfg
        from alpaca.trading.client import TradingClient
        key = os.environ["ALPACA_API_KEY"]
        secret = os.environ["ALPACA_SECRET_KEY"]
        self.client = TradingClient(key, secret, paper=not lcfg.live)

    def equity(self) -> float:
        return float(self.client.get_account().equity)

    def positions(self) -> dict[str, float]:
        return {p.symbol: float(p.qty) for p in self.client.get_all_positions()}

    def submit(self, order: dict):
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        req = MarketOrderRequest(
            symbol=order["symbol"],
            qty=order["qty"],
            side=OrderSide.BUY if order["side"] == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC if self.lcfg.crypto else TimeInForce.DAY,
        )
        return self.client.submit_order(req)


def run_once(data, strat: Strategy, btcfg: BTConfig, lcfg: LiveConfig, prices: dict[str, float]):
    """
    One rebalance cycle. `data` is recent history per symbol (for signals), `prices`
    are latest marks (for sizing). Returns the order plan and, if permitted, submits it.
    """
    targets = target_weights_now(data, strat, btcfg)

    # guard rails on real-money trading
    if lcfg.live and lcfg.execute and not _confirmed_for_real_money():
        raise RuntimeError(
            "Refusing to trade real money: set SYSTEMATIC_TRADER_CONFIRM=I_UNDERSTAND_THE_RISK")

    mode = ("LIVE" if lcfg.live else "PAPER") + ("/EXECUTE" if lcfg.execute else "/DRY-RUN")
    print(f"[{mode}] target weights: " + ", ".join(f"{k} {v:+.2f}" for k, v in targets.items()))

    if not lcfg.execute:
        # dry run: size against a nominal $100k so you can eyeball the plan
        orders = compute_orders(100_000.0, targets, {}, prices, lcfg)
        print(f"[DRY-RUN] would place {len(orders)} order(s) on a $100k notional book:")
        for o in orders:
            print(f"   {o['side'].upper():4} {o['symbol']:8} ~${o['notional']:>10,.0f}  (target w={o['target_weight']:+.2f})")
        return orders

    trader = AlpacaTrader(lcfg)
    eq = trader.equity()
    pos = trader.positions()
    orders = compute_orders(eq, targets, pos, prices, lcfg)
    print(f"[{mode}] account equity ${eq:,.0f}; placing {len(orders)} order(s)")
    for o in orders:
        res = trader.submit(o)
        print(f"   submitted {o['side']} {o['qty']} {o['symbol']} -> id {getattr(res, 'id', '?')}")
    return orders
