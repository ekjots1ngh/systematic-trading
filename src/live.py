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

from .backtest import BTConfig, target_weights_now, explain_targets
from .strategies import Strategy
from . import audit, notify, safety, reasoning
from .preflight import run_preflight


@dataclass
class LiveConfig:
    live: bool = False          # False -> paper endpoint
    execute: bool = False       # False -> dry run, no orders submitted
    max_gross: float = 1.0      # cap total |weight| summed across names (1.0 = no margin)
    max_per_name: float = 0.5   # cap |weight| on any single instrument
    crypto: bool = False        # set True if trading crypto symbols (BTC/USD etc.)
    whole_shares: bool = True   # round equity orders to whole shares (Alpaca blocks
                                # fractional short sales); set False for crypto
    approval_threshold: float = 5_000.0   # orders above this $ notional need human sign-off
    circuit_breaker_dd: float = 0.20      # halt if equity is >20% below its high-water mark
    skip_preflight: bool = False          # (testing only) bypass the pre-trade health check


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


class MockBroker:
    """An in-memory stand-in for a broker, so the whole control flow (preflight,
    reasoning, approval, ledger, alerts) can be exercised offline with no account.
    Used by run_live.py --simulate."""

    def __init__(self, equity: float = 100_000.0, positions: dict | None = None):
        self._equity = equity
        self._pos = dict(positions or {})
        self._n = 0

    def equity(self) -> float:
        return self._equity

    def positions(self) -> dict[str, float]:
        return dict(self._pos)

    def submit(self, order: dict):
        sign = 1 if order["side"] == "buy" else -1
        self._pos[order["symbol"]] = self._pos.get(order["symbol"], 0.0) + sign * order["qty"]
        self._n += 1
        return type("OrderAck", (), {"id": f"mock-{self._n:04d}"})


def _broker(lcfg: LiveConfig, simulate: bool):
    return MockBroker() if simulate else AlpacaTrader(lcfg)


def run_once(data, strat: Strategy, btcfg: BTConfig, lcfg: LiveConfig,
             prices: dict[str, float], simulate: bool = False):
    """
    One rebalance cycle with the full control stack:
      preflight health check -> kill switch -> compute targets + reasoning ->
      circuit breaker -> per-order approval gate -> submit -> ledger + alerts.
    Returns the list of orders that were submitted (or queued for approval).
    """
    log = audit.get_logger()
    ledger = audit.Ledger()
    log.info("=== run %s start (mode=%s) ===", ledger.run_id,
             ("LIVE" if lcfg.live else "PAPER") + ("/EXECUTE" if lcfg.execute else "/DRY-RUN"))

    # guard rails on real-money trading
    if lcfg.live and lcfg.execute and not _confirmed_for_real_money():
        raise RuntimeError(
            "Refusing to trade real money: set SYSTEMATIC_TRADER_CONFIRM=I_UNDERSTAND_THE_RISK")

    # --- kill switch ---
    if safety.is_halted():
        log.warning("HALT file present; refusing to trade.")
        ledger.record("halt", reason="HALT file present")
        notify.notify("Systematic Trader HALTED", "A HALT file is present; no trades placed.")
        return []

    # --- pre-trade health check ---
    if not lcfg.skip_preflight:
        report = run_preflight(data, strat, btcfg)
        for c in report.checks:
            ledger.record("preflight", name=c["name"], ok=c["ok"], detail=c["detail"])
        log.info("preflight:\n%s", report.summary())
        if not report.ok:
            failed = [c["name"] for c in report.checks if not c["ok"]]
            notify.notify("Systematic Trader: preflight FAILED",
                          "Trading blocked. Failed checks: " + ", ".join(failed))
            return []

    # --- targets with full reasoning ---
    detail = explain_targets(data, strat, btcfg)
    targets = {t: d["weight"] for t, d in detail.items()}

    if not lcfg.execute:
        orders = compute_orders(100_000.0, targets, {}, prices, lcfg)
        print(f"[DRY-RUN] {len(orders)} order(s) on a $100k notional book:")
        for o in orders:
            comps = detail[o["symbol"]]["components"]
            d = detail[o["symbol"]]
            print("  " + reasoning.describe_instrument(
                o["symbol"], comps, d["vol"], d["scaler"], d["weight"], 0.0, o))
        return orders

    # --- live/paper execution ---
    broker = _broker(lcfg, simulate)
    eq = broker.equity()
    pos = broker.positions()

    # --- circuit breaker ---
    tripped, dd = safety.circuit_breaker_tripped(eq, lcfg.circuit_breaker_dd)
    ledger.record("equity", equity=eq, drawdown=dd)
    if tripped:
        safety.set_halt(f"circuit breaker: drawdown {dd:.1%}")
        log.error("CIRCUIT BREAKER tripped at drawdown %.1f%%; halting.", dd * 100)
        notify.notify("Systematic Trader: CIRCUIT BREAKER",
                      f"Equity ${eq:,.0f}, drawdown {dd:.1%} breached "
                      f"{lcfg.circuit_breaker_dd:.0%}. Trading halted; HALT file written.")
        return []

    orders = compute_orders(eq, targets, pos, prices, lcfg)
    log.info("account equity $%s; %d candidate order(s)", f"{eq:,.0f}", len(orders))

    submitted, pending = [], []
    for o in orders:
        sym = o["symbol"]
        comps = detail[sym]["components"]
        rationale = reasoning.describe_instrument(
            sym, comps, detail[sym]["vol"], detail[sym]["scaler"],
            detail[sym]["weight"], pos.get(sym, 0.0), o)
        o["reason"] = rationale

        # --- approval gate for large orders ---
        if safety.needs_approval(o, lcfg.approval_threshold):
            pending.append(o)
            ledger.record("order_pending_approval", **o)
            log.warning("PENDING APPROVAL: %s", reasoning.one_line(sym, o, comps))
            continue

        ack = broker.submit(o)
        oid = getattr(ack, "id", "?")
        ledger.record("order_submitted", order_id=oid, **o)
        ledger.record("fill", order_id=oid, symbol=sym, side=o["side"], qty=o["qty"],
                      note="market order sent; fill confirmed asynchronously by broker")
        submitted.append(o)
        log.info("SUBMITTED %s -> %s | %s", reasoning.one_line(sym, o, comps), oid, rationale)
        notify.notify(f"Trade placed: {reasoning.one_line(sym, o, comps)}", rationale)

    if pending:
        safety.queue_for_approval(pending)
        names = ", ".join(reasoning.one_line(o["symbol"], o, detail[o["symbol"]]["components"])
                          for o in pending)
        notify.notify(f"{len(pending)} trade(s) need approval",
                      f"Above ${lcfg.approval_threshold:,.0f}. Approve with "
                      f"run_live.py --approve. Pending: {names}")

    log.info("=== run %s done: %d submitted, %d pending approval ===",
             ledger.run_id, len(submitted), len(pending))
    return submitted + pending


def approve_pending(strat, btcfg, lcfg: LiveConfig, prices: dict[str, float], simulate: bool = False):
    """Execute the orders a human parked for approval, then clear the queue."""
    log = audit.get_logger()
    ledger = audit.Ledger()
    pending = safety.load_pending()
    if not pending:
        log.info("no orders pending approval.")
        print("Nothing pending approval.")
        return []
    if safety.is_halted():
        log.warning("HALT present; not executing approved orders.")
        return []
    broker = _broker(lcfg, simulate)
    done = []
    for o in pending:
        ack = broker.submit(o)
        oid = getattr(ack, "id", "?")
        ledger.record("order_submitted", order_id=oid, approved=True, **o)
        log.info("APPROVED & SUBMITTED %s %s %s -> %s", o["side"], o["qty"], o["symbol"], oid)
        notify.notify(f"Approved trade placed: {o['side']} {o['qty']} {o['symbol']}",
                      o.get("reason", ""))
        done.append(o)
    safety.clear_pending()
    print(f"Executed {len(done)} approved order(s).")
    return done