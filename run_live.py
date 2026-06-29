"""
run_live.py
-----------
Compute today's target positions, run the safety stack, and (optionally) trade them.

Default behaviour is safe: PAPER account, DRY-RUN (prints the plan with reasoning,
submits nothing). You escalate deliberately:

    python run_live.py                      # dry run, prints plan + per-trade reasoning
    python run_live.py --simulate           # full control flow against a mock broker (offline)
    python run_live.py --paper --execute    # trade the paper account for real
    python run_live.py --live --execute     # REAL MONEY (needs the confirm env var)

Human controls:
    python run_live.py --halt               # stop all trading immediately (writes HALT file)
    python run_live.py --resume             # remove the halt
    python run_live.py --approve            # execute trades parked for human sign-off
    python run_live.py --status             # show halt state, pending approvals, recent trades

For real money you must also:  export SYSTEMATIC_TRADER_CONFIRM=I_UNDERSTAND_THE_RISK
"""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import warnings
warnings.filterwarnings("ignore")

from src import strategies, safety, audit
from src.backtest import BTConfig
from src.live import LiveConfig, run_once, approve_pending

UNIVERSE = ["SPY", "QQQ", "GLD", "TLT", "USO"]   # add "BTC/USD" with --crypto


def recent_history(symbols, lookback_days=400):
    from src.data import fetch_yf
    import datetime as dt
    start = (dt.date.today() - dt.timedelta(days=lookback_days * 2)).isoformat()
    return fetch_yf(symbols, start=start)


def load_data(args):
    """Get history + latest prices. Falls back to bundled data when offline/simulating."""
    if args.simulate or not args.execute:
        try:
            data = recent_history(UNIVERSE)
            if not data or all(len(df) < 130 for df in data.values()):
                raise RuntimeError("insufficient live data")
        except Exception as e:
            print(f"(live data unavailable: {e}; using bundled sample data)")
            from src import data as dmod
            data = dmod.load_universe(["SPX", "BTC", "WTI"])
    else:
        data = recent_history(["BTC/USD"] if args.crypto else UNIVERSE)
    prices = {t: float(df["close"].iloc[-1]) for t, df in data.items()}
    return data, prices


def show_status():
    print("HALT:", "ON (trading blocked)" if safety.is_halted() else "off")
    pending = safety.load_pending()
    print(f"Pending approval: {len(pending)} order(s)")
    for o in pending:
        print(f"   {o['side'].upper()} {o['qty']} {o['symbol']}  ~${o.get('notional',0):,.0f}")
    hist = audit.Ledger.history(limit=10)
    print(f"Recent ledger events ({len(hist)} shown):")
    for h in hist:
        print(f"   {h['ts'][:19]}  {h['event']:22} {h.get('symbol','')} "
              f"{h.get('side','')} {h.get('qty','')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use the real-money endpoint")
    ap.add_argument("--paper", action="store_true", help="use the paper endpoint (default)")
    ap.add_argument("--execute", action="store_true", help="actually submit orders")
    ap.add_argument("--simulate", action="store_true", help="run full flow against a mock broker (offline)")
    ap.add_argument("--crypto", action="store_true", help="trading crypto symbols")
    ap.add_argument("--approve", action="store_true", help="execute orders pending approval")
    ap.add_argument("--halt", action="store_true", help="stop all trading (write HALT file)")
    ap.add_argument("--resume", action="store_true", help="remove the HALT file")
    ap.add_argument("--status", action="store_true", help="show controls and recent activity")
    ap.add_argument("--target-vol", type=float, default=0.12)
    ap.add_argument("--approval-threshold", type=float, default=5000.0,
                    help="orders above this $ notional require approval")
    args = ap.parse_args()

    if args.halt:
        safety.set_halt("manual via --halt")
        print("HALT set. All trading is blocked until you run --resume."); return
    if args.resume:
        safety.clear_halt()
        print("HALT cleared. Trading re-enabled."); return
    if args.status:
        show_status(); return

    btcfg = BTConfig(portfolio_target_vol=args.target_vol)
    lcfg = LiveConfig(live=args.live, execute=args.execute or args.simulate,
                      crypto=args.crypto, whole_shares=not args.crypto,
                      approval_threshold=args.approval_threshold)
    ens = strategies.build_default_ensemble()
    data, prices = load_data(args)

    if args.approve:
        approve_pending(ens, btcfg, lcfg, prices, simulate=args.simulate)
        return

    run_once(data, ens, btcfg, lcfg, prices, simulate=args.simulate)


if __name__ == "__main__":
    main()
