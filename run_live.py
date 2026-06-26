"""
run_live.py
-----------
Compute today's target positions and (optionally) trade them on Alpaca.

Default behaviour is safe: PAPER account, DRY-RUN (prints the order plan, submits
nothing). You escalate deliberately:

    python run_live.py                      # dry run, prints what it would do
    python run_live.py --paper --execute    # trade the paper account for real
    python run_live.py --live --execute     # REAL MONEY (needs the confirm env var)

For real money you must also:
    export SYSTEMATIC_TRADER_CONFIRM=I_UNDERSTAND_THE_RISK

Signals are computed from recent history (yfinance by default). On your own machine
swap UNIVERSE for the symbols you actually want to trade.
"""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import warnings
warnings.filterwarnings("ignore")

from src import strategies
from src.backtest import BTConfig
from src.live import LiveConfig, run_once

# Symbols to trade live. Use tradeable tickers, not the cached index names.
# Liquid, fractionable proxies for a diversified futures-style book:
UNIVERSE = ["SPY", "QQQ", "GLD", "TLT", "USO"]   # add "BTC/USD" with --crypto


def recent_history(symbols, lookback_days=400):
    """Pull recent daily history for signal computation (yfinance, your machine)."""
    from src.data import fetch_yf
    import datetime as dt
    start = (dt.date.today() - dt.timedelta(days=lookback_days * 2)).isoformat()
    return fetch_yf(symbols, start=start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use the real-money endpoint")
    ap.add_argument("--paper", action="store_true", help="use the paper endpoint (default)")
    ap.add_argument("--execute", action="store_true", help="actually submit orders")
    ap.add_argument("--crypto", action="store_true", help="trading crypto symbols")
    ap.add_argument("--target-vol", type=float, default=0.12)
    args = ap.parse_args()

    btcfg = BTConfig(portfolio_target_vol=args.target_vol)
    lcfg = LiveConfig(live=args.live, execute=args.execute, crypto=args.crypto)
    ens = strategies.build_default_ensemble()

    if not args.execute:
        # dry run can work in any environment; if yfinance is unavailable, fall back
        # to the bundled data so you can still see the mechanism.
        try:
            data = recent_history(UNIVERSE)
            if not data or all(len(df) < 130 for df in data.values()):
                raise RuntimeError("insufficient live data returned")
            prices = {t: float(df["close"].iloc[-1]) for t, df in data.items()}
        except Exception as e:
            print(f"(live data unavailable: {e}; using bundled sample data for the dry run)")
            from src import data as dmod
            data = dmod.load_universe(["SPX", "BTC", "WTI"])
            prices = {t: float(df["close"].iloc[-1]) for t, df in data.items()}
    else:
        data = recent_history(["BTC/USD"] if args.crypto else UNIVERSE)
        prices = {t: float(df["close"].iloc[-1]) for t, df in data.items()}

    run_once(data, ens, btcfg, lcfg, prices)


if __name__ == "__main__":
    main()
