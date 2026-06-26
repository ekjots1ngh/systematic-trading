"""
data.py
-------
Loads price history. Two paths:

  load_cached(ticker)  -> reads the real daily data bundled in ./data (works anywhere,
                          including offline / sandboxed environments).
  fetch_yf(tickers)    -> pulls live history from Yahoo via yfinance when you run this
                          on your own machine. This is how you point the system at the
                          full futures-style universe (ES, GC, CL, ZN, 6E, ...) or any
                          liquid ETFs you actually want to trade.

Both return a DataFrame indexed by date with a 'close' column (and OHLCV when available),
so the rest of the system does not care where the prices came from.
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_cached(ticker: str) -> pd.DataFrame:
    """Load a bundled CSV (real historical data) by ticker name, e.g. 'SPX'."""
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if "close" not in df.columns:
        raise ValueError(f"{ticker}: expected a 'close' column")
    return df


def load_universe(tickers: list[str]) -> dict[str, pd.DataFrame]:
    return {t: load_cached(t) for t in tickers}


def fetch_yf(tickers: list[str], start: str = "2010-01-01", end: str | None = None
             ) -> dict[str, pd.DataFrame]:
    """
    Live data for your own machine. Requires `pip install yfinance` and internet.
    Example futures-proxy ETFs you can actually trade fractionally on most brokers:
        SPY (S&P 500), QQQ (Nasdaq), IEF/TLT (bonds), GLD (gold),
        USO (oil), UUP (USD), DBC (commodities), BTC-USD / ETH-USD (crypto).
    """
    import yfinance as yf
    out = {}
    for t in tickers:
        raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        if raw.empty:
            print(f"  warning: no data for {t}")
            continue
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
        out[t] = raw.rename(columns={"adj close": "close"})[["open", "high", "low", "close", "volume"]]
    return out
