"""
indicators.py
-------------
Technical indicators used to build trading signals.

Design rule that matters for a quant audience: every indicator here is *causal*.
The value at time t uses only information available at or before the close of day t.
There is no centering, no future-looking smoothing, no peeking. This is the single
most common way amateur backtests lie to themselves, so it is enforced by construction.
"""

import numpy as np
import pandas as pd


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return prices.rolling(window, min_periods=window).mean()


def ema(prices: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return prices.ewm(span=span, adjust=False, min_periods=span).mean()


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns."""
    return np.log(prices / prices.shift(1))


def realized_vol(returns: pd.Series, halflife: int = 30, annualize: bool = True) -> pd.Series:
    """
    EWMA realized volatility from daily returns.
    Uses an exponentially weighted estimate so recent regime changes are picked up
    quickly. Causal: only past returns enter the estimate at each point.
    """
    var = returns.ewm(halflife=halflife, min_periods=halflife).var()
    vol = np.sqrt(var)
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range - a volatility measure used for stops / position sizing."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder)."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def zscore(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling z-score of price vs its own moving average. The core mean-reversion
    feature: how many standard deviations is price stretched from its recent mean.
    """
    ma = prices.rolling(window, min_periods=window).mean()
    sd = prices.rolling(window, min_periods=window).std()
    return (prices - ma) / sd


def momentum(prices: pd.Series, lookback: int) -> pd.Series:
    """Total return over the past `lookback` trading days (the trend signal)."""
    return prices / prices.shift(lookback) - 1.0
