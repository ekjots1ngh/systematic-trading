"""
metrics.py
----------
Performance statistics computed from a daily return series.

These are the numbers a systematic desk actually looks at. Sharpe is the headline,
but on its own it is easy to game, so we report the full panel: risk-adjusted return,
tail behaviour, drawdown, and how hard the strategy has to trade to earn its money.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _clean(returns: pd.Series) -> pd.Series:
    return returns.dropna()


def annual_return(returns: pd.Series) -> float:
    r = _clean(returns)
    if len(r) == 0:
        return np.nan
    # geometric (CAGR-style) annualisation
    total = (1 + r).prod()
    years = len(r) / TRADING_DAYS
    return total ** (1 / years) - 1 if years > 0 else np.nan


def annual_vol(returns: pd.Series) -> float:
    return _clean(returns).std() * np.sqrt(TRADING_DAYS)


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio. rf is an annual risk-free rate."""
    r = _clean(returns)
    if r.std() == 0 or len(r) == 0:
        return np.nan
    excess = r - rf / TRADING_DAYS
    return excess.mean() / r.std() * np.sqrt(TRADING_DAYS)


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    """Like Sharpe but penalises only downside volatility."""
    r = _clean(returns)
    downside = r[r < 0].std()
    if downside == 0 or np.isnan(downside):
        return np.nan
    excess = r - rf / TRADING_DAYS
    return excess.mean() / downside * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the cumulative equity curve."""
    r = _clean(returns)
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd.min()


def calmar(returns: pd.Series) -> float:
    """Annual return divided by the magnitude of max drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return annual_return(returns) / abs(mdd)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of active days that were positive."""
    r = _clean(returns)
    r = r[r != 0]
    if len(r) == 0:
        return np.nan
    return (r > 0).mean()


def skew(returns: pd.Series) -> float:
    return _clean(returns).skew()


def tail_ratio(returns: pd.Series) -> float:
    """95th percentile gain divided by magnitude of 5th percentile loss. >1 is good."""
    r = _clean(returns)
    if len(r) == 0:
        return np.nan
    p5 = np.percentile(r, 5)
    p95 = np.percentile(r, 95)
    if p5 == 0:
        return np.nan
    return abs(p95 / p5)


def summary(returns: pd.Series, turnover: pd.Series | None = None) -> dict:
    """Full performance panel as a dict."""
    out = {
        "Annual Return": annual_return(returns),
        "Annual Vol": annual_vol(returns),
        "Sharpe": sharpe(returns),
        "Sortino": sortino(returns),
        "Calmar": calmar(returns),
        "Max Drawdown": max_drawdown(returns),
        "Hit Rate": hit_rate(returns),
        "Skew": skew(returns),
        "Tail Ratio": tail_ratio(returns),
    }
    if turnover is not None:
        # average daily turnover, annualised, expressed as multiples of capital traded
        out["Ann. Turnover"] = turnover.dropna().mean() * TRADING_DAYS
    return out


def format_summary(stats: dict) -> str:
    """Pretty-print a stats dict."""
    pct_keys = {"Annual Return", "Annual Vol", "Max Drawdown", "Hit Rate"}
    lines = []
    for k, v in stats.items():
        if isinstance(v, float) and np.isnan(v):
            val = "  n/a"
        elif k in pct_keys:
            val = f"{v:+.1%}"
        elif k == "Ann. Turnover":
            val = f"{v:.1f}x"
        else:
            val = f"{v:+.2f}"
        lines.append(f"  {k:<16} {val:>9}")
    return "\n".join(lines)
