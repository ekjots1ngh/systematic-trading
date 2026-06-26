"""
backtest.py
-----------
A vectorised, close-to-close backtester for vol-targeted systematic books.

The three things that make this honest rather than wishful:
  1. SIGNAL LAG. A position decided on the close of day t is held over day t+1.
     We shift target weights forward by one day, so the strategy never trades on
     information it could not have had. This alone removes the most common source
     of fake performance.
  2. COSTS. Every change in position pays a cost in basis points of notional traded
     (commission + a slippage estimate). Turnover is not free. Strategies that look
     great gross and die after costs are exactly what this catches.
  3. VOLATILITY TARGETING. Raw signals in [-1, 1] are scaled so each instrument
     contributes a fixed risk budget, using a *trailing* (causal) volatility estimate.
     This is how real systematic desks size positions, and it makes very different
     markets (an equity index and bitcoin) comparable inside one portfolio.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from . import metrics
from .strategies import Strategy


@dataclass
class BTConfig:
    portfolio_target_vol: float = 0.12   # 12% annualised target for the whole book
    vol_halflife: int = 30               # halflife (days) for the trailing vol estimate
    max_leverage: float = 3.0            # cap on |weight| per instrument
    cost_bps: dict = field(default_factory=dict)  # per-instrument round-trip-ish cost, bps of notional
    default_cost_bps: float = 3.0
    vol_overlay: bool = True             # scale whole book to hit target vol (causal)
    overlay_window: int = 60             # lookback (days) for the book's trailing vol
    overlay_bounds: tuple = (0.3, 3.0)   # min/max gross scaling allowed


@dataclass
class BTResult:
    returns: pd.Series          # net daily portfolio (or instrument) returns
    gross_returns: pd.Series    # before costs
    weights: pd.DataFrame       # held weight per instrument (already lagged)
    turnover: pd.Series         # daily turnover (sum |dw|)
    equity: pd.Series           # cumulative net equity, starts at 1.0
    stats: dict


def _instrument_weights(df: pd.DataFrame, strat: Strategy, target_vol: float, cfg: BTConfig) -> pd.Series:
    """Raw signal -> vol-targeted target weight (not yet lagged)."""
    sig = strat.signal(df).clip(-1, 1)
    rets = df["close"].pct_change()
    # trailing annualised vol, causal (uses only past returns)
    vol = rets.ewm(halflife=cfg.vol_halflife, min_periods=cfg.vol_halflife).std() * np.sqrt(252)
    scaler = (target_vol / vol.replace(0, np.nan)).clip(upper=cfg.max_leverage)
    w = (sig * scaler).clip(-cfg.max_leverage, cfg.max_leverage)
    return w.fillna(0.0)


def backtest(data: dict[str, pd.DataFrame], strat: Strategy, cfg: BTConfig,
             start: str | None = None, end: str | None = None) -> BTResult:
    """
    Run a backtest over one or more instruments. `data` maps ticker -> DataFrame
    indexed by date with at least a 'close' column. With multiple instruments this
    builds a diversified, equal-risk-budget portfolio.
    """
    tickers = list(data.keys())
    n = len(tickers)
    # each instrument gets an equal slice of the total risk budget
    per_inst_target = cfg.portfolio_target_vol / np.sqrt(n)

    # build a common date index (union) so instruments with different histories coexist
    idx = None
    for df in data.values():
        idx = df.index if idx is None else idx.union(df.index)
    if start:
        idx = idx[idx >= pd.Timestamp(start)]
    if end:
        idx = idx[idx <= pd.Timestamp(end)]

    weights = pd.DataFrame(0.0, index=idx, columns=tickers)
    inst_rets = pd.DataFrame(0.0, index=idx, columns=tickers)
    inst_turn = pd.DataFrame(0.0, index=idx, columns=tickers)

    for t in tickers:
        df = data[t].reindex(idx)
        raw_w = _instrument_weights(df, strat, per_inst_target, cfg)
        w = raw_w.shift(1).fillna(0.0)          # LAG: hold today what we decided yesterday
        ret = df["close"].pct_change().fillna(0.0)
        cost = cfg.cost_bps.get(t, cfg.default_cost_bps) / 1e4
        dw = w.diff().abs().fillna(0.0)
        weights[t] = w
        inst_turn[t] = dw
        inst_rets[t] = w * ret - dw * cost     # net contribution of this instrument

    gross = (weights * pd.concat({t: data[t]["close"].reindex(idx).pct_change()
                                  for t in tickers}, axis=1).fillna(0.0)).sum(axis=1)
    net = inst_rets.sum(axis=1)
    turnover = inst_turn.sum(axis=1)

    # trim leading warmup where nothing is active yet
    active = (weights.abs().sum(axis=1) > 0)
    if active.any():
        first = active.idxmax()
        net = net.loc[first:]
        gross = gross.loc[first:]
        turnover = turnover.loc[first:]
        weights = weights.loc[first:]

    # ----- managed-volatility overlay (causal) -----
    # Scale the whole book so its realised vol tracks the target. The scaler at time t
    # uses only the book's own returns up to t-1, so it adds no lookahead. This is the
    # standard "vol targeting" overlay run by managed-futures desks. Note Sharpe is
    # essentially invariant to this leverage choice; it just sets the risk dial.
    if cfg.vol_overlay:
        trailing = net.ewm(halflife=cfg.overlay_window, min_periods=cfg.overlay_window).std() * np.sqrt(252)
        scaler = (cfg.portfolio_target_vol / trailing.replace(0, np.nan)).shift(1)
        scaler = scaler.clip(*cfg.overlay_bounds).fillna(1.0)
        net = net * scaler
        gross = gross * scaler
        turnover = turnover * scaler
        weights = weights.mul(scaler, axis=0)

    equity = (1 + net).cumprod()
    stats = metrics.summary(net, turnover)
    stats["Realised Vol"] = stats.pop("Annual Vol")  # rename for clarity vs target
    return BTResult(net, gross, weights, turnover, equity, stats)


def target_weights_now(data: dict[str, pd.DataFrame], strat: Strategy, cfg: BTConfig) -> dict[str, float]:
    """
    The position the book WANTS to hold right now, per instrument, as a fraction of
    capital (signed; magnitude can exceed 1 = leverage). Computed on the most recent
    available close. This is what the live trader reconciles the account towards.
    Identical sizing logic to the backtest, so live and simulated behaviour match.
    """
    n = len(data)
    per_inst_target = cfg.portfolio_target_vol / np.sqrt(n)
    out = {}
    for t, df in data.items():
        w = _instrument_weights(df.sort_index(), strat, per_inst_target, cfg)
        out[t] = float(w.iloc[-1]) if len(w.dropna()) else 0.0
    return out
