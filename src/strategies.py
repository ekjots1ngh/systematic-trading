"""
strategies.py
-------------
Each strategy maps a price history to a *target position* in roughly [-1, +1],
where +1 means "fully long this instrument", -1 "fully short", 0 "flat".
This raw signal is later scaled by the volatility-targeting layer in backtest.py,
so a strategy never has to think about how big a position should be in dollar terms;
it only expresses a *direction and conviction*.

Every signal is shifted by one day before it is traded (done in the backtester),
so a position decided on the close of day t earns the return from t to t+1.
No strategy can see today's return when deciding today's position.

Why these three:
  - Time-series momentum / trend: the single most robust premium in systematic
    futures. Markets that have gone up tend to keep going up over weeks-to-months.
    This is what CTAs like Winton, AHL and Cabestan-style desks are built on.
  - Moving-average crossover: a smoother, lower-turnover way to express the same
    trend idea; good for showing the signal is not a single fragile parameter.
  - Mean reversion: the opposite regime. Over very short horizons, sharp moves
    partially reverse ("buy low, sell high"). It pays in choppy, range-bound markets
    and tends to be negatively correlated with trend, which helps the blend.
"""

import numpy as np
import pandas as pd

from . import indicators as ind


class Strategy:
    name = "base"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """Return target position in [-1, 1] indexed like df. Override me."""
        raise NotImplementedError


class TimeSeriesMomentum(Strategy):
    """
    Trend following. Look at the risk-adjusted return over `lookback` days.
    If the market has trended up, go long; if down, go short. We use a tanh of the
    risk-adjusted momentum so conviction scales smoothly and saturates (no single
    blow-up day dominates the signal).
    """
    def __init__(self, lookback: int = 120, vol_window: int = 60):
        self.lookback = lookback
        self.vol_window = vol_window
        self.name = f"TSMOM_{lookback}"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        px = df["close"]
        rets = ind.log_returns(px)
        mom = ind.momentum(px, self.lookback)
        vol = rets.rolling(self.vol_window, min_periods=self.vol_window).std() * np.sqrt(252)
        risk_adj = mom / vol.replace(0, np.nan)
        return np.tanh(risk_adj).clip(-1, 1)


class MACrossover(Strategy):
    """
    Fast EMA vs slow EMA. Position is the normalised gap between them, squashed by
    tanh. Long when fast is above slow (uptrend), short when below.
    """
    def __init__(self, fast: int = 30, slow: int = 120):
        self.fast = fast
        self.slow = slow
        self.name = f"MAX_{fast}_{slow}"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        px = df["close"]
        f = ind.ema(px, self.fast)
        s = ind.ema(px, self.slow)
        # normalise the gap by recent price volatility so it is comparable across assets
        norm = (f - s) / px.rolling(self.slow, min_periods=self.slow).std()
        return np.tanh(norm).clip(-1, 1)


class MeanReversion(Strategy):
    """
    Buy low, sell high. Use the rolling z-score of price vs its own mean. When price
    is stretched far above the mean (high z), lean short; when stretched below, lean
    long. Signal is the negative z-score, capped. An optional RSI gate avoids fading
    the strongest trends. Short holding horizon by design.
    """
    def __init__(self, window: int = 15, cap: float = 2.0, use_rsi_gate: bool = True):
        self.window = window
        self.cap = cap
        self.use_rsi_gate = use_rsi_gate
        self.name = f"MREV_{window}"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        px = df["close"]
        z = ind.zscore(px, self.window)
        raw = (-z / self.cap).clip(-1, 1)  # fade the stretch
        if self.use_rsi_gate:
            r = ind.rsi(px, 14)
            # don't fight an extreme trend: damp the signal when RSI is very extreme
            gate = 1.0 - ((r - 50).abs() / 50).clip(0, 1) * 0.5
            raw = raw * gate
        return raw.clip(-1, 1)


class Ensemble(Strategy):
    """
    Blend several strategies by averaging their target positions. Diversifying across
    *signals* (not just instruments) is one of the cleanest ways to raise Sharpe,
    because the component edges are imperfectly correlated.
    """
    def __init__(self, members: list[Strategy], weights: list[float] | None = None):
        self.members = members
        self.weights = weights or [1.0 / len(members)] * len(members)
        self.name = "Ensemble"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        sigs = [m.signal(df) * w for m, w in zip(self.members, self.weights)]
        combined = pd.concat(sigs, axis=1).sum(axis=1)
        return combined.clip(-1, 1)

    def component_signals(self, df: pd.DataFrame) -> dict:
        """
        Latest signal from each member plus the blended result. This is what lets the
        system explain a position: which sub-strategies wanted it and how strongly.
        """
        out = {}
        for m, w in zip(self.members, self.weights):
            s = m.signal(df).dropna()
            out[m.name] = {"signal": float(s.iloc[-1]) if len(s) else 0.0, "weight": w}
        combined = self.signal(df).dropna()
        out["combined"] = float(combined.iloc[-1]) if len(combined) else 0.0
        return out


def build_default_ensemble() -> Ensemble:
    """The blend used by the default config: two trend horizons plus mean reversion."""
    return Ensemble(
        members=[
            TimeSeriesMomentum(lookback=120),
            MACrossover(fast=30, slow=120),
            MeanReversion(window=15),
        ],
        weights=[0.45, 0.30, 0.25],
    )
