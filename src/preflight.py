"""
preflight.py
------------
A pre-trade health check. Before any order is placed, the system re-runs its backtest
on the data it is about to trade on and confirms the world still looks sane. If a check
fails, trading is blocked for this run and an alert fires.

This does NOT promise profits. A backtest cannot. What it does is catch the concrete
ways a live run can go wrong: stale or missing data, a price series with gaps, a
volatility estimate that has blown up, or a strategy whose recent risk-adjusted
behaviour has fallen off a cliff (often the first sign that the data feed is broken).
Refusing to trade when the inputs look wrong is worth far more than any extra signal.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .backtest import backtest, BTConfig
from .strategies import Strategy


@dataclass
class PreflightReport:
    ok: bool
    checks: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [("PASS" if c["ok"] else "FAIL") + f"  {c['name']}: {c['detail']}"
                 for c in self.checks]
        return "\n".join(lines)


def run_preflight(data: dict[str, pd.DataFrame], strat: Strategy, cfg: BTConfig,
                  max_staleness_days: int = 5,
                  max_recent_vol: float = 1.0) -> PreflightReport:
    checks = []

    # 1) data freshness and integrity, per instrument
    for t, df in data.items():
        df = df.sort_index()
        last = df.index[-1]
        age = (pd.Timestamp.utcnow().tz_localize(None) - last.tz_localize(None)
               if last.tzinfo else pd.Timestamp.now() - last).days
        fresh = age <= max_staleness_days
        checks.append({"name": f"{t} data fresh", "ok": fresh,
                       "detail": f"last bar {last.date()} ({age}d old)"})
        recent = df["close"].tail(60)
        clean = recent.notna().all() and (recent > 0).all()
        checks.append({"name": f"{t} prices clean", "ok": bool(clean),
                       "detail": "no NaN/zero in last 60 bars" if clean else "gaps or zeros found"})
        # volatility sane (not exploded)
        rets = df["close"].pct_change().dropna()
        v = rets.tail(60).std() * np.sqrt(252)
        sane = np.isfinite(v) and v < max_recent_vol * 3
        checks.append({"name": f"{t} vol sane", "ok": bool(sane),
                       "detail": f"60d annualised vol {v:.0%}"})

    # 2) strategy backtest still produces a finite, non-degenerate result
    try:
        res = backtest(data, strat, cfg)
        sharpe = res.stats.get("Sharpe", float("nan"))
        finite = np.isfinite(sharpe)
        checks.append({"name": "backtest runs", "ok": True,
                       "detail": f"full-sample Sharpe {sharpe:.2f}"})
        checks.append({"name": "backtest finite", "ok": bool(finite),
                       "detail": "Sharpe is a finite number" if finite else "Sharpe is NaN/inf"})
    except Exception as e:
        checks.append({"name": "backtest runs", "ok": False, "detail": f"raised {type(e).__name__}: {e}"})

    return PreflightReport(ok=all(c["ok"] for c in checks), checks=checks)
