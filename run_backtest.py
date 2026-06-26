"""
run_backtest.py
---------------
Runs the full study on real daily data and writes plots + a results table.

What it demonstrates, in order:
  1. The same systematic rules applied to three different markets (equity index,
     crypto, oil) standalone.
  2. A diversified, vol-targeted portfolio of all three - showing the Sharpe lift
     that diversification across markets buys you (the whole point of a CTA book).
  3. An out-of-sample split: parameters/intuition come from the early period, and
     we report performance on a later period the strategy never "saw". If the edge
     only existed in-sample, this is where it dies. It mostly doesn't here.
  4. A buy-and-hold benchmark, because a strategy is only interesting if it beats
     just owning the asset on a risk-adjusted basis.

Run:  python run_backtest.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import data, strategies
from src.backtest import backtest, BTConfig
from src import metrics

UNIVERSE = ["SPX", "BTC", "WTI"]
COSTS = {"SPX": 2.0, "WTI": 3.0, "BTC": 10.0}   # bps of notional traded; crypto is dearer
OVERLAP = ("2010-08-01", "2020-04-17")          # window where all three trade together
OOS_SPLIT = "2016-06-01"                         # train before, test after

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})


def buy_hold_stats(df, start, end):
    px = df["close"].loc[start:end]
    r = px.pct_change().dropna()
    return metrics.summary(r)


def main():
    uni = data.load_universe(UNIVERSE)
    ens = strategies.build_default_ensemble()
    cfg = BTConfig(cost_bps=COSTS, portfolio_target_vol=0.12)

    rows = []

    # 1) standalone, each market over its full history
    standalone = {}
    for t in UNIVERSE:
        res = backtest({t: uni[t]}, ens, cfg)
        standalone[t] = res
        s = res.stats
        bh = buy_hold_stats(uni[t], res.returns.index.min(), res.returns.index.max())
        rows.append([f"{t} systematic", f"{s['Sharpe']:.2f}", f"{s['Realised Vol']:.1%}",
                     f"{s['Annual Return']:.1%}", f"{s['Max Drawdown']:.1%}",
                     f"{bh['Sharpe']:.2f}"])

    # 2) diversified portfolio over the overlap window
    port = backtest(uni, ens, cfg, start=OVERLAP[0], end=OVERLAP[1])
    s = port.stats
    # equal-weight buy&hold of the three over the overlap, as a benchmark
    eqw = pd.concat({t: uni[t]["close"].pct_change() for t in UNIVERSE}, axis=1)\
            .loc[OVERLAP[0]:OVERLAP[1]].mean(axis=1).dropna()
    bh_sharpe = metrics.sharpe(eqw)
    rows.append(["PORTFOLIO (3 mkts)", f"{s['Sharpe']:.2f}", f"{s['Realised Vol']:.1%}",
                 f"{s['Annual Return']:.1%}", f"{s['Max Drawdown']:.1%}", f"{bh_sharpe:.2f}"])

    # 3) out-of-sample split on the portfolio
    ins = backtest(uni, ens, cfg, start=OVERLAP[0], end=OOS_SPLIT)
    oos = backtest(uni, ens, cfg, start=OOS_SPLIT, end=OVERLAP[1])
    rows.append([f"  in-sample (->{OOS_SPLIT[:7]})", f"{ins.stats['Sharpe']:.2f}",
                 f"{ins.stats['Realised Vol']:.1%}", f"{ins.stats['Annual Return']:.1%}",
                 f"{ins.stats['Max Drawdown']:.1%}", "-"])
    rows.append([f"  out-of-sample ({OOS_SPLIT[:7]}->)", f"{oos.stats['Sharpe']:.2f}",
                 f"{oos.stats['Realised Vol']:.1%}", f"{oos.stats['Annual Return']:.1%}",
                 f"{oos.stats['Max Drawdown']:.1%}", "-"])

    # ---- print table ----
    hdr = ["Strategy", "Sharpe", "Vol", "Ann.Ret", "MaxDD", "B&H Sharpe"]
    widths = [26, 7, 7, 8, 8, 10]
    line = "  ".join(h.ljust(w) for h, w in zip(hdr, widths))
    print("\n" + line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    print("\nNotes: B&H = buy & hold the asset over the same window. Costs included.")
    print("Sharpe is leverage-invariant; the vol overlay just sets the risk dial.\n")

    # ---- plots ----
    os.makedirs("output", exist_ok=True)

    # equity curves (standalone, log scale)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for t in UNIVERSE:
        ax.plot(standalone[t].equity.index, standalone[t].equity.values, label=f"{t} systematic", lw=1.2)
    ax.set_yscale("log"); ax.set_title("Systematic strategy equity by market (log scale, net of costs)")
    ax.set_ylabel("Growth of $1"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("output/equity_by_market.png"); plt.close(fig)

    # portfolio vs buy&hold over overlap, RISK-MATCHED (both levered to 12% vol) so the
    # comparison is apples-to-apples. At equal risk, the higher-Sharpe book ends higher.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(port.equity.index, port.equity.values, label="Diversified systematic (12% vol)", lw=1.6, color="#1b1b1b")
    eqw_scaled = eqw * (0.12 / (eqw.std() * np.sqrt(252)))
    bh_eq = (1 + eqw_scaled).cumprod()
    ax.plot(bh_eq.index, bh_eq.values, label="Buy & hold, risk-matched to 12% vol", lw=1.2, color="#c0392b", alpha=0.85)
    ax.set_yscale("log")
    ax.axvline(pd.Timestamp(OOS_SPLIT), ls="--", color="grey", lw=1)
    ax.text(pd.Timestamp(OOS_SPLIT), ax.get_ylim()[1]*0.7, " out-of-sample ->", fontsize=8, color="grey")
    ax.set_title(f"At equal risk: systematic Sharpe {s['Sharpe']:.2f} vs buy & hold {bh_sharpe:.2f}")
    ax.set_ylabel("Growth of $1 (log)"); ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig("output/portfolio_vs_benchmark.png"); plt.close(fig)

    # drawdown of the portfolio
    eq = port.equity
    dd = eq / eq.cummax() - 1
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.fill_between(dd.index, dd.values, 0, color="#c0392b", alpha=0.4)
    ax.set_title("Portfolio drawdown"); ax.set_ylabel("Drawdown")
    fig.tight_layout(); fig.savefig("output/drawdown.png"); plt.close(fig)

    # rolling 1y Sharpe of the portfolio
    roll = port.returns.rolling(252)
    rsharpe = (roll.mean() / roll.std()) * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(rsharpe.index, rsharpe.values, lw=1.1, color="#2c3e50")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("Portfolio rolling 1-year Sharpe"); ax.set_ylabel("Sharpe")
    fig.tight_layout(); fig.savefig("output/rolling_sharpe.png"); plt.close(fig)

    print("Saved plots to output/: equity_by_market.png, portfolio_vs_benchmark.png, "
          "drawdown.png, rolling_sharpe.png")

    return port


if __name__ == "__main__":
    main()
