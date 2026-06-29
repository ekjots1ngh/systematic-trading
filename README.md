# Systematic Trader

A small, honest, end to end systematic trading system. It defines trading rules,
backtests them on real daily data with realistic costs and no lookahead, sizes
positions the way a managed futures desk does (volatility targeting), and then runs
the *same* rules live or on paper through a real broker.

The point is not a flashy backtest. Flashy backtests are trivial to fake. The point is
a system whose results survive the things that usually kill them: execution lag, trading
costs, and out of sample testing.

## What it does, in one paragraph

It trades three uncorrelated markets (a stock index, bitcoin, and crude oil) using a
blend of two trend following signals and one mean reversion signal. Each market is
sized so it contributes an equal slice of risk, then the whole book is scaled to a 12%
annual volatility target. Positions decided on today's close are only traded tomorrow,
and every change in position pays a cost. The diversified portfolio is what matters:
the individual markets range from decent to a clear loser, but combined they produce a
Sharpe above 1 in window and around 0.8 out of sample, with the worst drawdown roughly
half of any single market.

## Results on real data (net of costs, no lookahead)

| Strategy                     | Sharpe | Vol   | Ann.Ret | MaxDD  | Buy&Hold Sharpe |
|------------------------------|:------:|:-----:|:-------:|:------:|:---------------:|
| SPX systematic (2000-2020)   |  0.41  | 12.1% |  +4.3%  | -29.5% |      0.28       |
| BTC systematic (2010-2026)   |  0.90  | 12.3% | +10.8%  | -27.3% |      1.19       |
| WTI systematic (1986-2026)   | -0.04  | 11.9% |  -1.2%  | -60.4% |      0.06       |
| **Portfolio (3 markets)**    | **1.15** | 10.2% | +11.9% | **-16.9%** |    1.07     |
| &nbsp;&nbsp;in sample (to 2016-06)    |  1.36  | 10.1% | +14.2% | -16.9% |  -  |
| &nbsp;&nbsp;out of sample (2016-06 on)|  0.79  | 10.0% |  +7.6% | -15.6% |  -  |

Read this honestly:

- One market (crude oil) loses money. Bitcoin standalone is beaten by simply holding it.
  This is not cherry picked. It is what real signals look like.
- The portfolio beats every single market on a risk adjusted basis and roughly halves the
  worst drawdown. That lift is the entire thesis of a systematic futures book:
  diversification across many markets, not a clever call on one.
- Out of sample Sharpe (0.79) is lower than in sample (1.36). It always is. The honest
  question is whether it survives at all, and here it does, with controlled drawdown.

Charts are written to `output/`:
- `equity_by_market.png` shows each market standalone (and that oil loses).
- `portfolio_vs_benchmark.png` compares the portfolio to buy and hold at *equal risk*.
- `drawdown.png` and `rolling_sharpe.png` show the pain and how the edge varies over time.

## How it is kept honest

These are the design choices a quant will actually check:

1. **One day execution lag.** Target weights are shifted forward a day, so a position
   decided at today's close earns tomorrow's return. No signal trades on information it
   could not have had. This single thing removes most fake performance.
2. **Costs on turnover.** Every change in position pays basis points of notional
   (commission plus a slippage estimate: 2 bps equities, 3 bps oil, 10 bps crypto).
3. **Causal volatility estimates.** Position sizing uses trailing exponentially weighted
   volatility, never full sample. The managed vol overlay scales the book using only its
   own past returns.
4. **Out of sample split.** Reported separately so degradation is visible, not hidden.
5. **A market that fails is left in.** Oil losing money stays in the table on purpose.

## Architecture

```
systematic-trader/
  config.yaml            universe, risk target, costs, strategy blend
  run_backtest.py        runs the full study, prints the table, writes charts
  run_live.py            computes today's targets and trades them (paper by default)
  data/                  real daily history (SPX, BTC, WTI) bundled as CSV
  src/
    indicators.py        causal technical indicators (EMA, ATR, RSI, z-score, momentum)
    strategies.py        TimeSeriesMomentum, MACrossover, MeanReversion, Ensemble
    backtest.py          vol targeted backtester with costs and the managed vol overlay
    metrics.py           Sharpe, Sortino, Calmar, drawdown, tail ratio, turnover
    data.py              cached loader plus a yfinance fetcher for live data
    live.py              Alpaca execution adapter with real money safety rails
```

The strategy layer never knows about dollars. It only outputs a direction and conviction
in [-1, 1]. The backtester and the live trader share the exact same sizing code, so what
you simulate is what you trade.

## Trust, control and audit

The live trader is wrapped in a control stack so a human stays in charge and can verify
everything after the fact. Every run, in order:

1. **Pre-trade health check.** Re-runs the backtest on the data it is about to trade and
   confirms it is sane: data is fresh, prices have no gaps, volatility has not exploded,
   and the strategy still produces a finite result. If any check fails, it does not trade
   and it alerts you. This cannot promise profits; it catches the data and model breakages
   that cause most live blow-ups.
2. **Kill switch.** If a file named `HALT` exists (create it, or run `--halt`), the system
   refuses to trade, full stop. `--resume` clears it. This is the instant stop button.
3. **Circuit breaker.** If account equity falls more than a set fraction (default 20%)
   below its high-water mark, trading halts automatically and you are alerted.
4. **Per-trade reasoning.** Every order carries a plain-English rationale built from the
   exact numbers that sized it: each sub-strategy's signal, the blended conviction, the
   trailing volatility, the vol-target scaler, the resulting weight, and the gap being
   closed. If the explanation and the order ever disagree, that is a visible bug.
5. **Approval gate.** Any single order above a notional limit (default $5,000) is not sent
   automatically. It is parked, you are alerted, and it executes only after `--approve`.
   Small routine trades flow through; only big ones need you to say yes.
6. **Notifications.** Email (SMTP) and/or SMS (Twilio) on every trade, every approval
   request, and every halt. Configured by environment variables so no secrets touch the
   repo. If neither is set up, alerts still go to the log.
7. **Audit ledger.** Every event (preflight result, equity, order, fill, approval, halt)
   is appended to `audit/ledger.jsonl`, an append-only record, plus a human-readable
   `audit/trader.log`. The complete life of any trade can be reconstructed from it.

Control commands:

```bash
python run_live.py --simulate     # run the whole stack offline against a mock broker
python run_live.py --status       # halt state, pending approvals, recent ledger events
python run_live.py --halt         # stop all trading now
python run_live.py --resume       # re-enable trading
python run_live.py --approve      # execute trades waiting for your sign-off
```

Try `--simulate` first: it exercises preflight, reasoning, the approval gate, the ledger
and alerts end to end without an account or any keys, and writes a real `audit/` folder
you can inspect.

## Run it

Backtest (works offline, uses the bundled real data):

```bash
pip install -r requirements.txt
python run_backtest.py
```

Live or paper trading on your own machine:

```bash
# get free keys at alpaca.markets (use PAPER keys first)
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...

python run_live.py                    # dry run: prints the order plan, trades nothing
python run_live.py --paper --execute  # trade the paper account for real
python run_live.py --live --execute   # REAL MONEY (see safety below)
```

Real money also requires an explicit confirmation:

```bash
export SYSTEMATIC_TRADER_CONFIRM=I_UNDERSTAND_THE_RISK
```

To point the backtest at the full futures style universe, replace the symbols in
`config.yaml` and use `data.fetch_yf([...])` instead of the bundled loader. Good
tradeable proxies: SPY, QQQ, IEF, TLT, GLD, USO, DBC, UUP, and BTC-USD / ETH-USD.

## Honest limitations

If asked where this would break, the right answers are:

- **Three markets is a toy universe.** Real managed futures books trade 50 to 100 plus
  markets, and most of the Sharpe comes from that breadth. More uncorrelated markets
  would make the out of sample Sharpe steadier.
- **The test window flatters trends.** 2010 to 2020 had strong, persistent trends
  (bitcoin above all). Trend following struggles in choppy, range bound, low volatility
  regimes. The mean reversion sleeve helps there but does not fully offset it.
- **Sharpe has wide error bars.** Over about ten years, the standard error on a Sharpe of
  1.15 is roughly 0.4. The honest claim is "Sharpe near 1, probably between 0.7 and 1.5",
  not "Sharpe is 1.15".
- **Fills are idealised.** Daily close to close with flat basis point costs ignores
  overnight gaps, market impact, and the fact that real fills happen at the open or
  intraday. A next open fill model with realistic slippage is the obvious upgrade.
- **Parameters were chosen by convention, not optimised.** That is deliberate (it avoids
  overfitting) but they were still picked by someone who has seen these series. Walk
  forward parameter selection would make this fully clean.

## What would make it institutional

More markets; carry and value signals alongside trend; a covariance based risk model
instead of equal risk budgeting; an explicit turnover penalty in the sizing; walk forward
parameter selection; and execution at the next open with a proper slippage model.

## Data

Daily history is real, pulled from public sources and cached in `data/`:
S&P 500 (2000 to 2020), Bitcoin USD (2010 to 2026), WTI crude (1986 to 2026).

---

*This is a research and engineering project. It is not financial advice. Live trading
risks real loss. No backtest, including this one, guarantees future returns.*
