"""
bot.py
------
A conversational, READ-ONLY assistant for the trading system, reachable over Telegram.
You message it in plain English ("why are we short gold?", "what's my biggest position?")
and it answers using the system's own data: the audit ledger (which stores the reasoning
behind every trade), the latest signals, and your live positions.

The "brain" is pluggable via the BOT_BACKEND environment variable:
  ollama    (default, FREE) runs a language model locally on your machine via Ollama.
            No API key, no per-question cost, nothing leaves your laptop.
  anthropic (paid) uses the Claude API for sharper answers. Needs ANTHROPIC_API_KEY.

SAFETY BY DESIGN:
  - Strictly read-only. It can explain, retrieve and summarise. It cannot place, cancel
    or approve trades, change config, or disable any safety control. Only the deterministic
    rules engine in run_live.py ever trades. If you ask the bot to act, it tells you the
    command to run yourself.
  - It only answers the one authorised Telegram chat id; messages from anyone else are ignored.

Run it (FREE local setup):
    1) install Ollama from https://ollama.com  and run:  ollama pull llama3.2
    2) pip install requests   (already installed)
    3) python bot.py
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import warnings
warnings.filterwarnings("ignore")

import requests

from src import audit
from src.backtest import BTConfig, explain_targets
from src import strategies

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = str(os.environ.get("TELEGRAM_CHAT_ID", ""))
BACKEND = os.environ.get("BOT_BACKEND", "ollama").lower()      # 'ollama' (free) or 'anthropic'
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")      # small, runs on modest laptops
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
ANTHROPIC_MODEL = os.environ.get("BOT_MODEL", "claude-haiku-4-5-20251001")
API = f"https://api.telegram.org/bot{TG_TOKEN}"
log = audit.get_logger("bot")

SYSTEM_PROMPT = """You are the read-only assistant for a personal systematic trading system.
You explain what the system is doing using ONLY the context provided to you in each message
(positions, latest signals, and the audit ledger of past trades with their recorded reasoning).

Rules:
- You are READ ONLY. You cannot place, cancel, or approve trades, change settings, or stop the
  system. If asked to do any of those, briefly explain you can't act, and tell the user the exact
  command to run themselves (e.g. `python run_live.py --halt` to stop trading, `--approve` to
  approve pending trades, `--status` to see state).
- Answer from the provided context. If the context does not contain the answer, say so plainly
  rather than guessing. Never invent positions, prices, or reasons.
- Be concise and concrete. This is a paper-trading research system, not financial advice.
- The reasoning for each trade in the ledger is authoritative; quote the relevant numbers."""

_cache = {"ts": 0, "signals": ""}


def _get_positions() -> str:
    if not (os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")):
        return "Positions: (Alpaca keys not set; cannot read live account.)"
    try:
        from alpaca.trading.client import TradingClient
        c = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
        acct = c.get_account(); pos = c.get_all_positions()
        if not pos:
            return f"Account equity ${float(acct.equity):,.0f}. No open positions."
        lines = [f"Account equity ${float(acct.equity):,.0f}. Positions:"]
        for p in pos:
            lines.append(f"  {p.symbol}: {p.qty} units, market value ${float(p.market_value):,.0f}, "
                         f"unrealised P/L ${float(p.unrealized_pl):,.2f}")
        return "\n".join(lines)
    except Exception as e:
        return f"Positions: (could not read account: {e})"


def _get_signals() -> str:
    if time.time() - _cache["ts"] < 900 and _cache["signals"]:
        return _cache["signals"]
    try:
        import datetime as dt
        from src.data import fetch_yf
        syms = ["SPY", "QQQ", "GLD", "TLT", "USO"]
        start = (dt.date.today() - dt.timedelta(days=800)).isoformat()
        data = fetch_yf(syms, start=start)
        if not data:
            from src import data as dmod
            data = dmod.load_universe(["SPX", "BTC", "WTI"])
        detail = explain_targets(data, strategies.build_default_ensemble(), BTConfig())
        lines = ["Latest signals and target weights:"]
        for t, d in detail.items():
            comps = ", ".join(f"{k} {v['signal']:+.2f}" for k, v in d["components"].items() if k != "combined")
            lines.append(f"  {t}: target weight {d['weight']:+.2%}; conviction {d['components']['combined']:+.2f} "
                         f"[{comps}]; trailing vol {d['vol']:.0%}")
        out = "\n".join(lines)
        _cache.update(ts=time.time(), signals=out)
        return out
    except Exception as e:
        return f"Signals: (could not compute: {e})"


def _get_ledger() -> str:
    rows = [r for r in audit.Ledger.read_all()
            if r["event"] in ("order_submitted", "order_pending_approval", "halt")][-15:]
    if not rows:
        return "Ledger: no trades recorded yet."
    lines = ["Recent trades from the audit ledger (most recent last):"]
    for r in rows:
        lines.append(f"  {r['ts'][:19]} {r['event']} {r.get('symbol','')} {r.get('side','')} "
                     f"{r.get('qty','')} | {r.get('reason','')}")
    return "\n".join(lines)



def handle_command(q: str):
    """Answer instantly from the ledger, no model needed. Returns text or None."""
    ql = q.lower().strip()
    if ql in ("/why", "why"):
        rows = [r for r in audit.Ledger.read_all() if r.get("reason")]
        return ("Most recent trade reasoning:\n\n" + rows[-1]["reason"]) if rows else \
               "No trades with recorded reasoning yet."
    if ql.startswith("/why ") or ql.startswith("why "):
        sym = q.split()[-1].upper()
        rows = [r for r in audit.Ledger.read_all()
                if r.get("symbol", "").upper() == sym and r.get("reason")]
        return (f"Latest {sym} trade reasoning:\n\n" + rows[-1]["reason"]) if rows else \
               f"No recorded reasoning found for {sym}."
    if ql in ("/status", "status"):
        from src import safety
        pend = safety.load_pending()
        return f"Halt: {'ON' if safety.is_halted() else 'off'}. Pending approval: {len(pend)} order(s)."
    return None

def build_context() -> str:
    return "\n\n".join([_get_positions(), _get_signals(), _get_ledger()])


def _ask_ollama(prompt: str) -> str:
    r = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
        "stream": False,
    }, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _ask_anthropic(prompt: str) -> str:
    from anthropic import Anthropic
    client = Anthropic()
    msg = client.messages.create(model=ANTHROPIC_MODEL, max_tokens=700,
                                 system=SYSTEM_PROMPT,
                                 messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def ask_llm(question: str) -> str:
    prompt = f"Here is the current system state:\n\n{build_context()}\n\nQuestion: {question}"
    return _ask_anthropic(prompt) if BACKEND == "anthropic" else _ask_ollama(prompt)


def send(text: str):
    requests.post(f"{API}/sendMessage", json={"chat_id": TG_CHAT, "text": text[:4000]}, timeout=15)


def _check_ollama() -> bool:
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json().get("models", [])
        names = [m.get("name", "") for m in tags]
        if not any(OLLAMA_MODEL in n for n in names):
            print(f"Ollama is running but model '{OLLAMA_MODEL}' isn't pulled. Run:  ollama pull {OLLAMA_MODEL}")
            return False
        return True
    except Exception:
        print(f"Can't reach Ollama at {OLLAMA_URL}. Install from https://ollama.com and make sure it's running.")
        return False


def main():
    for v in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]:
        if not os.environ.get(v):
            print(f"Missing environment variable: {v}"); return
    if BACKEND == "ollama" and not _check_ollama():
        return
    if BACKEND == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("BOT_BACKEND=anthropic but ANTHROPIC_API_KEY is not set."); return

    log.info("bot started; backend=%s; listening for chat %s", BACKEND, TG_CHAT)
    send(f"Assistant online (free local mode, model {OLLAMA_MODEL}). "
         f"Ask me about positions, signals, or any past trade. I'm read-only.")
    offset = None
    while True:
        try:
            r = requests.get(f"{API}/getUpdates", params={"timeout": 30, "offset": offset}, timeout=40).json()
            for u in r.get("result", []):
                offset = u["update_id"] + 1
                m = u.get("message") or {}
                if str(m.get("chat", {}).get("id")) != TG_CHAT:
                    continue
                q = m.get("text", "").strip()
                if not q:
                    continue
                log.info("Q: %s", q)
                cmd = handle_command(q)
                if cmd is not None:
                    send(cmd); continue
                send("Thinking...")
                try:
                    send(ask_llm(q))
                except Exception as e:
                    log.error("answer failed: %s", e)
                    send(f"Sorry, I hit an error answering that: {e}")
        except Exception as e:
            log.error("poll error: %s", e)
            time.sleep(5)


if __name__ == "__main__":
    main()
