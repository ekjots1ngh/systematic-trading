import os, smtplib, requests
from email.mime.text import MIMEText
from .audit import get_logger
log = get_logger()

def _telegram_configured():
    return all(os.environ.get(k) for k in ["TELEGRAM_BOT_TOKEN","TELEGRAM_CHAT_ID"])

def _send_telegram(subject, body):
    token=os.environ["TELEGRAM_BOT_TOKEN"]; chat=os.environ["TELEGRAM_CHAT_ID"]
    r=requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id":chat,"text":f"\U0001F4C8 {subject}\n{body}"}, timeout=15)
    r.raise_for_status()

def notify(subject, body):
    log.info("ALERT: %s | %s", subject, body.replace("\n"," "))
    if _telegram_configured():
        try: _send_telegram(subject, body); log.info("alert sent to Telegram")
        except Exception as e: log.error("Telegram alert failed: %s", e)
    else:
        log.info("(no Telegram configured; alert only logged)")
