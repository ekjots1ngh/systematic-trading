"""
notify.py
---------
Sends alerts when something the human should know about happens: a trade was placed,
a trade is waiting for approval, or the system halted itself.

Two channels, both optional and configured by environment variables. If neither is
configured the alert is still written to the log, so nothing is ever silently lost.

  EMAIL (SMTP)   set: ALERT_EMAIL_TO, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS
                 For Gmail, SMTP_PASS must be an App Password, not your normal one.

  SMS (Twilio)   set: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, ALERT_SMS_TO
                 Requires `pip install twilio`.
"""

import os
import smtplib
from email.mime.text import MIMEText

from .audit import get_logger

log = get_logger()


def _email_configured() -> bool:
    return all(os.environ.get(k) for k in
               ["ALERT_EMAIL_TO", "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS"])


def _sms_configured() -> bool:
    return all(os.environ.get(k) for k in
               ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM", "ALERT_SMS_TO"])


def _send_email(subject: str, body: str):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = os.environ["SMTP_USER"]
    msg["To"] = os.environ["ALERT_EMAIL_TO"]
    host = os.environ["SMTP_HOST"]
    port = int(os.environ["SMTP_PORT"])
    with smtplib.SMTP(host, port) as s:
        s.starttls()
        s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        s.send_message(msg)


def _send_sms(body: str):
    from twilio.rest import Client
    client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    client.messages.create(
        body=body,
        from_=os.environ["TWILIO_FROM"],
        to=os.environ["ALERT_SMS_TO"],
    )


def notify(subject: str, body: str):
    """Send an alert over whatever channels are configured; always log it too."""
    log.info("ALERT: %s | %s", subject, body.replace("\n", " "))
    if _email_configured():
        try:
            _send_email(subject, body)
            log.info("alert emailed to %s", os.environ["ALERT_EMAIL_TO"])
        except Exception as e:
            log.error("email alert failed: %s", e)
    if _sms_configured():
        try:
            _send_sms(f"{subject}\n{body}")
            log.info("alert SMS sent to %s", os.environ["ALERT_SMS_TO"])
        except Exception as e:
            log.error("SMS alert failed: %s", e)
    if not _email_configured() and not _sms_configured():
        log.info("(no email/SMS configured; alert only logged)")
