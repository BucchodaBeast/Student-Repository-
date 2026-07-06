"""
mailer.py — outbound email for Launchpad, via Gmail SMTP.

Configured entirely through environment variables so nothing breaks if
they're not set yet:

    SMTP_USERNAME     the Gmail address to send from (e.g. launchpad@gmail.com)
    SMTP_PASSWORD     a Gmail App Password (NOT the regular account password —
                      generate one at https://myaccount.google.com/apppasswords,
                      requires 2-Step Verification to be turned on first)
    SMTP_FROM_EMAIL   optional, defaults to SMTP_USERNAME
    SMTP_FROM_NAME    optional, defaults to "Launchpad"
    SMTP_HOST         optional, defaults to smtp.gmail.com (override for a
                      different provider — SendGrid, Mailgun, etc. all speak
                      the same SMTP protocol)
    SMTP_PORT         optional, defaults to 587 (STARTTLS)

If SMTP_USERNAME/SMTP_PASSWORD aren't set, every send_email() call is a
harmless no-op that just logs to stdout — the same "off unless configured"
pattern already used for GROQ_API_KEY and Google OAuth.
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
SMTP_FROM_EMAIL = os.environ.get("SMTP_FROM_EMAIL", SMTP_USERNAME)
SMTP_FROM_NAME = os.environ.get("SMTP_FROM_NAME", "Launchpad")

EMAIL_ENABLED = bool(SMTP_USERNAME and SMTP_PASSWORD)


def email_enabled():
    return EMAIL_ENABLED


def send_email(to_addresses, subject, body_text):
    """Sends a plain-text email to one or more recipients via BCC (so
    recipients never see each other's addresses).

    Never raises — a notification email failing should never break the
    upload/review request that triggered it. Returns True/False so callers
    can log or flash a note if they want, but aren't required to check it.
    """
    to_addresses = [a for a in (to_addresses or []) if a]
    if not to_addresses:
        return False

    if not EMAIL_ENABLED:
        print(f"[mailer] SMTP not configured — would have sent '{subject}' to {len(to_addresses)} recipient(s)")
        return False

    try:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
        msg["To"] = SMTP_FROM_EMAIL  # BCC pattern: the visible "To" is just us
        msg["Bcc"] = ", ".join(to_addresses)
        msg.attach(MIMEText(body_text, "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM_EMAIL, to_addresses, msg.as_string())
        return True
    except Exception as e:
        print(f"[mailer] Failed to send '{subject}': {e}")
        return False
