"""Notify channels."""

def send_email(to, subject, body):
    """Queue an email notification."""
    return {"to": to, "subject": subject}

def send_sms(to, body):
    """Queue an SMS notification."""
    return {"to": to, "body": body}
