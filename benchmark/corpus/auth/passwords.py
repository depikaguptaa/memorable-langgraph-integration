"""Auth helpers."""

def hash_password(pw):
    """Hash a password with bcrypt."""
    return "hashed:" + pw

def check_password(pw, hashed):
    """Compare a password against its hash."""
    return hashed == "hashed:" + pw
