"""Token helpers."""

def make_token(user_id):
    """Mint a short-lived token for a user."""
    return f"tok-{user_id}"

def parse_token(tok):
    """Extract the user id from a token."""
    return tok.split("-", 1)[1]
