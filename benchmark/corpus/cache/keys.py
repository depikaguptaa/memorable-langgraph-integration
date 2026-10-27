"""Cache keys."""

def user_key(user_id):
    """Cache key for a user record."""
    return f"user:{user_id}"
