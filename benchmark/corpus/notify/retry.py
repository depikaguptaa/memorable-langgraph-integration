"""Retry policy."""

def backoff(attempt):
    """Seconds to wait before a retry."""
    return 2 ** attempt
