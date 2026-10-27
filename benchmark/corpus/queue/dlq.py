"""DLQ."""

def park(dlq, item, reason):
    """Move a failed item to the dead-letter queue."""
    dlq.append((item, reason))
