"""Queue core."""

def push(q, item):
    """Append an item to the queue."""
    q.append(item)

def pop(q):
    """Remove and return the head item."""
    return q.pop(0)
