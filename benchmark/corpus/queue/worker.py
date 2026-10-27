"""Queue workers."""

def work(q, fn):
    """Drain the queue applying fn to each item."""
    while q:
        fn(pop(q))
