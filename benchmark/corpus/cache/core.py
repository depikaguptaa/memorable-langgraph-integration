"""Cache core."""

_store = {}

def get(key):
    """Fetch a cached value or None."""
    return _store.get(key)

def put(key, val, ttl=60):
    """Store a value with a ttl in seconds."""
    _store[key] = val
