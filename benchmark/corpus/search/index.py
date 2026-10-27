"""Search index."""

def add_doc(index, doc_id, text):
    """Add a document to the in-memory index."""
    index[doc_id] = text

def search(index, q):
    """Return doc ids containing the query."""
    return [d for d, t in index.items() if q in t]
