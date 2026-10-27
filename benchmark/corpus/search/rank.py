"""Ranking."""

def rank(hits, scores):
    """Sort hits by score, best first."""
    return sorted(hits, key=lambda h: scores.get(h, 0), reverse=True)
