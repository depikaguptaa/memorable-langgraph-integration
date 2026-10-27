"""Notify templates."""

def render(name, ctx):
    """Render a named template with context."""
    return name.format(**ctx)
