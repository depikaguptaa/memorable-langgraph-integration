"""Billing plans."""

def plan_price(plan):
    """Monthly price in cents for a plan name."""
    return {"free": 0, "pro": 2000}[plan]
