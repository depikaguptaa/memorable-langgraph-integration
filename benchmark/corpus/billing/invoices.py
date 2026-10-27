"""Invoices."""

def make_invoice(customer, cents):
    """Build an invoice dict for a customer."""
    return {"customer": customer, "cents": cents}

def invoice_total(inv):
    """Total in cents for an invoice."""
    return inv["cents"]

def mark_paid(inv):
    """Flag an invoice as paid."""
    inv["paid"] = True
