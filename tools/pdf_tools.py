from __future__ import annotations

import re

import fitz
import pandas as pd


LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s*\|\s*(.*?)\s*\|\s*(DEBIT|CREDIT)\s*\|\s*"
    r"(?:₹|Rs\.?\s*)?([\d,]+(?:\.\d{1,2})?)\s*\|\s*([A-Z_]+)$",
    flags=re.IGNORECASE,
)


def extract_transactions_from_pdf(pdf_bytes: bytes, user_id: str = "UPLOADED") -> pd.DataFrame:
    """Parse synthetic/demo statement PDFs and return normalized transaction rows."""
    rows: list[dict] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = "\n".join(page.get_text("text") for page in doc)

    for line in text.splitlines():
        line = line.strip()
        match = LINE_RE.match(line)
        if not match:
            continue
        dt, description, tx_type, amount, mode = match.groups()
        merchant = description.strip() if description.strip() else "Unknown"
        desc_lower = description.strip().lower()
        is_rec = any(kw in desc_lower for kw in ("emi", "rent", "loan", "broadband", "electricity", "bill", "subscription", "premium", "insurance", "sip")) or mode.upper() == "NACH"
        rows.append({
            "transaction_id": f"PDF-{len(rows) + 1:05d}",
            "user_id": user_id,
            "date": dt,
            "description": description.strip(),
            "merchant": merchant.title(),
            "amount": float(amount.replace(",", "")),
            "transaction_type": tx_type.upper(),
            "account_type": "BANK",
            "payment_mode": mode.upper(),
            "is_recurring": is_rec,
        })

    if not rows:
        raise ValueError("No supported transaction rows could be parsed from the PDF.")
    return pd.DataFrame(rows)
