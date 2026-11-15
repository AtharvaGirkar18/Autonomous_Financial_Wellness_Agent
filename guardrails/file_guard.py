"""Upload guardrail: validates raw files and sanitises parsed transactions.

Threats covered
  * wrong / spoofed file types (extension vs. magic bytes), double extensions
  * oversized files, huge row counts, PDF page bombs
  * active PDF content (JavaScript, Launch, embedded files, OpenAction)
  * CSV / spreadsheet formula injection (=, +, -, @ prefixes)
  * indirect prompt injection hidden in merchant / description cells
  * HTML / script payloads in text cells
  * impossible values (NaN, negative / absurd amounts, dates in the far past/future)
  * identity spoofing: an uploaded ``user_id`` column is NEVER trusted
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePath

import pandas as pd

from guardrails.input_guard import normalize, scan_untrusted_text

MAX_FILES = 8
MAX_CSV_BYTES = 5 * 1024 * 1024
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 40
MAX_ROWS = 50_000
MAX_TEXT_LEN = 120
MAX_AMOUNT = 100_000_000.0  # ₹10 crore single transaction is treated as corrupt/hostile
ALLOWED_CATEGORIES = {
    "Food & Dining", "Groceries", "Transportation", "Fuel", "Shopping", "Healthcare", "Education",
    "Entertainment", "Bills & Utilities", "Rent", "Travel", "Insurance", "EMI / Debt", "Investments",
    "Subscriptions", "Salary / Income", "Transfer", "Cash Withdrawal", "Other",
}
TEXT_COLUMNS = ["transaction_id", "description", "merchant", "category", "subcategory", "payment_mode", "account_type"]
_PDF_ACTIVE = re.compile(rb"/(JavaScript|JS|Launch|EmbeddedFile|OpenAction|RichMedia|XFA)\b")
_HTML_TAG = re.compile(r"<[^>]{0,200}>")
_FORMULA_PREFIX = ("=", "+", "-", "@", "\t", "\r")
_SAFE_NAME = re.compile(r"^[\w\-. ()\[\]]{1,120}$")


@dataclass
class FileCheck:
    ok: bool
    kind: str = ""
    name: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_upload(name: str, data: bytes) -> FileCheck:
    """Reject unsafe files *before* any parser touches them."""
    check = FileCheck(ok=False, name=str(PurePath(str(name or "")).name))
    if not data:
        check.errors.append("File is empty.")
        return check
    base = check.name
    if not _SAFE_NAME.match(base) or ".." in base:
        check.errors.append("Unsafe file name (only letters, digits, spaces and - _ . ( ) are allowed).")
        return check
    parts = base.lower().split(".")
    ext = parts[-1] if len(parts) > 1 else ""
    if ext not in {"csv", "pdf"}:
        check.errors.append(f"Unsupported file type “.{ext}”. Upload a .csv or .pdf statement.")
        return check
    if len(parts) > 2 and parts[-2] in {"exe", "bat", "cmd", "js", "vbs", "ps1", "sh", "scr", "jar", "html", "php"}:
        check.errors.append("Double extension detected (possible disguised executable).")
        return check
    check.kind = ext

    if ext == "pdf":
        if len(data) > MAX_PDF_BYTES:
            check.errors.append(f"PDF exceeds {MAX_PDF_BYTES // (1024 * 1024)} MB limit.")
        if not data.lstrip()[:5] == b"%PDF-":
            check.errors.append("File content is not a PDF (magic bytes mismatch).")
        elif _PDF_ACTIVE.search(data):
            check.errors.append("PDF contains active content (JavaScript / launch / embedded file) and was rejected.")
    else:
        if len(data) > MAX_CSV_BYTES:
            check.errors.append(f"CSV exceeds {MAX_CSV_BYTES // (1024 * 1024)} MB limit.")
        if b"\x00" in data[:4096]:
            check.errors.append("File contains binary data and is not a text CSV.")
        elif data.lstrip()[:5] == b"%PDF-" or data[:2] == b"MZ" or data[:4] == b"PK\x03\x04":
            check.errors.append("File content does not match a CSV (looks like PDF/executable/archive).")
        else:
            head = data[:2048].decode("utf-8", "ignore").lower()
            if "<script" in head or "<html" in head:
                check.errors.append("CSV contains HTML/script content and was rejected.")
    check.ok = not check.errors
    return check


def _clean_text(value, report: dict) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value)
    cleaned = normalize(s)
    if cleaned != re.sub(r"\s+", " ", s).strip():
        report["control_chars_removed"] += 1
    stripped = _HTML_TAG.sub(" ", cleaned)
    if stripped != cleaned:
        report["html_stripped"] += 1
        cleaned = re.sub(r"\s+", " ", stripped).strip()
    if scan_untrusted_text(cleaned):
        report["injection_scrubbed"] += 1
        cleaned = "[REDACTED: instruction-like text]"
    if cleaned.startswith(_FORMULA_PREFIX):
        report["formula_neutralized"] += 1
        cleaned = "'" + cleaned
    if len(cleaned) > MAX_TEXT_LEN:
        cleaned = cleaned[:MAX_TEXT_LEN]
        report["truncated_text"] += 1
    return cleaned


def to_amount(value) -> float:
    """Parse ₹1,234.50 / 1,234.50 Dr / (500) / -500 into a float (NaN if impossible)."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return float("nan")
    neg = s.startswith("(") and s.endswith(")")
    s = re.sub(r"(?i)(₹|rs\.?|inr|cr|dr|\s)", "", s).strip("()")
    s = s.replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return float("nan")
    return -abs(v) if neg else v


def sanitize_dataframe(df: pd.DataFrame, user_id: str) -> tuple[pd.DataFrame, dict]:
    """Return a safe copy of *df* plus a human-readable sanitisation report."""
    report = {
        "rows_in": int(len(df)), "rows_out": 0, "rows_truncated": 0,
        "formula_neutralized": 0, "injection_scrubbed": 0, "html_stripped": 0,
        "control_chars_removed": 0, "truncated_text": 0, "invalid_amount_dropped": 0,
        "extreme_amount_dropped": 0, "invalid_date_dropped": 0, "identity_overridden": False,
        "unknown_category_reset": 0,
    }
    x = df.copy()
    x.columns = [re.sub(r"\s+", "_", normalize(str(c)).lower()) for c in x.columns]
    if len(x) > MAX_ROWS:
        report["rows_truncated"] = int(len(x) - MAX_ROWS)
        x = x.head(MAX_ROWS).copy()

    for col in TEXT_COLUMNS:
        if col in x.columns:
            x[col] = [_clean_text(v, report) for v in x[col].tolist()]

    if "category" in x.columns:
        bad = ~x["category"].isin(ALLOWED_CATEGORIES) & x["category"].ne("")
        report["unknown_category_reset"] = int(bad.sum())
        x.loc[bad, "category"] = "Other"
        x.loc[x["category"].eq(""), "category"] = pd.NA

    # Never trust an uploaded identity column (prevents overwriting another user's rows).
    if "user_id" in x.columns and (x["user_id"].astype(str) != str(user_id)).any():
        report["identity_overridden"] = True
    x["user_id"] = str(user_id)

    if "amount" in x.columns:
        x["amount"] = x["amount"].map(to_amount)
        bad_amt = x["amount"].isna()
        report["invalid_amount_dropped"] = int(bad_amt.sum())
        x = x[~bad_amt]
        x["amount"] = x["amount"].abs()
        extreme = x["amount"].gt(MAX_AMOUNT) | x["amount"].eq(0)
        report["extreme_amount_dropped"] = int(extreme.sum())
        x = x[~extreme]

    if "date" in x.columns:
        x["date"] = pd.to_datetime(x["date"], errors="coerce", dayfirst=True)
        lo, hi = pd.Timestamp("2000-01-01"), pd.Timestamp.today().normalize() + pd.Timedelta(days=366)
        bad_dt = x["date"].isna() | (x["date"] < lo) | (x["date"] > hi)
        report["invalid_date_dropped"] = int(bad_dt.sum())
        x = x[~bad_dt]
        x["date"] = x["date"].dt.strftime("%Y-%m-%d")

    for flag in ("is_recurring", "is_anomaly", "is_duplicate", "is_subscription"):
        if flag in x.columns:
            x[flag] = x[flag].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})

    report["rows_out"] = int(len(x))
    return x.reset_index(drop=True), report
