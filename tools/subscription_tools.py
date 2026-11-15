from __future__ import annotations

import pandas as pd

from analytics.subscriptions import detect_subscriptions as _detect_subscriptions


def detect_subscriptions(df: pd.DataFrame) -> list[dict]:
    """Detect recurring subscription merchants from normalized transactions."""
    return _detect_subscriptions(df)
