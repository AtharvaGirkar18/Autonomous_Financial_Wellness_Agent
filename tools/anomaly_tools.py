from __future__ import annotations
import pandas as pd
from ml.anomaly_detection import detect_anomalies


def detect_transaction_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    return detect_anomalies(df)
