from __future__ import annotations

import pandas as pd

from agents.base import run_agent
from ml.categorization import HybridCategorizer
from config.settings import DATA_DIR


def categorization_node(state: dict) -> dict:
    def _run(s):
        df = pd.DataFrame(s["transactions"])
        training_df = df
        if "category" not in df.columns or df["category"].isna().all():
            reference = DATA_DIR / "synthetic_transactions.csv"
            if reference.exists():
                training_df = pd.read_csv(reference)
        cat = HybridCategorizer()
        cat.fit(training_df)
        out = cat.transform(df)
        out["category"] = out["predicted_category"]
        out["subcategory"] = out["predicted_subcategory"]
        return {
            "transactions": out.to_dict(orient="records"),
            "categorization_summary": {
                "method_counts": out["category_method"].value_counts().to_dict(),
                "avg_confidence": round(float(out["category_confidence"].mean()), 4),
            },
        }
    return run_agent(state, "Categorization Agent", _run, ["classify_transaction", "calculate_category_confidence"])
