import pandas as pd
from agents.base import run_agent
from analytics.spending import build_spending_analysis


def spending_node(state: dict) -> dict:
    def _run(s):
        df = pd.DataFrame(s.get("transactions", []))
        if not df.empty and ("category" not in df.columns or df["category"].isna().all()):
            from config.settings import DATA_DIR
            from ml.categorization import HybridCategorizer
            ref = DATA_DIR / "synthetic_transactions.csv"
            training_df = pd.read_csv(ref) if ref.exists() else df
            cat = HybridCategorizer()
            cat.fit(training_df)
            df = cat.transform(df)
            df["category"] = df["predicted_category"]
            s["transactions"] = df.to_dict(orient="records")
        return {"analytics": build_spending_analysis(df)}
    return run_agent(state, "Spending Analysis Agent", _run, ["calculate_monthly_spending", "calculate_category_spending", "calculate_savings_rate"])
