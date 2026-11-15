from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ml.anomaly_detection import detect_anomalies, evaluate_anomalies
from ml.categorization import HybridCategorizer, MLCategorizer, RuleBasedCategorizer, evaluate_categorization
from ml.forecasting import evaluate_forecast, moving_average_forecast, naive_monthly_forecast
from ml.agent_evaluation import run_agent_evaluation


def _categorization_experiment(df: pd.DataFrame) -> dict:
    train, test = train_test_split(df, test_size=0.25, random_state=42, stratify=df["category"])
    methods = {"rule_based": RuleBasedCategorizer(), "ml": MLCategorizer(), "hybrid": HybridCategorizer()}
    results = {}
    for name, model in methods.items():
        model.fit(train)
        pred = model.transform(test)
        results[name] = evaluate_categorization(test["category"], pred["predicted_category"])
    return results


def _anomaly_experiment(df: pd.DataFrame) -> dict:
    detected = detect_anomalies(df)
    anomaly_target = df["is_anomaly"].astype(bool).to_numpy()
    anomaly_pred = detected["is_detected_anomaly"].astype(bool).to_numpy()
    duplicate_target = df["is_duplicate"].astype(bool).to_numpy()
    duplicate_pred = detected["duplicate_candidate"].astype(bool).to_numpy()
    return {
        "anomaly": evaluate_anomalies(anomaly_target, anomaly_pred),
        "duplicate": evaluate_anomalies(duplicate_target, duplicate_pred),
        "detected_anomaly_count": int(anomaly_pred.sum()),
        "detected_duplicate_count": int(duplicate_pred.sum()),
    }


def _forecast_experiment(df: pd.DataFrame) -> dict:
    """Evaluate one-step monthly expense forecasting per synthetic user, then pool predictions."""
    naive_true, naive_pred = [], []
    ma_true, ma_pred = [], []
    for _, user_df in df.groupby("user_id"):
        debit = user_df[user_df["transaction_type"].eq("DEBIT")].copy()
        debit["month"] = pd.to_datetime(debit["date"]).dt.to_period("M").astype(str)
        monthly = debit.groupby("month")["amount"].sum().sort_index()
        if len(monthly) < 5:
            continue
        train = monthly.iloc[:-1]
        actual = float(monthly.iloc[-1])
        naive_true.append(actual)
        naive_pred.append(float(naive_monthly_forecast(train, 1)[0]))
        ma_true.append(actual)
        ma_pred.append(float(moving_average_forecast(train, 1, 3)[0]))

    if not naive_true:
        empty = {"mae": None, "rmse": None, "mape": None}
        return {"naive": empty, "moving_average": empty}
    return {
        "naive": evaluate_forecast(naive_true, naive_pred),
        "moving_average": evaluate_forecast(ma_true, ma_pred),
    }


def _write_artifacts(result: dict, output_dir: str | None) -> None:
    if not output_dir:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation_results.json").write_text(json.dumps(result, indent=2))

    cat_rows = []
    for method, metrics in result["categorization"].items():
        cat_rows.append({"method": method, **metrics})
    pd.DataFrame(cat_rows).to_csv(out / "categorization_comparison.csv", index=False)

    anomaly_rows = []
    for method, metrics in result["anomaly"].items():
        if method in {"anomaly", "duplicate"}:
            anomaly_rows.append({"method": method, **{k: v for k, v in metrics.items() if k in {"precision", "recall", "f1"}}})
    pd.DataFrame(anomaly_rows).to_csv(out / "anomaly_comparison.csv", index=False)
    _write_plots(result, output_dir)


def _write_plots(result: dict, output_dir: str | None) -> None:
    if not output_dir:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cat = pd.DataFrame([
        {"method": name, "accuracy": metrics["accuracy"], "f1": metrics["f1"]}
        for name, metrics in result["categorization"].items()
    ])
    ax = cat.set_index("method")[["accuracy", "f1"]].plot(kind="bar", figsize=(8, 5), ylim=(0, 1.05))
    ax.set_title("Transaction Categorization Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Method")
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(out / "categorization_comparison.png", dpi=160); plt.close(fig)

    an = pd.DataFrame([
        {"method": name, "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"]}
        for name, metrics in result["anomaly"].items() if name in {"anomaly", "duplicate"}
    ])
    ax = an.set_index("method")[["precision", "recall", "f1"]].plot(kind="bar", figsize=(8, 5), ylim=(0, 1.05))
    ax.set_title("Anomaly and Duplicate Detection")
    ax.set_ylabel("Score")
    ax.set_xlabel("Detector")
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(out / "anomaly_comparison.png", dpi=160); plt.close(fig)

    fc = pd.DataFrame([
        {"method": name, "MAE": metrics["mae"]}
        for name, metrics in result["forecast"].items()
    ])
    ax = fc.set_index("method")["MAE"].plot(kind="bar", figsize=(8, 5))
    ax.set_title("Forecast MAE Comparison")
    ax.set_ylabel("MAE (₹)")
    ax.set_xlabel("Method")
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(out / "forecast_mae_comparison.png", dpi=160); plt.close(fig)

def run_evaluation(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    result = {
        "dataset": {
            "rows": int(len(df)),
            "users": int(df["user_id"].nunique()),
            "months": int(pd.to_datetime(df["date"]).dt.to_period("M").nunique()),
        },
        "categorization": _categorization_experiment(df),
        "anomaly": _anomaly_experiment(df),
        "forecast": _forecast_experiment(df),
    }
    result["agentic"] = run_agent_evaluation(csv_path, output_dir=None)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/synthetic_transactions.csv")
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    args = parser.parse_args()
    result = run_evaluation(args.csv)
    _write_artifacts(result, args.output_dir)
    print(json.dumps(result, indent=2))
