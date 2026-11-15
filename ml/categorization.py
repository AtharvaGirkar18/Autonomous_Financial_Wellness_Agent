from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline


MERCHANT_RULES = {
    "SWIGGY": ("Food & Dining", "Food Delivery"),
    "ZOMATO": ("Food & Dining", "Food Delivery"),
    "LOCAL CAFE": ("Food & Dining", "Cafe"),
    "UBER": ("Transportation", "Cab"),
    "OLA": ("Transportation", "Cab"),
    "METRO": ("Transportation", "Metro"),
    "DMART": ("Groceries", "Supermarket"),
    "BIGBASKET": ("Groceries", "Grocery Delivery"),
    "NETFLIX": ("Subscriptions", "OTT"),
    "SPOTIFY": ("Subscriptions", "Music"),
    "AMAZON PRIME": ("Subscriptions", "OTT"),
    "YOUTUBE PREMIUM": ("Subscriptions", "OTT"),
    "AIRTEL": ("Bills & Utilities", "Telecom"),
    "TATA POWER": ("Bills & Utilities", "Electricity"),
    "HDFC": ("EMI / Debt", "EMI"),
    "LIC": ("Insurance", "Insurance Premium"),
    "SALARY": ("Salary / Income", "Salary"),
    "ATM": ("Cash Withdrawal", "ATM"),
}


def rule_category(text: str) -> tuple[str, str, float, str]:
    upper = str(text).upper()
    for key, (category, subcategory) in sorted(MERCHANT_RULES.items(), key=lambda x: len(x[0]), reverse=True):
        if key in upper:
            return category, subcategory, 0.97, f"Merchant matched rule: {key}"
    return "Other", "Other", 0.35, "No merchant rule matched"


@dataclass
class RuleBasedCategorizer:
    def fit(self, train_df: pd.DataFrame) -> None:
        return None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        preds = out.apply(
            lambda row: rule_category(f"{row.get('description', '')} {row.get('merchant', '')}"),
            axis=1,
            result_type="expand",
        )
        out["predicted_category"] = preds[0]
        out["predicted_subcategory"] = preds[1]
        out["category_confidence"] = preds[2].astype(float)
        out["category_reason"] = preds[3]
        out["category_method"] = "rule"
        return out


@dataclass
class MLCategorizer:
    ml_pipeline: Pipeline | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        labeled = train_df[train_df["category"].notna()].copy()
        if len(labeled) < 30 or labeled["category"].nunique() < 2:
            return
        texts = labeled["description"].fillna("") + " " + labeled["merchant"].fillna("")
        self.ml_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        self.ml_pipeline.fit(texts, labeled["category"])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.ml_pipeline is None:
            out["predicted_category"] = "Other"
            out["predicted_subcategory"] = "Other"
            out["category_confidence"] = 0.20
            out["category_reason"] = "ML model unavailable"
            out["category_method"] = "ml_unavailable"
            return out
        texts = out["description"].fillna("") + " " + out["merchant"].fillna("")
        out["predicted_category"] = self.ml_pipeline.predict(texts)
        try:
            out["category_confidence"] = self.ml_pipeline.predict_proba(texts).max(axis=1)
        except Exception:
            out["category_confidence"] = 0.60
        out["category_subcategory"] = ""
        out["predicted_subcategory"] = out["category_subcategory"]
        out["category_reason"] = "TF-IDF + Logistic Regression prediction"
        out["category_method"] = "ml"
        return out.drop(columns=["category_subcategory"])


@dataclass
class HybridCategorizer:
    ml_pipeline: Pipeline | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        model = MLCategorizer()
        model.fit(train_df)
        self.ml_pipeline = model.ml_pipeline

    def predict_row(self, row: pd.Series) -> dict:
        category, subcategory, confidence, reason = rule_category(
            f"{row.get('description', '')} {row.get('merchant', '')}"
        )
        if confidence >= 0.9:
            return {
                "category": category,
                "subcategory": subcategory,
                "confidence": confidence,
                "reason": reason,
                "method": "rule",
            }
        if self.ml_pipeline is not None:
            text = f"{row.get('description', '')} {row.get('merchant', '')}"
            predicted = self.ml_pipeline.predict([text])[0]
            try:
                proba = float(self.ml_pipeline.predict_proba([text]).max())
            except Exception:
                proba = 0.60
            return {
                "category": predicted,
                "subcategory": "",
                "confidence": round(proba, 3),
                "reason": "TF-IDF + Logistic Regression prediction",
                "method": "ml",
            }
        return {
            "category": "Other",
            "subcategory": "Other",
            "confidence": 0.35,
            "reason": "Ambiguous transaction; LLM fallback may be used",
            "method": "fallback",
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        preds = out.apply(self.predict_row, axis=1, result_type="expand")
        out["predicted_category"] = preds["category"]
        out["predicted_subcategory"] = preds["subcategory"]
        out["category_confidence"] = preds["confidence"].astype(float)
        out["category_reason"] = preds["reason"]
        out["category_method"] = preds["method"]
        return out


def evaluate_categorization(y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }
