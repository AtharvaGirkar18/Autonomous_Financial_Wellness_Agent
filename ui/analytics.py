import pandas as pd
import plotly.express as px
import streamlit as st


def render_analytics(state: dict):
    a = state.get("analytics", {})
    cat = pd.DataFrame([{"category": k, "amount": v} for k, v in a.get("category_totals", {}).items()])
    if not cat.empty:
        st.plotly_chart(px.pie(cat, names="category", values="amount", title="Category Spending"), use_container_width=True)

    trends = a.get("trends", {})
    trend_rows = []
    if isinstance(trends, dict):
        for category, values in trends.items():
            # Older or partial workflow states can contain missing or malformed
            # trend entries. Keep the UI usable and treat missing growth as zero.
            values = values if isinstance(values, dict) else {}
            trend_rows.append({
                "category": category,
                "growth_3m": values.get("growth_3m", 0.0),
                "direction": values.get("direction", "Insufficient history"),
            })
    tdf = pd.DataFrame(trend_rows, columns=["category", "growth_3m", "direction"])
    if not tdf.empty:
        tdf["growth_3m"] = pd.to_numeric(tdf["growth_3m"], errors="coerce").fillna(0.0)
        tdf = tdf.sort_values("growth_3m", ascending=False)
    if not tdf.empty:
        tdf["growth_pct"] = (tdf["growth_3m"] * 100).round(1)
        st.plotly_chart(
            px.bar(tdf.head(10), x="category", y="growth_pct", title="3-Month Category Growth (%)"),
            use_container_width=True,
        )
        st.dataframe(tdf, use_container_width=True, hide_index=True)

    monthly = pd.DataFrame([
        {"month": month, "savings": income - a.get("monthly_expenses", {}).get(month, 0)}
        for month, income in a.get("monthly_income", {}).items()
    ])
    if not monthly.empty:
        st.plotly_chart(px.line(monthly, x="month", y="savings", markers=True, title="Monthly Savings Trend"), use_container_width=True)
