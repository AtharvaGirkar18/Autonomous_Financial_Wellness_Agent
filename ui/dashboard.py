from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def render_overview(state: dict):
    h = state.get("health", {})
    a = state.get("analytics", {})
    alerts = state.get("alerts", [])
    monthly_income = h.get("monthly_income")
    if monthly_income is None or float(monthly_income or 0) <= 0:
        monthly_income = a.get("income_total", 0) / max(1, len(a.get("monthly_income", {})))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Financial Health", f"{h.get('score', 0):.0f}/100")
    c2.metric("Current Balance", f"₹{state.get('current_balance', 0):,.0f}")
    c3.metric("Monthly Income", f"₹{float(monthly_income or 0):,.0f}")
    c4.metric("Monthly Savings", f"₹{h.get('monthly_savings', 0):,.0f}")
    c5.metric("Active Alerts", len(alerts))

    with st.container(border=True):
        readiness = state.get("investment_readiness", {})
        goal_rows = state.get("goals", [])
        gc1, gc2, gc3 = st.columns(3)
        gc1.metric("Emergency Coverage", f"{h.get('emergency_months', 0):.2f} months")
        gc2.metric("Investment Readiness", readiness.get("level", "Run readiness check"))
        gc3.metric("Active Goals", f"{len(goal_rows)} Goal(s)" if goal_rows else "0 Goals")

    # ── Active Goals Section ──────────────────────────────────────────────────
    if goal_rows:
        st.markdown("### 🎯 Active Financial Goals")
        cols = st.columns(min(len(goal_rows), 3))
        for idx, g in enumerate(goal_rows):
            with cols[idx % len(cols)]:
                with st.container(border=True):
                    status = g.get("status", "On Track")
                    is_on_track = str(status).lower() == "on track"
                    icon = "🟢" if is_on_track else "🟡"
                    st.markdown(f"**{g.get('name', 'Goal')}**")
                    st.caption(f"{icon} Status: **{status}**")
                    target = float(g.get("target_amount", 0))
                    current = float(g.get("current_amount", 0))
                    pct = min(1.0, current / max(1.0, target)) if target > 0 else 0.0
                    st.progress(pct, text=f"Saved: ₹{current:,.0f} / ₹{target:,.0f} ({pct*100:.0f}%)")
                    st.caption(
                        f"📅 Target: {g.get('target_date', 'N/A')} · "
                        f"Req: ₹{g.get('required_monthly_contribution', 0):,.0f}/mo"
                    )
    else:
        st.info("ℹ️ No active goals yet. Add your targets in the **Goals** tab to track timeline feasibility.")

    monthly_income = a.get("monthly_income", {})
    monthly_expenses = a.get("monthly_expenses", {})
    months = sorted(set(monthly_income) | set(monthly_expenses))
    monthly = pd.DataFrame([
        {"month": month, "Income": monthly_income.get(month, 0), "Expenses": monthly_expenses.get(month, 0)}
        for month in months
    ])
    if not monthly.empty:
        st.plotly_chart(
            px.bar(monthly, x="month", y=["Income", "Expenses"], barmode="group", title="Monthly Income vs Expenses"),
            use_container_width=True,
        )

    components = pd.DataFrame([
        {"component": k.replace("_", " ").title(), "score": v}
        for k, v in h.get("components", {}).items()
    ])
    if not components.empty:
        st.plotly_chart(
            px.bar(components.sort_values("score"), x="score", y="component", orientation="h", range_x=[0, 100], title="Financial Health Components"),
            use_container_width=True,
        )
