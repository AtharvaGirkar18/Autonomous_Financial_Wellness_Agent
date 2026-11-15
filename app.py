from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agents.orchestrator import run_workflow
from config.settings import DATA_DIR, settings
from models.database import init_db
from tools.llm import llm_status
from tools.pdf_tools import extract_transactions_from_pdf
from ui.advisor import render_advisor
from ui.analytics import render_analytics
from ui.dashboard import render_overview
from ui.observability import render_trace
from ui.simulations import render_simulations
from ui.transactions import render_transactions

# Load .env (Ollama is keyless — no GOOGLE_API_KEY required for default config).
load_dotenv(override=True)

st.set_page_config(
    page_title="Autonomous Financial Wellness Agent",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_db()


def load_profiles() -> pd.DataFrame:
    path = DATA_DIR / "synthetic_users.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame([
        {"user_id": "USER001", "name": "Demo User", "monthly_income": 75000, "starting_balance": 32000}
    ])


def execute_analysis(user_id: str, transactions: pd.DataFrame, profile: dict, goals: list[dict], request: str):
    starting_balance = float(profile.get("starting_balance", 32000))
    current_balance = starting_balance
    if isinstance(transactions, pd.DataFrame) and "transaction_type" in transactions.columns and "amount" in transactions.columns:
        cr = float(pd.to_numeric(transactions.loc[transactions["transaction_type"].astype(str).str.upper().eq("CREDIT"), "amount"], errors="coerce").sum())
        dr = float(pd.to_numeric(transactions.loc[transactions["transaction_type"].astype(str).str.upper().eq("DEBIT"), "amount"], errors="coerce").sum())
        net = cr - dr
        if net > 0:
            current_balance = starting_balance + net
    return run_workflow({
        "user_id": user_id,
        "profile": profile,
        "request": request,
        "transactions": transactions.to_dict(orient="records"),
        "current_balance": current_balance,
        "starting_balance": starting_balance,
        "goals": goals,
        "trace": [],
    })


profiles = load_profiles()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # AI status chip — reflects current LLM_PROVIDER setting
    status = llm_status()
    provider_label = {
        "ollama": f"🦙 Ollama · {status['model']}",
        "google": f"✨ Gemini · {status['model']}",
        "grok": f"🤖 Grok · {status['model']}",
        "openai": f"🔵 OpenAI · {status['model']}",
    }.get(status["provider"], status["model"])

    if status["enabled"]:
        st.success(f"✅ AI connected — {provider_label}")
    elif status["provider"] == "ollama":
        st.warning("⚠️ Ollama offline — run `ollama serve` and `ollama pull qwen2.5:7b`")
    else:
        st.warning("⚠️ AI offline — set key in .env file")

    # MCP + Governance status
    try:
        from mcp.registry import mcp_registry
        tool_count = len(mcp_registry.list_tools())
        st.caption(f"🔧 MCP: {tool_count} tools across {len(mcp_registry._backends)} backends")
    except Exception:
        pass

    if settings.governance_enabled:
        st.caption("🛡️ Governance: enabled · GuardrailBus: active")

    st.divider()

    # ── Demo / Data ───────────────────────────────────────────────────────────
    st.header("📂 Demo / Data")
    user_id = st.selectbox("Synthetic user", profiles["user_id"].tolist(), index=0)
    profile_row = profiles.loc[profiles["user_id"].eq(user_id)].iloc[0].to_dict()
    demo = st.button("Load Demo Financial Profile", use_container_width=True)
    uploaded = st.file_uploader(
        "Upload CSV or PDF statement",
        type=["csv", "pdf"],
        help="CSV must have columns: date, description, amount, transaction_type. See data/README_datasets.md for free public datasets.",
    )




profile = profile_row
# Startup & restart: goals start empty so the user can add custom goals dynamically.
goals: list[dict] = []

if "state" not in st.session_state:
    st.session_state.state = None

if demo:
    tx_path = DATA_DIR / "synthetic_transactions.csv"
    df = pd.read_csv(tx_path)
    df = df[df["user_id"].eq(user_id)].copy()
    st.session_state.state = execute_analysis(
        user_id, df, profile, goals,
        "Run my complete financial wellness analysis and proactive monitoring cycle.",
    )

elif uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".pdf"):
            df = extract_transactions_from_pdf(uploaded.getvalue(), user_id=user_id)
        else:
            df = pd.read_csv(uploaded)
            if "user_id" not in df.columns:
                df["user_id"] = user_id
        st.session_state.state = execute_analysis(
            user_id, df, profile, goals,
            "Run my complete financial wellness analysis and proactive monitoring cycle.",
        )
    except Exception as exc:
        st.error(f"Could not process the uploaded statement: {exc}")

state = st.session_state.state
if state is None:
    st.title("💰 Autonomous Financial Wellness Agent")
    st.caption("Data → Intelligence → Agents → Tools → Reasoning → Prediction → Action → Feedback")
    st.info("Click **Load Demo Financial Profile** to run the complete agentic workflow (Ollama Qwen 2.5 7B, no API key needed).")
    st.markdown(
        """
        ### What this system proves
        **Local LLM:** Qwen 2.5 7B runs entirely on your machine via Ollama — no cloud API key required.

        **Agents of Governance:** A GovernanceAgent validates every plan pre-flight and signs off post-flight.

        **GuardrailBus:** A composable, layered guardrail pipeline replaces ad-hoc checks with a unified policy surface.

        **Multiple MCPs:** Three MCP backends (Finance, Analytics, Data) provide swappable tool backends — no hardcoded API calls.

        **Distributed Coordination:** A DistributedCoordinator runs independent agents in parallel with a dependency-aware scheduler and event bus.

        **Multi-agent coordination:** Specialist agents update shared state before the advisor and reviewer run.

        **Memory:** Recommendations, alerts, health history and agent traces persist in SQLite.
        """
    )
    st.stop()

st.title("💰 Autonomous Financial Wellness Agent")
st.caption(f"Profile: {profile.get('name', user_id)} · Synthetic demonstration data")

tabs = st.tabs([
    "Overview", "Transactions", "Spending Analysis", "Anomalies", "Subscriptions", "Budget",
    "Cash Flow", "Goals", "What-If", "AI Advisor", "Alerts", "Agent Trace",
])

with tabs[0]:
    render_overview(state)
with tabs[1]:
    render_transactions(state)
with tabs[2]:
    render_analytics(state)
with tabs[3]:
    anomalies = pd.DataFrame(state.get("anomalies", []))
    if anomalies.empty:
        st.success("No anomalies detected in this run.")
    else:
        st.subheader("Unusual transactions")
        st.dataframe(anomalies, use_container_width=True, hide_index=True)
    duplicates = pd.DataFrame(state.get("duplicate_candidates", []))
    if not duplicates.empty:
        st.subheader("Duplicate charge candidates")
        st.dataframe(duplicates, use_container_width=True, hide_index=True)
with tabs[4]:
    subscriptions = pd.DataFrame(state.get("subscriptions", []))
    st.dataframe(subscriptions, use_container_width=True, hide_index=True)
with tabs[5]:
    budget_state = state.get("budget", {})
    budget_rows = pd.DataFrame(budget_state.get("rows", []))
    st.caption(f"Budget month: {budget_state.get('month', 'latest')}")
    st.dataframe(budget_rows, use_container_width=True, hide_index=True)
    st.markdown("### Edit category budgets")
    with st.form("budget_editor"):
        overrides = {}
        categories = list(budget_state.get("categories", {}).keys())
        cols = st.columns(4)
        for idx, category in enumerate(categories):
            with cols[idx % 4]:
                overrides[category] = st.number_input(
                    category, min_value=0.0, value=float(budget_state["categories"].get(category, 0)), step=250.0, key=f"budget_{category}"
                )
        submitted = st.form_submit_button("Save budgets and re-run analysis")
    if submitted:
        new_state = {**state, "budget_overrides": overrides, "request": "Update my budget and reassess financial health.", "trace": []}
        st.session_state.state = run_workflow(new_state)
        st.rerun()
with tabs[6]:
    forecast = state.get("forecast", {})
    st.metric("Lowest projected balance", f"₹{forecast.get('lowest_projected_balance', 0):,.0f}")
    st.caption(f"Projected low: {forecast.get('lowest_projected_date', 'N/A')}")
    st.dataframe(pd.DataFrame(forecast.get("forecast", [])), use_container_width=True, hide_index=True)
with tabs[7]:
    goals_view = pd.DataFrame(state.get("goals", []))
    if goals_view.empty:
        st.info("No active goals yet. Add a goal below to evaluate savings pace and feasibility.")
    else:
        st.subheader("Your Active Goals")
        st.dataframe(goals_view, use_container_width=True, hide_index=True)
        if st.button("Clear all goals"):
            new_state = {**state, "goals": [], "request": "Cleared all financial goals.", "trace": []}
            st.session_state.state = run_workflow(new_state)
            st.rerun()
    st.markdown("### Add a financial goal")
    with st.form("goal_editor"):
        goal_name = st.text_input("Goal name", "Emergency Fund")
        goal_target = st.number_input("Target amount", min_value=1000.0, value=100000.0, step=5000.0)
        goal_current = st.number_input("Current saved amount", min_value=0.0, value=0.0, step=5000.0)
        goal_months = st.number_input("Timeline (months)", min_value=1, value=12, step=1)
        goal_submit = st.form_submit_button("Add goal and re-run analysis")
    if goal_submit:
        from datetime import date
        from dateutil.relativedelta import relativedelta
        target_date = (date.today() + relativedelta(months=int(goal_months))).isoformat()
        new_goal = {
            "user_id": user_id,
            "name": goal_name,
            "target_amount": float(goal_target),
            "target_date": target_date,
            "current_amount": float(goal_current),
            "monthly_contribution": float(state.get("health", {}).get("monthly_savings", 0)),
        }
        new_state = {**state, "goals": state.get("goals", []) + [new_goal], "request": "Reassess my goals and financial health.", "trace": []}
        st.session_state.state = run_workflow(new_state)
        st.rerun()
with tabs[8]:
    render_simulations(state)
with tabs[9]:
    render_advisor(state)
    st.markdown("---")
    prompt = st.text_input("Ask the financial advisor", placeholder="Can I afford a ₹12 lakh car?")
    if st.button("Run advisor workflow") and prompt:
        st.session_state.state = run_workflow({**state, "request": prompt, "trace": []})
        st.rerun()
with tabs[10]:
    alerts = pd.DataFrame(state.get("alerts", []))
    if alerts.empty:
        st.success("No proactive alerts for this run.")
    else:
        st.dataframe(alerts, use_container_width=True, hide_index=True)
with tabs[11]:
    render_trace(state)

st.markdown("---")
st.caption(
    "Disclaimer: This system provides educational and financial wellness insights, not regulated investment, tax, or professional financial advice. "
    "Predictions are estimates; verify important decisions independently."
)
