import pandas as pd
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from agents.usage_review_agent import LABELS, potential_savings, reviewed_subscriptions, usage_review_node


def render_subscriptions(state):
    st.subheader("Subscription usage review")
    if not state.get("subscriptions"):
        st.info("No recurring subscription charges were detected for this profile.")
        return
    try:
        rows = reviewed_subscriptions(state["user_id"], state["subscriptions"])
    except SQLAlchemyError:
        st.error("Saved reviews could not be read. Reload the page to retry.")
        return
    state["subscriptions"] = rows
    st.dataframe(pd.DataFrame([{
        "Service": r["merchant"], "Billing": r["billing_frequency"],
        "Monthly equivalent (₹)": r["monthly_cost"], "Annual equivalent (₹)": r["yearly_cost"],
        "Observed months": r["observed_months"], "Usage": LABELS[r["usage_status"]],
    } for r in rows]), use_container_width=True, hide_index=True)
    monthly, annual = potential_savings(rows)
    left, right = st.columns(2)
    left.metric("Potential monthly savings", f"₹{monthly:,.2f}")
    right.metric("Potential annual savings", f"₹{annual:,.2f}")
    st.caption("These estimates count services you confirm you no longer use. Saving a review does not cancel a service.")
    user = state["user_id"]
    merchant = st.selectbox("Service to review", [r["merchant"] for r in rows], key=f"usage_service_{user}")
    prefix = f"usage_{user}_{merchant}"
    saved = next(r["usage_status"] for r in rows if r["merchant"] == merchant)
    note = st.text_input("How do you use this service?", max_chars=300,
                         placeholder="I used Spotify 4 months out of the last 6.", key=prefix + "_note")
    if st.button("Interpret note with AI", key=prefix + "_interpret"):
        try:
            with st.spinner("Interpreting your usage note…"):
                result = usage_review_node(state, merchant, note=note)
            st.session_state[prefix + "_choice"] = result["suggestion"]
            if result["source"] == "usage_frequency":
                st.info(f"{result['reason']} Suggested status: {LABELS[result['suggestion']]}. Confirm below to save.")
                if result["action_source"] == "model":
                    st.write(f"AI-selected action: {result['action']}")
                else:
                    st.caption(f"No usable AI response. Calculator fallback: {result['action']}")
            elif result["source"] == "manual":
                st.warning("No usable model response. Choose the status yourself below.")
            elif result["suggestion"] == "unknown":
                st.info("The note was inconclusive. Choose a status or leave it unreviewed.")
            else:
                st.info(f"AI suggestion: {LABELS[result['suggestion']]}. Confirm below to save it.")
        except ValueError as exc:
            st.session_state[prefix + "_choice"] = saved
            st.warning(str(exc))
    st.radio("Usage status", list(LABELS), index=list(LABELS).index(saved),
                      format_func=LABELS.get, horizontal=True, key=prefix + "_choice")
    def save_review():
        try:
            usage_review_node(state, merchant, confirmed_status=st.session_state[prefix + "_choice"])
            st.session_state[prefix + "_feedback"] = "saved"
        except (ValueError, SQLAlchemyError):
            st.session_state[prefix + "_feedback"] = "error"

    st.button("Confirm and save usage", key=prefix + "_save", on_click=save_review)
    feedback = st.session_state.pop(prefix + "_feedback", None)
    if feedback == "saved":
        st.success(f"Saved your review for {merchant}.")
    elif feedback == "error":
        st.error("The review could not be saved. Reload the profile and try again.")
