import streamlit as st

from tools.simulation_tools import simulate_car, simulate_job_loss, simulate_salary_change


def render_simulations(state: dict):
    a = state.get("analytics", {})
    h = state.get("health", {})
    choice = st.selectbox("Scenario", ["Buy Car", "Salary Change", "Job Loss"])

    if choice == "Buy Car":
        price = st.number_input("Car price", 100000.0, 10000000.0, 1200000.0, step=50000.0)
        down = st.number_input("Down payment", 0.0, price, min(250000.0, price), step=10000.0)
        rate = st.number_input("Annual interest rate (%)", 0.0, 30.0, 9.0, step=0.1)
        months = st.slider("Tenure (months)", 12, 84, 60)
        if st.button("Simulate car purchase"):
            profile = state.get("profile", {})
            result = simulate_car(
                h.get("monthly_savings", 0),
                price,
                down,
                rate,
                months,
                state.get("current_balance", 32000),
                h.get("debt_ratio", 0) / 100,
                profile.get("monthly_income", 0),
            )
            st.json(result)

    elif choice == "Salary Change":
        old = st.number_input("Current salary", 10000.0, 500000.0, float(state.get("profile", {}).get("monthly_income", 60000)), step=5000.0)
        new = st.number_input("New salary", 10000.0, 500000.0, old + 15000, step=5000.0)
        if st.button("Simulate salary change"):
            st.json(simulate_salary_change(h.get("monthly_savings", 0), old, new))

    else:
        essential = h.get("monthly_essential_expenses", a.get("essential_total", 0) / 6)
        balance = state.get("current_balance", 32000)
        if st.button("Simulate 6-month job loss"):
            st.json(simulate_job_loss(essential, balance))
