import pandas as pd
import streamlit as st


def render_transactions(state: dict):
    df = pd.DataFrame(state.get("transactions", []))
    if df.empty:
        st.info("No transactions loaded.")
        return
    q = st.text_input("Search merchant/description")
    if q:
        mask = df["merchant"].astype(str).str.contains(q, case=False, na=False) | df["description"].astype(str).str.contains(q, case=False, na=False)
        df = df[mask]
    cats = st.multiselect("Category", sorted(df["category"].dropna().unique()))
    if cats:
        df = df[df["category"].isin(cats)]
    st.dataframe(df, use_container_width=True, hide_index=True)
