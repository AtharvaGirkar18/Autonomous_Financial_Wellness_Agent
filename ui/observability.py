import streamlit as st


def render_trace(state: dict):
    st.subheader("Agent Activity / Observability")
    for item in state.get("trace", []):
        cols = st.columns([1.2, 2.4, 1, 3])
        cols[0].write(item.get("start_time", ""))
        cols[1].write(item.get("agent", ""))
        cols[2].write(item.get("status", ""))
        cols[3].write(", ".join(item.get("tool_calls", [])))
        with st.expander(f"{item.get('agent', 'Agent')} details", expanded=False):
            st.write("Input:", item.get("input_summary", ""))
            st.write("Output:", item.get("output_summary", ""))
            if item.get("error"):
                st.error(item["error"])

    with st.expander("Planner state"):
        st.json(state.get("plan", {}))
    with st.expander("Reviewer"):
        st.json(state.get("reviewer", {}))
