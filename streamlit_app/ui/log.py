"""Bellman update log display helpers."""
from __future__ import annotations

import pandas as pd
import streamlit as st

__all__ = ["render_bellman_log"]


def render_bellman_log(history_log: list[dict]) -> None:
    """Display Bellman update equations and history table."""
    st.subheader("Bellman Update Log")
    st.latex(r"Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]") # Bellman equation
    
    if not history_log:
        st.info("Click 'Take Next Step' to see the Q-value updates.")
        return
    
    # Show latest update prominently
    last_log = history_log[-1]
    st.markdown(f"**Episode {last_log['Episode']}, Step {last_log['Step']} ({last_log['Type']}):**")
    st.latex(last_log['Equation'])
    
    # Show history table (newest on top)
    log_df = pd.DataFrame(history_log)
    st.dataframe(
        log_df[["Episode", "Step", "State", "Action", "Type", "Equation", "New Q"]].sort_values(
            by=["Episode", "Step"], ascending=False
        ),
        use_container_width=True,
        height=200
    )

