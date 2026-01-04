"""Reusable chart helpers for Streamlit."""
from __future__ import annotations

import pandas as pd
import streamlit as st

__all__ = ["q_history_linechart"]


def q_history_linechart(history: list[dict]) -> None:
    """Render line chart of Q-value evolution using st.line_chart."""
    if not history:
        st.info("Train the agent to see Q-value history.")
        return
    df = pd.DataFrame(history)
    if "Episode" in df.columns:
        df = df.set_index("Episode")
    st.line_chart(df)

