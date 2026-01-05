"""Chart components for Q-learning demo visualization."""
from __future__ import annotations

import pandas as pd
import streamlit as st

__all__ = ["render_q_history_chart", "render_steps_chart"]

def render_q_history_chart(q_history: list[dict]) -> None:

    if q_history:

        hist_df = pd.DataFrame(q_history)
        if not hist_df.empty:
            hist_df = hist_df.set_index("Episode")
            # Downsample for display if too many data points (performance optimization)
            if len(hist_df) > 100:
                step = len(hist_df) // 100
                hist_df = hist_df.iloc[::step]
            st.line_chart(hist_df)
    else:
        st.write("No history yet.")

def render_steps_chart(steps_per_episode: list[int]) -> None:
    """Render a line chart showing steps taken per completed episode.
    
    Args:
        steps_per_episode: List of step counts for each completed episode.
                          Index 0 = episode 1, etc.
    """
    if not steps_per_episode:
        st.write("No completed episodes yet.")
        return
    
    # Create DataFrame with 1-indexed episode numbers
    df = pd.DataFrame({
        "Episode": range(1, len(steps_per_episode) + 1),
        "Steps": steps_per_episode,
    })
    df = df.set_index("Episode")
    
    st.line_chart(df)
