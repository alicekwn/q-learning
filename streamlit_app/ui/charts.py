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

def render_steps_chart(steps_per_episode: list[int], q_history_plot: list[dict] = None) -> None:
    """Render a line chart showing steps taken per completed episode.
    
    Only shows steps for episodes that are in q_history_plot (same as evolving Q-values plot)
    to avoid performance issues with large episode counts.
    
    Args:
        steps_per_episode: List of step counts for each completed episode.
                          Index 0 = episode 0, etc.
        q_history_plot: List of Q-value snapshots, each with an "Episode" key.
                       If provided, only shows steps for these episodes.
    """
    if not steps_per_episode:
        st.write("No completed episodes yet.")
        return
    
    # If q_history_plot is provided, only show steps for episodes in it
    if q_history_plot:
        # Extract episode numbers from q_history_plot
        episode_numbers = [entry.get("Episode") for entry in q_history_plot if "Episode" in entry]
        
        if not episode_numbers:
            st.write("No episode data yet.")
            return
        
        # Filter steps_per_episode to only include episodes in q_history_plot
        # Note: Episode N in q_history_plot means N episodes have been completed
        # steps_per_episode[0] = first completed episode, steps_per_episode[1] = second, etc.
        # So Episode N corresponds to steps_per_episode[N-1] (for N > 0)
        # Episode 0 is the initial state with no steps yet
        filtered_data = []
        for ep_num in episode_numbers:
            if ep_num == 0:
                # Episode 0 is initial state, no steps yet
                continue
            # Episode N corresponds to steps_per_episode[N-1]
            steps_idx = ep_num - 1
            if 0 <= steps_idx < len(steps_per_episode):
                filtered_data.append({
                    "Episode": ep_num,  # Use episode number from q_history_plot
                    "Steps": steps_per_episode[steps_idx]
                })
        
        if not filtered_data:
            st.write("No matching episode data yet.")
            return
        
        df = pd.DataFrame(filtered_data)
        df = df.set_index("Episode")
    else:
        # Fallback: show all episodes if q_history_plot not provided
        df = pd.DataFrame({
            "Episode": range(1, len(steps_per_episode) + 1),
            "Steps": steps_per_episode,
        })
        df = df.set_index("Episode")
    
    st.line_chart(df)
