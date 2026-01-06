"""Bellman update log display helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st

__all__ = ["render_bellman_log"]


def render_bellman_log(history_log: list[dict]) -> None:
    """Display Bellman update equations and history table."""
    st.subheader("Bellman Update Log")
    st.latex(
        r"Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]"
    )  # Bellman equation

    if not history_log:
        st.info("Click 'Take Next Step' to see the Q-value updates.")
        return

    # Show latest update prominently
    last_log = history_log[-1]
    st.markdown(
        f"**Episode {last_log['Episode']}, Step {last_log['Step']} ({last_log['Type']}):**",
        help=f"Action chosen for the next step is ${last_log['Next action']}$. After taking action ${last_log['Action (a)']}$ at state ${last_log['State (s)']}$, the dog will land on state ${last_log['Next state']}$.",
    )

    st.latex(
        last_log["Equation"],
        help=rf"By comparing $Q({last_log['Next state']}, a')$ for all possible actions $a'$, the maximum $Q({last_log['Next state']}, a')$ is when $a' = {last_log['Next action']}$, where $Q({last_log['Next state']}, {last_log['Next action']}) = {last_log['Max next Q']:.4f}$. (Note that when there's a tie, $a'$ is chosen randomly between the tied actions.)",
    )

    # Show history table (newest on top)
    log_df = pd.DataFrame(history_log)
    st.dataframe(
        log_df[
            ["Episode", "Step", "State (s)", "Action (a)", "Type", "Equation", "New Q"]
        ].sort_values(by=["Episode", "Step"], ascending=False),
        width="stretch",
        height=300,
    )
