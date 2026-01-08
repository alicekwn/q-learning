"""Bellman update log display helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st

__all__ = ["render_bellman_log", "render_bellman_log_econ"]


def render_bellman_log(history_log: list[dict]) -> None:
    """Display Bellman update equations and history table."""
    st.subheader("Bellman Update Log")
    st.latex(
        r"Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]"
    )  # Bellman equation

    if not history_log:
        st.info("Train a new episode step by step to see the Q-value updates.")
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


def render_bellman_log_econ(history_log: list[dict]) -> None:
    """Display Bellman update equations and history table for economics pricing (dual Q-matrices).

    Args:
        history_log: List of step log entries, alternating between Q1 (Alice) and Q2 (Bob) entries
    """
    st.subheader("Bellman Update Log")
    st.latex(
        r"Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \delta \max_{a'} Q(s', a') - Q(s, a) \right]"
    )  # Bellman equation

    if not history_log:
        st.info("Start training to see the Q-value updates.")
        return

    # Separate Q1 and Q2 logs
    q1_logs = [log for log in history_log if log.get("Player") == "Alice (Q1)"]
    q2_logs = [log for log in history_log if log.get("Player") == "Bob (Q2)"]

    # Get latest updates if available
    last_q1 = q1_logs[-1] if q1_logs else None
    last_q2 = q2_logs[-1] if q2_logs else None

    # Show history tables in tabs
    tab1, tab2 = st.tabs([r"👩🏼‍💼 Alice ($Q_1$)", r"🧑🏼‍💼 Bob ($Q_2$)"])

    with tab1:
        if last_q1:
            st.markdown(
                f"**Step {last_q1['Step']} ({last_q1['Type']}):**",
                help=f"After taking action a1={last_q1['Action (a1)']} and a2={last_q1['Action (a2)']} at state {last_q1['State (s)']}, the next state is {last_q1['Next state']}.",
            )
            st.latex(last_q1["Equation"])
            # Show reward calculation
            p1 = last_q1.get("p1", 0)
            c = last_q1.get("c", 0)
            q1 = last_q1.get("q1", 0)
            pi1 = last_q1.get("Reward (π1)", 0)
            st.latex(
                rf"\pi_1 = (p_1 - c) \times q_1 = ({p1:.1f} - {c:.1f}) \times {q1:.4f} = {pi1:.4f}"
            )
        if q1_logs:
            log_df_1 = pd.DataFrame(q1_logs)
            display_cols_1 = [
                "Step",
                "State (s)",
                "Action (a1)",
                "Action (a2)",
                "Type",
                "Equation",
                "New Q",
                "Reward (π1)",
            ]
            available_cols_1 = [
                col for col in display_cols_1 if col in log_df_1.columns
            ]
            st.dataframe(
                log_df_1[available_cols_1].sort_values(by="Step", ascending=False),
                width="stretch",
                height=300,
            )
        else:
            st.info(r"No $Q_1$ updates yet.")

    with tab2:
        if last_q2:
            st.markdown(
                f"**Step {last_q2['Step']} ({last_q2['Type']}):**",
                help=f"After taking action a1={last_q2['Action (a1)']} and a2={last_q2['Action (a2)']} at state {last_q2['State (s)']}, the next state is {last_q2['Next state']}.",
            )
            st.latex(last_q2["Equation"])
            # Show reward calculation
            p2 = last_q2.get("p2", 0)
            c = last_q2.get("c", 0)
            q2 = last_q2.get("q2", 0)
            pi2 = last_q2.get("Reward (π2)", 0)
            st.latex(
                rf"\pi_2 = (p_2 - c) \times q_2 = ({p2:.1f} - {c:.1f}) \times {q2:.4f} = {pi2:.4f}"
            )
        if q2_logs:
            log_df_2 = pd.DataFrame(q2_logs)
            display_cols_2 = [
                "Step",
                "State (s)",
                "Action (a1)",
                "Action (a2)",
                "Type",
                "Equation",
                "New Q",
                "Reward (π2)",
            ]
            available_cols_2 = [
                col for col in display_cols_2 if col in log_df_2.columns
            ]
            st.dataframe(
                log_df_2[available_cols_2].sort_values(by="Step", ascending=False),
                width="stretch",
                height=300,
            )
        else:
            st.info("No Q2 updates yet.")
