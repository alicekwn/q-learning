"""Dog & Bone Q-learning demo page."""
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
st.set_page_config(page_title="The Dog & The Bone – Q-Learning", layout="wide")

ROOT = Path(__file__).resolve().parent.parent.parent  # project root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from streamlit_app.ui.controls import tab_controls, inline_help
from streamlit_app.ui.grid import render_grid
from streamlit_app.ui.log import render_bellman_log
from streamlit_app.state import (
    init_session_state,
    reset_episode,
    step_agent,
    run_batch_training,
    rewind_checkpoint,
    forward_checkpoint,
    jump_to_latest,
    get_display_state,
    is_in_playback_mode,
)

st.title("The Dog & The Bone")

st.markdown(
    """
    In this demo, we use Q-learning to help the dog 🐶 learn how to find the bone 🍖! 
    Change the parameters to see how they affect learning, and use the controls to step through the process.
    """
)

# Two-column layout: controls on left, main content on right
col_controls, col_spacer, col_main = st.columns([1, 0.1, 5])

with col_controls:
    # Styling panel background
    st.markdown(
        """
        <style>
        [data-testid="column"]:first-child {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Render controls
    config_tab1 = tab_controls("tab1")

    st.markdown("---")
    if st.button("Reset / Initialize Model", type="primary", key="tab1_reset"):
        init_session_state(config_tab1)
        st.rerun()

# Initialize on first load
if "tab1_q_table" not in st.session_state:
    init_session_state(config_tab1)

display_state = get_display_state(config_tab1)
in_playback = is_in_playback_mode(config_tab1)

# Visual separator
with col_spacer:
    st.markdown(
        """<div style='border-left: 2px solid #e0e0e0; height: 100vh;'></div>""",
        unsafe_allow_html=True,
    )

with col_main:
    # Playback indicator
    if in_playback:
        action = display_state.get("action_type", "unknown")
        meta = display_state.get("metadata", {})
        if action == "batch":
            desc = f"Batch Training ({meta.get('episodes', '?')} episodes)"
        else:
            desc = f"Episode {meta.get('episode', '?')}, Step {meta.get('step', '?')}"
        st.warning(f"⏮️ **Playback Mode** - {desc}")

    # --- A. GRID ---
    ready = display_state.get("ready_for_episode", True)
    render_grid(
        config_tab1["grid_size"],
        display_state["current_state"],
        config_tab1["goal_pos"],
        display_state["current_path"],
        show_path=not ready,
    )

    # --- B. CONTROLS AREA ---
    col_step, col_info = st.columns([1, 2])

    with col_step:
        ready = display_state.get("ready_for_episode", True)
        if ready:
            if display_state.get("is_terminal", st.session_state.tab1_is_terminal) and display_state["total_episodes"] > 0:
                st.success(
                    f"🎉 Goal Reached! Episode {display_state['total_episodes']} complete."
                )
            if st.button("Train a new episode step by step", key="tab1_new_episode", disabled=in_playback):
                jump_to_latest(config_tab1)
                reset_episode(config_tab1)
                st.rerun()
        else:
            if st.button("👟 Take Next Step", key="tab1_step", disabled=in_playback):
                jump_to_latest(config_tab1)
                step_agent(config_tab1)
                st.rerun()

        # Fast forward
        if ready and not in_playback:
            st.markdown("Or")
            n_episodes = st.number_input(
                "Speed up training by running this many episodes:",
                min_value=1,
                value=1,
                key="tab1_episodes",
            )
            if st.button("⏩ Fast Forward", key="tab1_batch"):
                run_batch_training(n_episodes, config_tab1)
                st.rerun()

        # Metrics
        if "tab1_total_episodes" in st.session_state:
            ready = display_state.get("ready_for_episode", True)
            if ready:
                st.metric(
                    label="**Number of Episodes Trained**:",
                    value=display_state["total_episodes"],
                )
            else:
                st.metric(
                    label="**Current Training Episode**:",
                    value=display_state["total_episodes"] + 1,
                )
        st.markdown("---")
        inline_help(
            "**Controls**",
            "Use these buttons to navigate through the training history. Each action refers to each time a button is clicked, which could be either a step in an episode or a batch training of episodes.",
        )

        # Playback controls
        col_r1, col_r2, col_r3 = st.columns([1.2, 1, 1])
        with col_r1:
            checkpoints = st.session_state.get("tab1_checkpoints", [])
            playback_idx = st.session_state.get("tab1_playback_index", -1)
            at_first_checkpoint = playback_idx == 0
            prev_disabled = len(checkpoints) == 0 or at_first_checkpoint
            if st.button("⏮️ Previous action", key="tab1_rewind", disabled=prev_disabled):
                rewind_checkpoint(config_tab1)
                st.rerun()
        with col_r2:
            if st.button("⏭️ Next action", key="tab1_forward", disabled=not in_playback):
                forward_checkpoint(config_tab1)
                st.rerun()
        with col_r3:
            if st.button("⏩ Latest action", key="tab1_latest", disabled=not in_playback):
                jump_to_latest(config_tab1)
                st.rerun()
        st.markdown("---")

    with col_info:
        render_bellman_log(display_state["step_log"])

    # --- D. Q-TABLE & PLOTS ---
    col_q, col_plot = st.columns([1, 2])
    with col_q:
        st.subheader("Current Q-Matrix")
        st.dataframe(
            display_state["q_table"].style.highlight_max(axis=1, color="lightgreen"),
            use_container_width=True,
        )
    with col_plot:
        st.subheader("Evolving Q-Values")
        if display_state["q_history_plot"]:
            import pandas as pd

            hist_df = pd.DataFrame(display_state["q_history_plot"])
            if not hist_df.empty:
                hist_df = hist_df.set_index("Episode")
                st.line_chart(hist_df)
        else:
            st.write("No history yet.")

