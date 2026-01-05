"""Dog & Bone Q-learning demo page with 1D Line and 2D Grid tabs."""
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
st.set_page_config(page_title="The Dog & The Bone – Q-Learning", layout="wide")

ROOT = Path(__file__).resolve().parent.parent.parent  # project root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from streamlit_app.ui.controls import parameters_1d, parameters_2d
from streamlit_app.ui.training import render_training_controls
from streamlit_app.ui.grid import render_grid_1d, render_grid_2d
from streamlit_app.ui.bellman import render_bellman_log
from streamlit_app.ui.charts import render_q_history_chart, render_steps_chart
from streamlit_app.ui.policy_plot import render_policy_1d, render_policy_2d
from streamlit_app.state import (
    init_session_state,
    reset_episode,
    step_agent,
    run_batch_training,
    get_display_state,
    is_in_playback_mode,
    # 2D functions
    init_session_state_2d,
    reset_episode_2d,
    step_agent_2d,
    run_batch_training_2d,
)
from streamlit_app.ui.training import playback_indicator

st.title("The Dog & The Bone")

st.markdown(
    """
    In this demo, we use Q-learning to help the dog 🐶 learn how to find the bone 🍖! 
    Change the parameters to see how they affect learning, and use the controls to step through the process.
    """
)

# Create tabs
tab_1d, tab_2d = st.tabs(["1D Grid", "2D Grid"])

# ==============================================================================
# TAB 1: 1D Line
# ==============================================================================
with tab_1d:
    # Top panel: Controls in horizontal layout
    config_tab1 = parameters_1d("tab1")
    
    # Reset button in its own row
    col_reset, col_spacer = st.columns([1, 5])
    with col_reset:
        if st.button("Reset / Initialize Model", type="primary", key="tab1_reset"):
            init_session_state(config_tab1)
            st.rerun()

    st.markdown("---")

    # Initialize on first load
    if "tab1_q_table" not in st.session_state:
        init_session_state(config_tab1)


    display_state_1d = get_display_state(config_tab1)
    in_playback_1d = is_in_playback_mode(config_tab1)

    # Render playback indicator
    playback_indicator(display_state_1d, in_playback_1d) 

    # --- A. GRID ---
    ready = display_state_1d.get("ready_for_episode", True)
    # Show dog if Fixed mode OR if training has started (not ready)
    show_dog = (config_tab1["start_mode"] == "Fixed") or (not ready)
    render_grid_1d(
        config_tab1["start_pos"],
        config_tab1["end_pos"],
        display_state_1d["current_state"],
        config_tab1["goal_pos"],
        display_state_1d["current_path"],
        show_path=not ready,
        show_dog=show_dog,
    )

    # --- B. CONTROLS AREA ---
    col_step, col_info = st.columns([1, 2])

    with col_step:
        # Training controls
        render_training_controls(
            config_tab1,
            display_state_1d,
            in_playback_1d,
            reset_episode,
            step_agent,
            run_batch_training,
        )
        
        st.markdown("---")

        # Q-Matrix
        st.subheader("Current Q-Matrix")
        st.dataframe(
            display_state_1d["q_table"].style.highlight_max(axis=1, color="lightgreen"),
            use_container_width=True,
        )

        # --- plot vector field diagram using q matrix ---
        render_policy_1d(
            display_state_1d["q_table"],
            config_tab1["start_pos"],
            config_tab1["end_pos"],
            config_tab1["goal_pos"]
        )

    with col_info:
        render_bellman_log(display_state_1d["step_log"])
        # Evolving Q-Values plot (collapsible)
        with st.expander("📈 Evolving Q-Values", expanded=True):
            render_q_history_chart(display_state_1d["q_history_plot"])

        # Steps per Episode plot (collapsible)
        with st.expander("📊 Steps per Episode", expanded=True):
            render_steps_chart(display_state_1d["steps_per_episode"])

# ==============================================================================
# TAB 2: 2D Grid
# ==============================================================================
with tab_2d:
    # Top panel: Controls in horizontal layout
    config_tab2 = parameters_2d("tab2")
    
    # Reset button in its own row
    col_reset_2d, col_spacer_2d = st.columns([1, 5])
    with col_reset_2d:
        if st.button("Reset / Initialize Model", type="primary", key="tab2_reset"):
            init_session_state_2d(config_tab2)
            st.rerun()

    st.markdown("---")

    # Initialize on first load
    if "tab2_q_table" not in st.session_state:
        init_session_state_2d(config_tab2)

    display_state_2d = get_display_state(config_tab2)
    in_playback_2d = is_in_playback_mode(config_tab2)

    # Render playback indicator
    playback_indicator(display_state_2d, in_playback_2d)

    # --- A. GRID & Training Controls & Playback Controls ---
    col_grid_2d, col_step_2d = st.columns([3, 1])
    with col_grid_2d:
        ready_2d = display_state_2d.get("ready_for_episode", True)
        # Show dog if Fixed starting position mode OR if training has started (not ready)
        show_dog_2d = (config_tab2["start_mode"] == "Fixed") or (not ready_2d)
        render_grid_2d(
            config_tab2["x_start"],
            config_tab2["x_end"],
            config_tab2["y_start"],
            config_tab2["y_end"],
            display_state_2d["current_state"],
            (config_tab2["goal_x"], config_tab2["goal_y"]),
            display_state_2d["current_path"],
            show_path=not ready_2d,
            show_dog=show_dog_2d,
        )

    with col_step_2d:
        # Render training and playback controls
        render_training_controls(
            config_tab2,
            display_state_2d,
            in_playback_2d,
            reset_episode_2d,
            step_agent_2d,
            run_batch_training_2d,
        )

    # --- B. Q-TABLE & Q-VALUES PLOT AREA ---
    col_q_2d, col_info_2d = st.columns([1, 2])

    with col_q_2d:
        st.subheader("Current Q-Matrix")
        st.dataframe(
            display_state_2d["q_table"].style.highlight_max(axis=1, color="lightgreen"),
            use_container_width=True, 
            height="content",
        )
        # --- plot vector field diagram using q matrix ---
        render_policy_2d(
            display_state_2d["q_table"],
            config_tab2["x_start"],
            config_tab2["x_end"],
            config_tab2["y_start"],
            config_tab2["y_end"],
            (config_tab2["goal_x"], config_tab2["goal_y"])
        )

    with col_info_2d:
        render_bellman_log(display_state_2d["step_log"])

        # Evolving Q-Values plot (collapsible)
        with st.expander("📈 Evolving Q-Values", expanded=True):
            render_q_history_chart(display_state_2d["q_history_plot"])
        
        # Steps per Episode plot (collapsible)
        with st.expander("📊 Steps per Episode", expanded=True):
            render_steps_chart(display_state_2d["steps_per_episode"])


