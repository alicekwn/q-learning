"""Dog & Bone Q-learning demo page with 1D Line and 2D Grid tabs."""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="The Dog & The Bone – Q-Learning", layout="wide")

ROOT = Path(__file__).resolve().parent.parent.parent  # project root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from streamlit_app.ui.controls import parameters_1d, parameters_2d, inline_help
from streamlit_app.ui.training import render_training_controls
from streamlit_app.ui.grid import render_grid_1d, render_grid_2d
from streamlit_app.ui.bellman import render_bellman_log
from streamlit_app.ui.charts import render_q_history_chart, render_steps_chart
from streamlit_app.ui.policy_plot import render_policy_1d, render_policy_2d
from streamlit_app.state import (
    get_display_state,
    is_in_playback_mode,
    init_session_state,
    reset_episode,
    step_agent,
    run_batch_training,  # 1D
    init_session_state_2d,
    reset_episode_2d,
    step_agent_2d,
    run_batch_training_2d,  # 2D
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
    st.markdown(
        """
        <div style="text-align: left;">
        In this example, the dog is living in a 1D grid, hence the only actions it can take are to move left (L) or right (R). Note that when the dog hits the boundary of the grid, it will stay in the same position. <br>
        The state space is all the positions the dog can be on the 1D grid. <br>
        <br>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # --- A. TOP PANEL: Controls in horizontal layout ---
    config_tab1 = parameters_1d("tab1")

    # Reset button in its own row
    col_reset, col_spacer = st.columns([1, 5])
    with col_reset:
        if st.button(
            "Reset / Initialize Model",
            type="primary",
            key="tab1_reset",
            help="After changing any of the settings / parameters above, click the button to reset the model and start training.",
        ):
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

    # --- B. GRID ---
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

    # --- C. CONTROLS & DISPLAY AREA ---
    col_step, col_info = st.columns([1, 2])

    with col_step:

        st.markdown(" ")

        # Render training controls
        render_training_controls(
            config_tab1,
            display_state_1d,
            in_playback_1d,
            reset_episode,
            step_agent,
            run_batch_training,
        )

        st.markdown("---")

        # Render Q-Matrix
        st.subheader("Current Q-Matrix")
        st.dataframe(
            display_state_1d["q_table"].style.highlight_max(axis=1, color="lightgreen"),
            use_container_width=True,
        )

        # --- Plot vector field diagram using q matrix ---
        render_policy_1d(
            display_state_1d["q_table"],
            config_tab1["start_pos"],
            config_tab1["end_pos"],
            config_tab1["goal_pos"],
        )

    with col_info:
        render_bellman_log(display_state_1d["step_log"])

        # Evolving Q-Values plot and steps per episode plot (collapsible)
        with st.expander("📈 Evolving Q-Values", expanded=True):
            render_q_history_chart(display_state_1d["q_history_plot"])

        with st.expander("📊 Steps per Episode", expanded=True):
            render_steps_chart(
                display_state_1d["steps_per_episode"],
                display_state_1d["q_history_plot"],
            )

    # --- D. Q&A ---
    st.markdown("---")
    st.subheader("Q&A")
    with st.expander(
        "Q.1 Does the index position of the goal matter? (i.e. whether the Q value would change if goal is placed at index 0 or index +5) ",
        expanded=False,
    ):
        st.markdown(
            """
        The only thing that matter is the relative distance from the goal. The Q value of a state 1 step to the left of the goal should be the same as the Q value of a state 1 step to the right of the goal.
        """
        )

    with st.expander(
        "Q.2 Does the starting position of each episode matter?", expanded=False
    ):
        st.markdown(
            """
        If the starting position is fixed, and if it's on the left side of the goal position, then the dog would never explore the right side of the grid. However, if you combine two Q matrix: where one is derived from a starting position on the left of the goal, and the other is derived from a starting position on the right of the goal, then you can see the combined Q-matrix being the same as the Q-matrix when starting position is randomised.

        Hence starting position being randomised is more preferred.
        """
        )


# ==============================================================================
# TAB 2: 2D Grid
# ==============================================================================
with tab_2d:
    st.markdown(
        """
        <div style="text-align: left;">
        In this example, the dog is living in a 2D grid, hence the only actions it can take are to move up (U), down (D), left (L) or right (R). Note that when the dog hits the boundary of the grid, it will stay in the same position. <br>
        The state space is all the positions the dog can be on the 2D grid. <br>
        <br>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # --- A. TOP PANEL: Controls in horizontal layout ---
    config_tab2 = parameters_2d("tab2")

    # Reset button in its own row
    col_reset_2d, col_spacer_2d = st.columns([1, 5])
    with col_reset_2d:
        if st.button(
            "Reset / Initialize Model",
            type="primary",
            key="tab2_reset",
            help="After changing any of the settings / parameters above, click the button to reset the model and start training.",
        ):
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

    # --- B. GRID & Training Controls & Playback Controls ---
    col_spacer_2d, header_2d, col_grid_2d, col_spacer_2d = st.columns([0.2, 1, 4, 0.5])
    with header_2d:
        st.subheader("2D Grid")
    with col_grid_2d:
        ready_2d = display_state_2d.get("ready_for_episode", True)
        # Show dog if Fixed starting position mode OR if training has started (not ready)
        show_dog_2d = (config_tab2["start_mode"] == "Fixed") or (not ready_2d)
        with st.expander("Visualization", expanded=True):
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
        # Only show path when actively training
        if not ready_2d:
            st.write(f"**Current Path:** {display_state_2d['current_path']}")

    with col_spacer_2d:
        st.markdown(" ")

    st.markdown("---")
    # --- C. Q-TABLE & Q-VALUES PLOT AREA ---
    col_q_2d, col_info_2d = st.columns([1, 2])

    with col_q_2d:
        # Render training and playback controls
        render_training_controls(
            config_tab2,
            display_state_2d,
            in_playback_2d,
            reset_episode_2d,
            step_agent_2d,
            run_batch_training_2d,
        )
        st.markdown("---")
        st.subheader("Current Q-Matrix")
        st.dataframe(
            display_state_2d["q_table"].style.highlight_max(axis=1, color="lightgreen"),
            use_container_width=True,
            height="content",
        )
        # --- Plot vector field diagram using q matrix ---
        render_policy_2d(
            display_state_2d["q_table"],
            config_tab2["x_start"],
            config_tab2["x_end"],
            config_tab2["y_start"],
            config_tab2["y_end"],
            (config_tab2["goal_x"], config_tab2["goal_y"]),
        )
    with col_info_2d:
        render_bellman_log(display_state_2d["step_log"])

        # Evolving Q-Values plot (collapsible)
        with st.expander("📈 Evolving Q-Values", expanded=True):
            render_q_history_chart(display_state_2d["q_history_plot"])

        # Steps per Episode plot (collapsible)
        with st.expander("📊 Steps per Episode", expanded=True):
            render_steps_chart(
                display_state_2d["steps_per_episode"],
                display_state_2d["q_history_plot"],
            )
    # --- D. Q&A ---
    st.markdown("---")
    st.subheader("Q&A")
    with st.expander(
        "Q.1 In this example, will the Q matrix converge to different values each time the model is trained?",
        expanded=False,
    ):
        st.markdown(
            """
        No, as more episodes are trained (more than 10000 episodes in this case), the Q matrix will converge to the same values each time the model is trained, regardless of the starting position. At first, when less episodes are trained, the Q matrix might show only one optimal action for each state. But as the model learns, there might be 2 optimal actions for some states, indicating that either path is optimal. 
        """
        )
