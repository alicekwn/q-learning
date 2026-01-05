"""Dog & Bone Q-learning demo page with 1D Line and 2D Grid tabs."""
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
st.set_page_config(page_title="The Dog & The Bone – Q-Learning", layout="wide")

ROOT = Path(__file__).resolve().parent.parent.parent  # project root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from streamlit_app.ui.controls import tab_controls_horizontal, tab_controls_2d_horizontal, inline_help
from streamlit_app.ui.grid import render_grid, render_grid_2d
from streamlit_app.ui.log import render_bellman_log
from streamlit_app.ui.charts import render_steps_chart
from streamlit_app.ui.policy_plot import render_policy_1d, render_policy_2d
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
    # 2D functions
    init_session_state_2d,
    reset_episode_2d,
    step_agent_2d,
    run_batch_training_2d,
)

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
    config_tab1 = tab_controls_horizontal("tab1")
    
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

    display_state = get_display_state(config_tab1)
    in_playback = is_in_playback_mode(config_tab1)

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
    # Show dog if Fixed mode OR if training has started (not ready)
    show_dog = (config_tab1["start_mode"] == "Fixed") or (not ready)
    render_grid(
        config_tab1["start_pos"],
        config_tab1["end_pos"],
        display_state["current_state"],
        config_tab1["goal_pos"],
        display_state["current_path"],
        show_path=not ready,
        show_dog=show_dog,
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
        st.subheader("Current Q-Matrix")
        st.dataframe(
            display_state["q_table"].style.highlight_max(axis=1, color="lightgreen"),
            use_container_width=True,
        )

        # --- plot vector field diagram using q matrix ---
        render_policy_1d(
            display_state["q_table"],
            config_tab1["start_pos"],
            config_tab1["end_pos"],
            config_tab1["goal_pos"]
        )

    with col_info:
        render_bellman_log(display_state["step_log"])
        # Evolving Q-Values plot (collapsible)
        with st.expander("📈 Evolving Q-Values", expanded=False):
            if display_state["q_history_plot"]:
                import pandas as pd

                hist_df = pd.DataFrame(display_state["q_history_plot"])
                if not hist_df.empty:
                    hist_df = hist_df.set_index("Episode")
                    # Downsample for display if too many data points (performance optimization)
                    if len(hist_df) > 100:
                        step = len(hist_df) // 100
                        hist_df = hist_df.iloc[::step]
                    st.line_chart(hist_df)
            else:
                st.write("No history yet.")

        # Steps per Episode plot (collapsible)
        with st.expander("📊 Steps per Episode", expanded=False):
            render_steps_chart(display_state["steps_per_episode"])



# ==============================================================================
# TAB 2: 2D Grid
# ==============================================================================
with tab_2d:
    # Top panel: Controls in horizontal layout
    config_tab2 = tab_controls_2d_horizontal("tab2")
    
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

    # Playback indicator
    if in_playback_2d:
        action = display_state_2d.get("action_type", "unknown")
        meta = display_state_2d.get("metadata", {})
        if action == "batch":
            desc = f"Batch Training ({meta.get('episodes', '?')} episodes)"
        else:
            desc = f"Episode {meta.get('episode', '?')}, Step {meta.get('step', '?')}"
        st.warning(f"⏮️ **Playback Mode** - {desc}")

    # --- A. GRID ---
    col_grid_2d, col_step_2d = st.columns([3, 1])
    with col_grid_2d:
        ready_2d = display_state_2d.get("ready_for_episode", True)
        # Show dog if Fixed mode OR if training has started (not ready)
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
        ready_2d = display_state_2d.get("ready_for_episode", True)
        if ready_2d:
            if display_state_2d.get("is_terminal", st.session_state.get("tab2_is_terminal", True)) and display_state_2d["total_episodes"] > 0:
                st.success(
                    f"🎉 Goal Reached! Episode {display_state_2d['total_episodes']} complete."
                )
            if st.button("Train a new episode step by step", key="tab2_new_episode", disabled=in_playback_2d):
                jump_to_latest(config_tab2)
                reset_episode_2d(config_tab2)
                st.rerun()
        else:
            if st.button("👟 Take Next Step", key="tab2_step", disabled=in_playback_2d):
                jump_to_latest(config_tab2)
                step_agent_2d(config_tab2)
                st.rerun()

        # Fast forward
        if ready_2d and not in_playback_2d:
            st.markdown("Or")
            n_episodes_2d = st.number_input(
                "Speed up training by running this many episodes:",
                min_value=1,
                value=1,
                key="tab2_episodes",
            )
            if st.button("⏩ Fast Forward", key="tab2_batch"):
                run_batch_training_2d(n_episodes_2d, config_tab2)
                st.rerun()

        # Metrics
        if "tab2_total_episodes" in st.session_state:
            ready_2d = display_state_2d.get("ready_for_episode", True)
            if ready_2d:
                st.metric(
                    label="**Number of Episodes Trained**:",
                    value=display_state_2d["total_episodes"],
                )
            else:
                st.metric(
                    label="**Current Training Episode**:",
                    value=display_state_2d["total_episodes"] + 1,
                )
        st.markdown("---")
        inline_help(
            "**Controls**",
            "Use these buttons to navigate through the training history. Each action refers to each time a button is clicked, which could be either a step in an episode or a batch training of episodes.",
        )

        # Playback controls
        col_r1_2d, col_r2_2d, col_r3_2d = st.columns([1.2, 1, 1])
        with col_r1_2d:
            checkpoints_2d = st.session_state.get("tab2_checkpoints", [])
            playback_idx_2d = st.session_state.get("tab2_playback_index", -1)
            at_first_checkpoint_2d = playback_idx_2d == 0
            prev_disabled_2d = len(checkpoints_2d) == 0 or at_first_checkpoint_2d
            if st.button("⏮️ Previous action", key="tab2_rewind", disabled=prev_disabled_2d):
                rewind_checkpoint(config_tab2)
                st.rerun()
        with col_r2_2d:
            if st.button("⏭️ Next action", key="tab2_forward", disabled=not in_playback_2d):
                forward_checkpoint(config_tab2)
                st.rerun()
        with col_r3_2d:
            if st.button("⏩ Latest action", key="tab2_latest", disabled=not in_playback_2d):
                jump_to_latest(config_tab2)
                st.rerun()
        st.markdown("---")

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
        with st.expander("📈 Evolving Q-Values", expanded=False):
            if display_state_2d["q_history_plot"]:
                import pandas as pd

                hist_df = pd.DataFrame(display_state_2d["q_history_plot"])
                if not hist_df.empty:
                    hist_df = hist_df.set_index("Episode")
                    # Downsample for display if too many data points (performance optimization)
                    if len(hist_df) > 100:
                        step = len(hist_df) // 100
                        hist_df = hist_df.iloc[::step]
                    st.line_chart(hist_df)
            else:
                st.write("No history yet.")
        
        # Steps per Episode plot (collapsible)
        with st.expander("📊 Steps per Episode", expanded=False):
            render_steps_chart(display_state_2d["steps_per_episode"])


    