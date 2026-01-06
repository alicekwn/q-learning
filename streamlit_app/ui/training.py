"""Training controls for Q-learning demo."""

from __future__ import annotations

import streamlit as st
from streamlit_app.state import jump_to_latest
from streamlit_app.ui.controls import inline_help, playback_controls
from streamlit_app.state import is_in_playback_mode, get_display_state

__all__ = ["render_training_controls"]


def render_training_controls(
    config: dict,
    display_state: dict,
    in_playback: bool,
    reset_episode_fn,
    step_agent_fn,
    run_batch_training_fn,
) -> None:
    """Render training controls (buttons, metrics, help, playback).

    Args:
        config: Configuration dict with 'tab_id' key
        display_state: Display state dict with 'ready_for_episode', 'is_terminal', 'total_episodes'
        in_playback: Whether currently in playback mode
        reset_episode_fn: Function to reset episode (takes config)
        step_agent_fn: Function to step agent (takes config)
        run_batch_training_fn: Function to run batch training (takes episodes, config)
    """
    tab_id = config.get("tab_id", "default")
    ready = display_state.get("ready_for_episode", True)

    # Metrics
    total_episodes_key = f"{tab_id}_total_episodes"
    if total_episodes_key in st.session_state:
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

    # Ready state: show "Train a new episode" button
    if ready:
        # Show success message if goal reached
        is_terminal_key = f"{tab_id}_is_terminal"
        is_terminal = display_state.get(
            "is_terminal", st.session_state.get(is_terminal_key, True)
        )
        if is_terminal and display_state["total_episodes"] > 0:
            st.success(
                f"🎉 Goal Reached! Episode {display_state['total_episodes']} complete."
            )

        if st.button(
            "Train a new episode step by step",
            key=f"{tab_id}_new_episode",
            disabled=in_playback,
        ):
            jump_to_latest(config)
            reset_episode_fn(config)
            st.rerun()
    else:
        # Not ready: show "Take Next Step" button
        if st.button("👟 Take Next Step", key=f"{tab_id}_step", disabled=in_playback):
            jump_to_latest(config)
            step_agent_fn(config)
            st.rerun()

    # Fast forward section
    if ready and not in_playback:
        st.markdown("Or")
        n_episodes = st.number_input(
            "Speed up training by running this many episodes:",
            min_value=1,
            value=1,
            key=f"{tab_id}_episodes",
        )
        if st.button("⏩ Fast Forward", key=f"{tab_id}_batch"):
            run_batch_training_fn(n_episodes, config)
            st.rerun()

    st.markdown("---")

    # Playback controls
    inline_help(
        "**Playback Controls**",
        "Use these buttons to navigate through the training history. Each action refers to each time a button is clicked, which could be either a step in an episode or a batch training of episodes.",
    )
    playback_controls(config, in_playback)


def playback_indicator(display_state: dict, in_playback: bool) -> None:
    """
    Check if the playback mode is active and display the appropriate indicator.
    """
    if in_playback:
        action = display_state.get("action_type", "unknown")
        meta = display_state.get("metadata", {})
        if action == "batch":
            episodes_trained = meta.get("episodes", 0)
            total_episodes = display_state.get("total_episodes", 0)
            if episodes_trained > 0 and total_episodes >= episodes_trained:
                # Calculate episode range: start = total - episodes + 1, end = total
                start_episode = total_episodes - episodes_trained + 1
                end_episode = total_episodes
                if start_episode == end_episode:
                    desc = f"Batch Trained {meta.get('episodes', '0')} episodes (episode {start_episode})"
                else:
                    desc = f"Batch Trained {meta.get('episodes', '0')} episodes (episodes {start_episode}-{end_episode})"
            else:
                desc = f"Batch Trained ({episodes_trained} episodes)"
        else:
            desc = f"Episode {meta.get('episode', '0')}, Step {meta.get('step', '0')}"
        st.warning(f"⏪ **Playback Mode** - {desc}")
