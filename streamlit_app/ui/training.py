"""Training controls for Q-learning demo."""

from __future__ import annotations

import time
import streamlit as st
from streamlit_app.state import jump_to_latest
from streamlit_app.ui.controls import (
    inline_help,
    playback_controls,
    playback_controls_econ,
)
from streamlit_app.state_econ import (
    jump_to_latest_econ,
    pick_random_starting_prices_econ,
)

__all__ = ["render_training_controls", "render_training_controls_econ"]


def handle_autoplay(
    config: dict,
    _display_state: dict,  # Unused but kept for API consistency
    step_agent_fn,
) -> None:
    """Handle autoplay logic: check if autoplay is active and take steps automatically.

    Args:
        config: Configuration dict with 'tab_id' key
        _display_state: Display state dict (unused but kept for API consistency)
        step_agent_fn: Function to step agent (takes config)
    """
    tab_id = config.get("tab_id", "default")
    autoplay_key = f"{tab_id}_autoplay_active"
    autoplay_last_step_key = f"{tab_id}_autoplay_last_step_time"
    autoplay_pending_rerun_key = f"{tab_id}_autoplay_pending_rerun"
    autoplay_delay = 0.5  # seconds between steps

    # Check if autoplay is active
    if st.session_state.get(autoplay_key, False):
        # Check if enough time has passed since last step
        current_time = time.time()
        last_step_time = st.session_state.get(autoplay_last_step_key, 0)
        time_since_last_step = current_time - last_step_time

        if time_since_last_step < autoplay_delay:
            time.sleep(autoplay_delay - time_since_last_step)

        # Take exactly one step per run so the grid and log update together
        # This step will move the agent closer to or reach the goal
        jump_to_latest(config)
        step_agent_fn(config)

        # Update last step time
        st.session_state[autoplay_last_step_key] = time.time()

        # Check if episode is now complete after taking the step
        # (step_agent_fn may have completed the episode and reached the goal)
        ready_after_step = st.session_state.get(f"{tab_id}_ready_for_episode", True)
        is_terminal_after_step = st.session_state.get(f"{tab_id}_is_terminal", True)

        if ready_after_step and is_terminal_after_step:
            # Episode completed and goal reached, stop autoplay
            st.session_state[autoplay_key] = False
            if autoplay_last_step_key in st.session_state:
                del st.session_state[autoplay_last_step_key]
            # Trigger one final rerun to ensure the completed state with goal reached is displayed
            # This ensures the user sees "🎉 Goal Reached! Episode xx complete." message
            st.session_state[autoplay_pending_rerun_key] = True
        else:
            # Episode not complete yet, continue autoplay on next rerun
            # The step was taken, so we need another rerun to continue
            st.session_state[autoplay_pending_rerun_key] = True


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
    autoplay_key = f"{tab_id}_autoplay_active"
    ready = display_state.get("ready_for_episode", True)

    # Handle autoplay if active (before rendering buttons)
    # This needs to happen early so autoplay can continue on each rerun
    # Check if autoplay is active (not just if ready=False) so autoplay can complete the final step
    if st.session_state.get(autoplay_key, False):
        handle_autoplay(config, display_state, step_agent_fn)
        # After handle_autoplay, ready state may have changed, so refresh it from session state
        ready = st.session_state.get(f"{tab_id}_ready_for_episode", True)

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
            # Stop autoplay when starting a new episode
            autoplay_key = f"{tab_id}_autoplay_active"
            autoplay_last_step_key = f"{tab_id}_autoplay_last_step_time"
            if autoplay_key in st.session_state:
                st.session_state[autoplay_key] = False
            if autoplay_last_step_key in st.session_state:
                del st.session_state[autoplay_last_step_key]
            jump_to_latest(config)
            reset_episode_fn(config)
            st.rerun()
    else:
        col1, col2 = st.columns(2)
        with col1:
            # Not ready: show "Take Next Step" button
            if st.button(
                "👟 Take Next Step", key=f"{tab_id}_step", disabled=in_playback
            ):
                # Stop autoplay if manually stepping
                autoplay_key = f"{tab_id}_autoplay_active"
                autoplay_last_step_key = f"{tab_id}_autoplay_last_step_time"
                if autoplay_key in st.session_state:
                    st.session_state[autoplay_key] = False
                if autoplay_last_step_key in st.session_state:
                    del st.session_state[autoplay_last_step_key]
                jump_to_latest(config)
                step_agent_fn(config)
                st.rerun()
        with col2:
            autoplay_key = f"{tab_id}_autoplay_active"
            if st.button(
                "Autoplay to complete this episode",
                key=f"{tab_id}_autoplay",
                disabled=in_playback,
            ):
                # Start autoplay
                autoplay_last_step_key = f"{tab_id}_autoplay_last_step_time"
                autoplay_pending_rerun_key = f"{tab_id}_autoplay_pending_rerun"
                st.session_state[autoplay_key] = True
                # Reset timer and take first step immediately
                jump_to_latest(config)
                step_agent_fn(config)
                st.session_state[autoplay_last_step_key] = time.time()
                ready_after_step = st.session_state.get(
                    f"{tab_id}_ready_for_episode", True
                )
                st.session_state[autoplay_pending_rerun_key] = not ready_after_step

    # Fast forward section
    if ready and not in_playback:
        st.markdown("Or")
        n_episodes = st.number_input(
            "Speed up training by running this many episodes:",
            min_value=1,
            value=1,
            key=f"{tab_id}_episodes",
            help="Note that you can't fast forward episodes in the middle of training an episode step by step. This feature is only available when a complete episode is trained.",
        )
        if st.button("⏩ Fast Forward", key=f"{tab_id}_batch"):
            # Stop autoplay when fast forwarding
            autoplay_key = f"{tab_id}_autoplay_active"
            autoplay_last_step_key = f"{tab_id}_autoplay_last_step_time"
            if autoplay_key in st.session_state:
                st.session_state[autoplay_key] = False
            if autoplay_last_step_key in st.session_state:
                del st.session_state[autoplay_last_step_key]
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


def render_training_controls_econ(
    config: dict,
    display_state: dict,
    in_playback: bool,
    step_agent_fn,
    run_batch_training_fn,
    run_until_convergence_fn,
) -> None:
    """Render training controls for economics pricing (buttons, metrics, help, playback).

    Args:
        config: Configuration dict with 'tab_id' key
        display_state: Display state dict with 'ready_for_training', 'step_count', 'convergence_info'
        in_playback: Whether currently in playback mode
        step_agent_fn: Function to step agent (takes config)
        run_batch_training_fn: Function to run batch training (takes steps, config)
        run_until_convergence_fn: Function to run until convergence (takes config)
    """
    tab_id = config.get("tab_id", "default")
    ready = display_state.get("ready_for_training", True)
    step_count = display_state.get("step_count", 0)
    convergence_info = display_state.get("convergence_info")
    starting_prices_picked = display_state.get("starting_prices_picked", True)
    start_mode = config.get("start_mode", "Randomised")

    # Check if we're at the latest checkpoint (should behave like live mode)
    checkpoints = st.session_state.get(f"{tab_id}_checkpoints", [])
    latest_idx = st.session_state.get(
        f"{tab_id}_latest_checkpoint_index", len(checkpoints) - 1 if checkpoints else -1
    )
    latest_idx = min(latest_idx, len(checkpoints) - 1) if checkpoints else -1
    playback_idx = st.session_state.get(f"{tab_id}_playback_index", -1)
    at_latest_checkpoint = (
        in_playback and playback_idx >= latest_idx if latest_idx >= 0 else False
    )

    # If at latest checkpoint, treat as live mode for button display
    effective_in_playback = in_playback and not at_latest_checkpoint

    # Check if we need to pick starting prices first (randomized mode)
    if start_mode == "Randomised" and not starting_prices_picked:
        if st.button(
            "🎲 Randomly pick starting price pair",
            key=f"{tab_id}_pick_starting",
            disabled=effective_in_playback,
            type="primary",
        ):
            jump_to_latest_econ(config)
            pick_random_starting_prices_econ(config)
            st.rerun()
        st.info("Pick a random starting price pair to begin training.")
        return

    # Metrics
    st.metric(
        label="**Total Steps**:",
        value=step_count,
    )

    # Show convergence info if available
    if convergence_info:
        if convergence_info.get("converged", False):
            st.success(
                f"✅ **Converged!** After {convergence_info['periods_run']:,} steps, "
                f"policy was stable for {convergence_info['stable_periods']:,} steps. "
                f"Final ε = {convergence_info['epsilon_final']:.6f}"
            )
        else:
            st.warning(
                f"⚠️ **Not converged** after {convergence_info['periods_run']:,} steps. "
                f"Stable for {convergence_info['stable_periods']:,} steps. "
                f"Final ε = {convergence_info['epsilon_final']:.6f}"
            )

    # Training controls
    if not ready or convergence_info is None:
        # Show "Take Next Step" button (disabled only if in playback mode but not at latest)
        if st.button(
            "👟 Take Next Step", key=f"{tab_id}_step", disabled=effective_in_playback
        ):
            jump_to_latest_econ(config)
            step_agent_fn(config)
            st.rerun()

    # Fast forward section (only when ready and effectively not in playback)
    if ready and not effective_in_playback:
        st.markdown("---")
        st.markdown("**Fast Forward Options:**")

        # Option 1: Fast forward N steps
        n_steps = st.number_input(
            "Fast forward this many steps:",
            min_value=1,
            max_value=100000,
            value=1000,
            step=100,
            key=f"{tab_id}_fast_steps",
            help="Run N steps without showing intermediate updates.",
        )

        if st.button("⏩ Fast Forward N Steps", key=f"{tab_id}_batch_steps"):
            jump_to_latest_econ(config)
            run_batch_training_fn(n_steps, config)
            st.rerun()

        # Option 2: Fast forward until convergence
        if st.button(
            "⏯️ Fast Forward Until Convergence",
            key=f"{tab_id}_converge",
            help="Run until policy is stable for stable_required steps (or max_periods reached).",
        ):
            jump_to_latest_econ(config)
            run_until_convergence_fn(config)
            st.rerun()

    st.markdown("---")

    # Playback controls
    inline_help(
        "**Playback Controls**",
        "Use these buttons to navigate through the training history. Each action refers to each time a button is clicked, which could be either a single step or a batch of steps.",
    )
    playback_controls_econ(config, in_playback)
