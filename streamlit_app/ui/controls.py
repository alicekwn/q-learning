"""Streamlit sidebar & control widgets for dog and bone demo."""

from __future__ import annotations
from streamlit_app.state import (
    rewind_checkpoint,
    forward_checkpoint,
    jump_to_latest,
    jump_to_start,
)

import streamlit as st

__all__ = [
    "inline_help",
    "parameters_1d",
    "parameters_2d",
    "parameters_econ",
    "playback_controls",
    "playback_controls_econ",
]


def inline_help(text: str, help_text: str) -> None:
    """Display text with an inline help icon"""
    st.markdown(
        f"""
    <style>
    .tooltip-container {{
        position: relative;
        display: inline-block;
    }}
    .tooltip-container .tooltip-text {{
        visibility: hidden;
        width: 220px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
        font-weight: normal;
    }}
    .tooltip-container:hover .tooltip-text {{
        visibility: visible;
        opacity: 1;
    }}
    </style>
    {text}
    <span class="tooltip-container">
        <span style="
            display: inline-block;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background-color: #6c757d;
            color: white;
            text-align: center;
            line-height: 16px;
            font-size: 11px;
            cursor: help;
            margin-left: 4px;
            vertical-align: middle;
        ">?</span>
        <span class="tooltip-text">{help_text}</span>
    </span>
    """,
        unsafe_allow_html=True,
    )


def q_learning_parameters(tab_id: str) -> dict:
    """Render Q-learning parameters in horizontal layout with two expander rows.

    Returns a dict with keys:
      alpha, gamma, epsilon, reward_val, tab_id
    """

    col_a, col_g, col_e, col_r = st.columns([1, 1, 1, 1])

    with col_a:
        alpha: float = st.slider(
            r"$\alpha$ (Learning Rate)", 0.0, 1.0, 0.5, 0.01, key=f"{tab_id}_alpha"
        )
    with col_g:
        gamma: float = st.slider(
            r"$\gamma$ (Discount Factor)", 0.0, 1.0, 0.9, 0.01, key=f"{tab_id}_gamma"
        )
    with col_e:
        epsilon: float = st.slider(
            r"$\epsilon$ (Exploration Rate)",
            0.0,
            1.0,
            0.2,
            0.01,
            key=f"{tab_id}_epsilon",
        )
    with col_r:
        reward_val: float = st.number_input(
            r"Reward Value $r$", value=1.0, key=f"{tab_id}_reward"
        )

    return {
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "reward_val": reward_val,
    }


def parameters_1d(tab_id: str) -> dict:
    """Render 1D controls in horizontal layout with two expander rows.

    Returns a dict with keys:
      start_pos, end_pos, goal_pos, start_mode, fixed_start_pos,
      alpha, gamma, epsilon, reward_val, tab_id
    """
    # Row 1: Environment Settings
    with st.expander("🐶 Environment Settings", expanded=True):
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

        with col1:
            start_pos: int = st.number_input(
                "Start Position",
                min_value=-10,
                max_value=10,
                value=0,
                key=f"{tab_id}_start_pos",
            )
        with col2:
            end_pos: int = st.number_input(
                "End Position",
                min_value=-10,
                max_value=10,
                value=5,
                key=f"{tab_id}_end_pos",
            )

        # Ensure end_pos > start_pos
        if end_pos <= start_pos:
            st.error("End Position must be greater than Start Position!")
            end_pos = start_pos + 1

        with col3:
            goal_pos: int = st.number_input(
                "Goal Position (Bone)",
                min_value=start_pos,
                max_value=end_pos,
                value=end_pos - 1,
                key=f"{tab_id}_goal_pos",
            )

        with col4:
            start_mode: str = st.radio(
                "Starting position for each episode:",
                ["Randomised", "Fixed"],
                index=0,
                key=f"{tab_id}_start_mode",
                horizontal=True,
                help="Choose whether the dog starts at the same fixed position or at a random position for each episode.",
            )

        with col5:
            fixed_start_pos: int = start_pos
            if start_mode == "Fixed":
                fixed_start_pos = st.number_input(
                    "Fixed Start Position",
                    min_value=start_pos,
                    max_value=end_pos,
                    value=start_pos,
                    key=f"{tab_id}_fixed_start",
                    help="When the starting position is fixed, the dog will start at the same position for each episode.",
                )
            else:
                st.info("The starting position is chosen at random for each episode.")

    # Row 2: Q-Learning Parameters
    with st.expander("🧠 Q-Learning Parameters", expanded=True):
        q_learning_params = q_learning_parameters(tab_id)

    return {
        "tab_id": tab_id,
        "start_pos": start_pos,
        "end_pos": end_pos,
        "goal_pos": goal_pos,
        "start_mode": start_mode,
        "fixed_start_pos": fixed_start_pos,
        "alpha": q_learning_params["alpha"],
        "gamma": q_learning_params["gamma"],
        "epsilon": q_learning_params["epsilon"],
        "reward_val": q_learning_params["reward_val"],
    }


def parameters_2d(tab_id: str) -> dict:
    """Render 2D controls in horizontal layout with two expander rows.

    Returns a dict with keys:
      x_start, x_end, y_start, y_end, goal_x, goal_y, start_mode,
      fixed_start_x, fixed_start_y, alpha, gamma, epsilon, reward_val, tab_id
    """
    # Row 1: Environment Settings
    with st.expander("🐶 Environment Settings", expanded=True):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            st.markdown(
                """
                <div style="text-align: left;">
                Choose the X axis range of the grid: <br>
                <br>
                </div>
                """,
                unsafe_allow_html=True,
            )
            x_start: int = st.number_input(
                "X Start", min_value=-10, max_value=10, value=0, key=f"{tab_id}_x_start"
            )
            x_end: int = st.number_input(
                "X End", min_value=-10, max_value=10, value=3, key=f"{tab_id}_x_end"
            )
        if x_end <= x_start:
            st.error("X End > X Start!")
            x_end = x_start + 1

        with col2:
            st.markdown(
                """
                <div style="text-align: left;">
                Choose the Y axis range of the grid: <br>
                <br>
                </div>
                """,
                unsafe_allow_html=True,
            )
            y_start: int = st.number_input(
                "Y Start", min_value=-10, max_value=10, value=0, key=f"{tab_id}_y_start"
            )
            y_end: int = st.number_input(
                "Y End", min_value=-10, max_value=10, value=3, key=f"{tab_id}_y_end"
            )

        if y_end <= y_start:
            st.error("Y End > Y Start!")
            y_end = y_start + 1

        with col3:
            st.markdown(
                """
                <div style="text-align: left;">
                Choose the position of the bone (Goal): <br>
                <br>
                </div>
                """,
                unsafe_allow_html=True,
            )
            goal_x: int = st.number_input(
                "Goal position (X axis)",
                min_value=x_start,
                max_value=x_end,
                value=x_end - 1,
                key=f"{tab_id}_goal_x",
            )
            goal_y: int = st.number_input(
                "Goal position (Y axis)",
                min_value=y_start,
                max_value=y_end,
                value=y_end - 1,
                key=f"{tab_id}_goal_y",
            )

        with col4:
            start_mode: str = st.radio(
                "Choose the starting position for each episode:",
                ["Randomised", "Fixed"],
                index=0,
                key=f"{tab_id}_start_mode",
                horizontal=True,
                help="Choose whether the dog starts at the same fixed position or a random position for each episode.",
            )

            fixed_start_x: int = x_start
            fixed_start_y: int = y_start
            if start_mode == "Fixed":
                fixed_start_x = st.number_input(
                    "Start X",
                    min_value=x_start,
                    max_value=x_end,
                    value=x_start,
                    key=f"{tab_id}_fixed_start_x",
                )
                fixed_start_y = st.number_input(
                    "Start Y",
                    min_value=y_start,
                    max_value=y_end,
                    value=y_start,
                    key=f"{tab_id}_fixed_start_y",
                )
            else:
                st.info("The starting position is chosen at random for each episode.")

    # Row 2: Q-Learning Parameters
    with st.expander("🧠 Q-Learning Parameters", expanded=True):
        q_learning_params = q_learning_parameters(tab_id)

    return {
        "tab_id": tab_id,
        "x_start": x_start,
        "x_end": x_end,
        "y_start": y_start,
        "y_end": y_end,
        "goal_x": goal_x,
        "goal_y": goal_y,
        "start_mode": start_mode,
        "fixed_start_x": fixed_start_x,
        "fixed_start_y": fixed_start_y,
        "alpha": q_learning_params["alpha"],
        "gamma": q_learning_params["gamma"],
        "epsilon": q_learning_params["epsilon"],
        "reward_val": q_learning_params["reward_val"],
    }


def parameters_econ(tab_id: str) -> dict:
    """Render economics pricing controls in horizontal layout with two expander rows.

    Returns a dict with keys:
      tab_id, k1, k2, c, m, alpha, delta, beta, seed, check_every, stable_required, max_periods
    """
    from streamlit_app.state_econ import calculate_prices

    # Row 1: Environment Settings
    with st.expander("💰 Environment Settings", expanded=True):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            k1: float = st.number_input(
                r"$k_1$",
                min_value=0.1,
                max_value=20.0,
                value=7.0,
                step=0.1,
                key=f"{tab_id}_k1",
                help="Parameter in demand functions: q1 = k1 - p1 + k2 * p2",
            )
        with col2:
            k2: float = st.number_input(
                r"$k_2$",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                key=f"{tab_id}_k2",
                help="Cross-price parameter in demand functions",
            )
        with col3:
            c: float = st.number_input(
                r"$c$ (Marginal Cost)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.1,
                key=f"{tab_id}_c",
                help="Marginal cost for both players",
            )
        with col4:
            m_options = [4, 7, 10, 13, 16]
            m: int = st.select_slider(
                r"$m$ (Action space size)",
                options=m_options,
                value=7,
                key=f"{tab_id}_m",
                help="Number of equally spaced prices in the action space. Must be of form 4+3σ for σ=0,1,2,... (i.e., 4, 7, 10, 13, 16,...)",
            )

        # Calculate and display key prices
        PRICES = None
        try:
            PRICES, p_e, p_c = calculate_prices(k1, k2, c, m)
            price_start = PRICES[0]
            price_end = PRICES[-1]

            # Format action space for display (round each price to 1 decimal place)
            prices_display = [f"{p:.1f}" for p in PRICES]
            st.info(
                f"**Equilibrium price** $p_e = {p_e:.2f}$ | "
                f"**Collusion price** $p_c = {p_c:.2f}$ | "
                f"**Price range**: [{price_start:.2f}, {price_end:.2f}] | "
                f"**Action space** $A$ = {{{', '.join(prices_display)}}}"
            )
        except Exception as e:
            st.error(f"Error calculating prices: {e}")
            # Use defaults
            PRICES = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            p_e = 6.0
            p_c = 8.0

        # Starting prices section
        st.markdown("---")
        start_mode: str = st.radio(
            "Starting prices for initialization:",
            ["Randomised", "Fixed"],
            index=0,
            key=f"{tab_id}_start_mode",
            horizontal=True,
            help="Choose whether starting prices are randomized or fixed for each initialization.",
        )

        # Default fixed prices (middle of range)
        default_p1 = PRICES[len(PRICES) // 2] if PRICES else 7.0
        default_p2 = PRICES[len(PRICES) // 2] if PRICES else 7.0
        fixed_start_p1: float = default_p1
        fixed_start_p2: float = default_p2

        if start_mode == "Fixed":
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                fixed_start_p1 = st.number_input(
                    "Fixed starting price for Alice (p1):",
                    min_value=float(PRICES[0]) if PRICES else 4.0,
                    max_value=float(PRICES[-1]) if PRICES else 10.0,
                    value=float(default_p1),
                    step=0.1,
                    key=f"{tab_id}_fixed_start_p1",
                    help="When fixed, Alice will always start at this price.",
                )
            with col_p2:
                fixed_start_p2 = st.number_input(
                    "Fixed starting price for Bob (p2):",
                    min_value=float(PRICES[0]) if PRICES else 4.0,
                    max_value=float(PRICES[-1]) if PRICES else 10.0,
                    value=float(default_p2),
                    step=0.1,
                    key=f"{tab_id}_fixed_start_p2",
                    help="When fixed, Bob will always start at this price.",
                )

            # Validate that fixed prices are in PRICES
            if PRICES:
                if fixed_start_p1 not in PRICES:
                    st.warning(
                        f"⚠️ p1 must be in the action space: {[f'{p:.1f}' for p in PRICES]}"
                    )
                if fixed_start_p2 not in PRICES:
                    st.warning(
                        f"⚠️ p2 must be in the action space: {[f'{p:.1f}' for p in PRICES]}"
                    )
        else:
            st.info(
                "Starting prices will be chosen randomly from the action space for each initialization."
            )

    # Row 2: Q-Learning Parameters
    with st.expander("🧠 Q-Learning Parameters", expanded=True):
        col_a, col_d, col_b = st.columns([1, 1, 1])

        with col_a:
            alpha: float = st.slider(
                r"$\alpha$ (Learning Rate)",
                0.0,
                1.0,
                0.125,
                0.001,
                key=f"{tab_id}_alpha",
            )
        with col_d:
            delta: float = st.slider(
                r"$\delta$ (Discount Factor)",
                0.0,
                1.0,
                0.95,
                0.01,
                key=f"{tab_id}_delta",
            )
        with col_b:
            beta: float = st.number_input(
                r"$\beta$ (Exponential Decay Rate)",
                min_value=0.0,
                max_value=1e-3,
                value=2e-5,
                step=1e-6,
                format="%.6f",
                key=f"{tab_id}_beta",
                help="Exploration rate decays as ε_t = exp(-β * t)",
            )

    # Advanced parameters (collapsed by default)
    with st.expander("⚙️ Advanced Parameters", expanded=False):
        col_seed, col_check, col_stable, col_max = st.columns([1, 1, 1, 1])

        with col_seed:
            seed: int = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=1000,
                value=43,
                key=f"{tab_id}_seed",
            )
        with col_check:
            check_every: int = st.number_input(
                "Check Every",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                key=f"{tab_id}_check_every",
                help="Check policy stability every N steps",
            )
        with col_stable:
            stable_required: int = st.number_input(
                "Stable Required",
                min_value=1000,
                max_value=1000000,
                value=100000,
                step=10000,
                key=f"{tab_id}_stable_required",
                help="Number of stable steps required for convergence",
            )
        with col_max:
            max_periods: int = st.number_input(
                "Max Periods",
                min_value=10000,
                max_value=10000000,
                value=2000000,
                step=100000,
                key=f"{tab_id}_max_periods",
                help="Maximum number of steps before stopping",
            )

    return {
        "tab_id": tab_id,
        "k1": k1,
        "k2": k2,
        "c": c,
        "m": m,
        "alpha": alpha,
        "delta": delta,
        "beta": beta,
        "seed": seed,
        "check_every": check_every,
        "stable_required": stable_required,
        "max_periods": max_periods,
        "start_mode": start_mode,
        "fixed_start_p1": fixed_start_p1,
        "fixed_start_p2": fixed_start_p2,
    }


def playback_controls(config: dict, in_playback: bool) -> None:
    """Render playback controls for navigating through training history.

    Args:
        config: Configuration dict with 'tab_id' key
        in_playback: Whether currently in playback mode
    """
    tab_id = config.get("tab_id", "default")

    col_start, col_prev, col_next, col_latest = st.columns([1, 1, 1, 1])

    checkpoints = st.session_state.get(f"{tab_id}_checkpoints", [])
    playback_idx = st.session_state.get(f"{tab_id}_playback_index", -1)
    # Disable if: no checkpoints, only initial checkpoint (no actions taken), or at first checkpoint
    no_actions_taken = (
        len(checkpoints) <= 1
    )  # Only initial checkpoint exists (no actions yet)
    at_first_checkpoint = playback_idx == 0
    prev_disabled = no_actions_taken or at_first_checkpoint

    with col_start:
        if st.button("⏮️ Initial State", key=f"{tab_id}_start", disabled=prev_disabled):
            jump_to_start(config)
            st.rerun()

    with col_prev:
        if st.button("◀️ Prev action", key=f"{tab_id}_rewind", disabled=prev_disabled):
            rewind_checkpoint(config)
            st.rerun()

    with col_next:
        if st.button(
            "▶️ Next action", key=f"{tab_id}_forward", disabled=not in_playback
        ):
            forward_checkpoint(config)
            st.rerun()

    with col_latest:
        if st.button(
            "⏭️ Latest action", key=f"{tab_id}_latest", disabled=not in_playback
        ):
            jump_to_latest(config)
            st.rerun()


def playback_controls_econ(config: dict, in_playback: bool) -> None:
    """Render playback controls for navigating through training history (economics version).

    Args:
        config: Configuration dict with 'tab_id' key
        in_playback: Whether currently in playback mode
    """
    from streamlit_app.state_econ import (
        rewind_checkpoint_econ,
        forward_checkpoint_econ,
        jump_to_latest_econ,
        jump_to_start_econ,
        get_display_state_econ,
    )

    tab_id = config.get("tab_id", "default")

    col_start, col_prev, col_next, col_latest = st.columns([1, 1, 1, 1])

    checkpoints = st.session_state.get(f"{tab_id}_checkpoints", [])

    # Get current display state to check step_count
    display_state = get_display_state_econ(config)
    step_count = display_state.get("step_count", 0)

    # Determine button states
    # Prev and Initial: disabled when step_count = 0 or no checkpoints
    no_actions_taken = len(checkpoints) <= 1
    at_step_zero = step_count == 0
    prev_disabled = no_actions_taken or at_step_zero

    # Next and Latest: disabled when not in playback mode OR when at the latest checkpoint
    # Get the latest checkpoint index to check if we're at the latest
    latest_idx = st.session_state.get(
        f"{tab_id}_latest_checkpoint_index", len(checkpoints) - 1 if checkpoints else -1
    )
    latest_idx = min(latest_idx, len(checkpoints) - 1) if checkpoints else -1

    if in_playback:
        playback_idx = st.session_state.get(f"{tab_id}_playback_index", -1)
        # If we're at the latest checkpoint (or beyond), disable Next and Latest
        at_latest_checkpoint = playback_idx >= latest_idx if latest_idx >= 0 else False
        next_disabled = at_latest_checkpoint
        latest_disabled = at_latest_checkpoint
    else:
        # Not in playback mode (live mode) - buttons should be disabled
        next_disabled = True
        latest_disabled = True

    with col_start:
        if st.button(
            "⏮️ Initial State", key=f"{tab_id}_start_econ", disabled=prev_disabled
        ):
            jump_to_start_econ(config)
            st.rerun()

    with col_prev:
        if st.button(
            "◀️ Prev action", key=f"{tab_id}_rewind_econ", disabled=prev_disabled
        ):
            rewind_checkpoint_econ(config)
            st.rerun()

    with col_next:
        if st.button(
            "▶️ Next action", key=f"{tab_id}_forward_econ", disabled=next_disabled
        ):
            forward_checkpoint_econ(config)
            st.rerun()

    with col_latest:
        if st.button(
            "⏭️ Latest action", key=f"{tab_id}_latest_econ", disabled=latest_disabled
        ):
            jump_to_latest_econ(config)
            st.rerun()
