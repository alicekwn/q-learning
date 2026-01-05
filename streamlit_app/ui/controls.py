"""Streamlit sidebar & control widgets for LineWorld Q-learning demo."""
from __future__ import annotations

import streamlit as st

__all__ = [
    "sidebar_config",
    "tab_controls",
    "tab_controls_2d",
    "tab_controls_horizontal",
    "tab_controls_2d_horizontal",
]


def sidebar_config() -> dict:
    """Render sidebar and return selected hyper-parameters/settings.

    Returns a dict with keys:
      grid_size, goal_pos, start_mode, fixed_start_pos,
      alpha, gamma, epsilon, reward_val
    """
    st.sidebar.header("🐶 Environment Settings")

    grid_size: int = st.sidebar.number_input("Grid Size", min_value=3, max_value=20, value=6)
    goal_pos: int = st.sidebar.number_input(
        "Goal Position (Bone)", min_value=0, max_value=grid_size - 1, value=5
    )

    st.sidebar.subheader("Starting Position")
    start_mode: str = st.sidebar.radio("Start Position Mode", ["Fixed", "Randomized"], index=0)

    fixed_start_pos: int = 0
    if start_mode == "Fixed":
        fixed_start_pos = st.sidebar.number_input(
            "Fixed Start Position", min_value=0, max_value=grid_size - 1, value=2
        )
    else:
        st.sidebar.info("Dog will start at a random position each episode.")

    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Q-Learning Parameters")
    alpha: float = st.sidebar.slider("Alpha (Learning Rate)", 0.0, 1.0, 0.5, 0.01)
    gamma: float = st.sidebar.slider("Gamma (Discount Factor)", 0.0, 1.0, 0.9, 0.01)
    epsilon: float = st.sidebar.slider("Epsilon (Exploration Rate)", 0.0, 1.0, 0.2, 0.01)
    reward_val: float = st.sidebar.number_input("Reward Value", value=1.0)

    return {
        "grid_size": grid_size,
        "goal_pos": goal_pos,
        "start_mode": start_mode,
        "fixed_start_pos": fixed_start_pos,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "reward_val": reward_val,
    }


def tab_controls(tab_id: str) -> dict:
    """Render controls scoped to a specific tab (for independent configs).

    Returns a dict with keys:
      start_pos, end_pos, goal_pos, start_mode, fixed_start_pos,
      alpha, gamma, epsilon, reward_val, tab_id
    """
    st.subheader("🐶 Environment Settings")

    st.markdown("**Grid Range**")
    col_range1, col_range2 = st.columns(2)
    with col_range1:
        start_pos: int = st.number_input(
            "Start Position", min_value=-10, max_value=10, value=0, key=f"{tab_id}_start_pos"
        )
    with col_range2:
        end_pos: int = st.number_input(
            "End Position", min_value=-10, max_value=10, value=5, key=f"{tab_id}_end_pos"
        )
    
    # Ensure end_pos > start_pos
    if end_pos <= start_pos:
        st.error("End Position must be greater than Start Position!")
        end_pos = start_pos + 1

    goal_pos: int = st.number_input(
        "Goal Position (Bone)",
        min_value=start_pos,
        max_value=end_pos,
        value=end_pos,
        key=f"{tab_id}_goal_pos",
    )

    st.markdown("**Dog Starting Position**")
    start_mode: str = st.radio(
        "Start Position Mode", ["Fixed", "Randomized"], index=0, key=f"{tab_id}_start_mode"
    )

    fixed_start_pos: int = start_pos
    if start_mode == "Fixed":
        fixed_start_pos = st.number_input(
            "Fixed Start Position",
            min_value=start_pos,
            max_value=end_pos,
            value=start_pos,
            key=f"{tab_id}_fixed_start",
        )
    else:
        st.info("Dog will start at a random position each episode.")

    st.markdown("---")
    st.subheader("🧠 Q-Learning Parameters")
    alpha: float = st.slider(
        "Alpha (Learning Rate)", 0.0, 1.0, 0.5, 0.01, key=f"{tab_id}_alpha"
    )
    gamma: float = st.slider(
        "Gamma (Discount Factor)", 0.0, 1.0, 0.9, 0.01, key=f"{tab_id}_gamma"
    )
    epsilon: float = st.slider(
        "Epsilon (Exploration Rate)", 0.0, 1.0, 0.2, 0.01, key=f"{tab_id}_epsilon"
    )
    reward_val: float = st.number_input("Reward Value", value=1.0, key=f"{tab_id}_reward")

    return {
        "tab_id": tab_id,
        "start_pos": start_pos,
        "end_pos": end_pos,
        "goal_pos": goal_pos,
        "start_mode": start_mode,
        "fixed_start_pos": fixed_start_pos,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "reward_val": reward_val,
    }


def inline_help(text: str, help_text: str) -> None:
    """Display text with an inline help icon"""
    st.markdown(f"""
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
    """, unsafe_allow_html=True)


def tab_controls_2d(tab_id: str) -> dict:
    """Render controls scoped to a specific tab for 2D grid environment.

    Returns a dict with keys:
      x_start, x_end, y_start, y_end, goal_x, goal_y, start_mode, 
      fixed_start_x, fixed_start_y, alpha, gamma, epsilon, reward_val, tab_id
    """
    st.subheader("🐶 Environment Settings (2D Grid)")

    st.markdown("**X-Axis Range (horizontal)**")
    col_x1, col_x2 = st.columns(2)
    with col_x1:
        x_start: int = st.number_input(
            "X Start", min_value=-10, max_value=10, value=0, key=f"{tab_id}_x_start"
        )
    with col_x2:
        x_end: int = st.number_input(
            "X End", min_value=-10, max_value=10, value=3, key=f"{tab_id}_x_end"
        )
    
    if x_end <= x_start:
        st.error("X End must be greater than X Start!")
        x_end = x_start + 1

    st.markdown("**Y-Axis Range (vertical)**")
    col_y1, col_y2 = st.columns(2)
    with col_y1:
        y_start: int = st.number_input(
            "Y Start", min_value=-10, max_value=10, value=0, key=f"{tab_id}_y_start"
        )
    with col_y2:
        y_end: int = st.number_input(
            "Y End", min_value=-10, max_value=10, value=3, key=f"{tab_id}_y_end"
        )
    
    if y_end <= y_start:
        st.error("Y End must be greater than Y Start!")
        y_end = y_start + 1

    st.markdown("**Goal Position (Bone)**")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        goal_x: int = st.number_input(
            "Goal X", min_value=x_start, max_value=x_end, value=x_end, key=f"{tab_id}_goal_x"
        )
    with col_g2:
        goal_y: int = st.number_input(
            "Goal Y", min_value=y_start, max_value=y_end, value=y_end, key=f"{tab_id}_goal_y"
        )

    st.markdown("**Dog Starting Position**")
    start_mode: str = st.radio(
        "Start Position Mode", ["Fixed", "Randomized"], index=0, key=f"{tab_id}_start_mode"
    )

    fixed_start_x: int = x_start
    fixed_start_y: int = y_start
    if start_mode == "Fixed":
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fixed_start_x = st.number_input(
                "Start X", min_value=x_start, max_value=x_end, value=x_start, key=f"{tab_id}_fixed_start_x"
            )
        with col_s2:
            fixed_start_y = st.number_input(
                "Start Y", min_value=y_start, max_value=y_end, value=y_start, key=f"{tab_id}_fixed_start_y"
            )
    else:
        st.info("Dog will start at a random position each episode.")

    st.markdown("---")
    st.subheader("🧠 Q-Learning Parameters")
    alpha: float = st.slider(
        "Alpha (Learning Rate)", 0.0, 1.0, 0.5, 0.01, key=f"{tab_id}_alpha"
    )
    gamma: float = st.slider(
        "Gamma (Discount Factor)", 0.0, 1.0, 0.9, 0.01, key=f"{tab_id}_gamma"
    )
    epsilon: float = st.slider(
        "Epsilon (Exploration Rate)", 0.0, 1.0, 0.2, 0.01, key=f"{tab_id}_epsilon"
    )
    reward_val: float = st.number_input("Reward Value", value=1.0, key=f"{tab_id}_reward")

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
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "reward_val": reward_val,
    }


def tab_controls_horizontal(tab_id: str) -> dict:
    """Render 1D controls in horizontal layout with two expander rows.

    Returns a dict with keys:
      start_pos, end_pos, goal_pos, start_mode, fixed_start_pos,
      alpha, gamma, epsilon, reward_val, tab_id
    """
    # Row 1: Environment Settings
    with st.expander("🐶 Environment Settings", expanded=True):
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1.5])
        
        with col1:
            start_pos: int = st.number_input(
                "Start Position", min_value=-10, max_value=10, value=0, key=f"{tab_id}_start_pos"
            )
        with col2:
            end_pos: int = st.number_input(
                "End Position", min_value=-10, max_value=10, value=5, key=f"{tab_id}_end_pos"
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
                value=end_pos,
                key=f"{tab_id}_goal_pos",
            )

        with col4:
            start_mode: str = st.radio(
                "Dog Start Mode", ["Fixed", "Randomized"], index=0, key=f"{tab_id}_start_mode", horizontal=True
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
                )
            else:
                st.info("Random start each episode")

    # Row 2: Q-Learning Parameters
    with st.expander("🧠 Q-Learning Parameters", expanded=True):
        col_a, col_g, col_e, col_r = st.columns([1, 1, 1, 1])
        
        with col_a:
            alpha: float = st.slider(
                "Alpha (Learning Rate)", 0.0, 1.0, 0.5, 0.01, key=f"{tab_id}_alpha"
            )
        with col_g:
            gamma: float = st.slider(
                "Gamma (Discount Factor)", 0.0, 1.0, 0.9, 0.01, key=f"{tab_id}_gamma"
            )
        with col_e:
            epsilon: float = st.slider(
                "Epsilon (Exploration Rate)", 0.0, 1.0, 0.2, 0.01, key=f"{tab_id}_epsilon"
            )
        with col_r:
            reward_val: float = st.number_input("Reward Value", value=1.0, key=f"{tab_id}_reward")

    
    return {
        "tab_id": tab_id,
        "start_pos": start_pos,
        "end_pos": end_pos,
        "goal_pos": goal_pos,
        "start_mode": start_mode,
        "fixed_start_pos": fixed_start_pos,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "reward_val": reward_val,
    }


def tab_controls_2d_horizontal(tab_id: str) -> dict:
    """Render 2D controls in horizontal layout with two expander rows.

    Returns a dict with keys:
      x_start, x_end, y_start, y_end, goal_x, goal_y, start_mode, 
      fixed_start_x, fixed_start_y, alpha, gamma, epsilon, reward_val, tab_id
    """
    # Row 1: Environment Settings
    with st.expander("🐶 Environment Settings", expanded=True):
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        
        with col1:
            x_start: int = st.number_input(
                "X Start", min_value=-10, max_value=10, value=0, key=f"{tab_id}_x_start"
            )
            y_start: int = st.number_input(
                "Y Start", min_value=-10, max_value=10, value=0, key=f"{tab_id}_y_start"
            )

        with col2:
            x_end: int = st.number_input(
                "X End", min_value=-10, max_value=10, value=3, key=f"{tab_id}_x_end"
            )
        
            y_end: int = st.number_input(
                "Y End", min_value=-10, max_value=10, value=3, key=f"{tab_id}_y_end"
            )

        if x_end <= x_start:
            st.error("X End > X Start!")
            x_end = x_start + 1
        
        if y_end <= y_start:
            st.error("Y End > Y Start!")
            y_end = y_start + 1

        with col3:
            goal_x: int = st.number_input(
                "Goal X", min_value=x_start, max_value=x_end, value=x_end, key=f"{tab_id}_goal_x"
            )
            goal_y: int = st.number_input(
                "Goal Y", min_value=y_start, max_value=y_end, value=y_end, key=f"{tab_id}_goal_y"
            )

        with col4:
            start_mode: str = st.radio(
                "Dog Start Mode", ["Fixed", "Randomized"], index=0, key=f"{tab_id}_start_mode", horizontal=True
            )

        with col5:
            fixed_start_x: int = x_start
            fixed_start_y: int = y_start
            if start_mode == "Fixed":
                fixed_start_x = st.number_input(
                    "Start X", min_value=x_start, max_value=x_end, value=x_start, key=f"{tab_id}_fixed_start_x"
                )
                fixed_start_y = st.number_input(
                    "Start Y", min_value=y_start, max_value=y_end, value=y_start, key=f"{tab_id}_fixed_start_y"
                )
            else:
                st.info("Random start each episode")

    # Row 2: Q-Learning Parameters
    with st.expander("🧠 Q-Learning Parameters", expanded=True):
        col_a, col_g, col_e, col_r = st.columns([1, 1, 1, 1])
        
        with col_a:
            alpha: float = st.slider(
                "Alpha (Learning Rate)", 0.0, 1.0, 0.5, 0.01, key=f"{tab_id}_alpha"
            )
        with col_g:
            gamma: float = st.slider(
                "Gamma (Discount Factor)", 0.0, 1.0, 0.9, 0.01, key=f"{tab_id}_gamma"
            )
        with col_e:
            epsilon: float = st.slider(
                "Epsilon (Exploration Rate)", 0.0, 1.0, 0.2, 0.01, key=f"{tab_id}_epsilon"
            )
        with col_r:
            reward_val: float = st.number_input("Reward Value", value=1.0, key=f"{tab_id}_reward")

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
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "reward_val": reward_val,
    }