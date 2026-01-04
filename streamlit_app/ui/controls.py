"""Streamlit sidebar & control widgets for LineWorld Q-learning demo."""
from __future__ import annotations

import streamlit as st

__all__ = [
    "sidebar_config",
    "tab_controls",
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
      grid_size, goal_pos, start_mode, fixed_start_pos,
      alpha, gamma, epsilon, reward_val, tab_id
    """
    st.subheader("🐶 Environment Settings")

    grid_size: int = st.number_input(
        "Grid Size", min_value=3, max_value=20, value=6, key=f"{tab_id}_grid_size"
    )
    goal_pos: int = st.number_input(
        "Goal Position (Bone)",
        min_value=0,
        max_value=grid_size - 1,
        value=5,
        key=f"{tab_id}_goal_pos",
    )

    st.markdown("**Starting Position**")
    start_mode: str = st.radio(
        "Start Position Mode", ["Fixed", "Randomized"], index=0, key=f"{tab_id}_start_mode"
    )

    fixed_start_pos: int = 0
    if start_mode == "Fixed":
        fixed_start_pos = st.number_input(
            "Fixed Start Position",
            min_value=0,
            max_value=grid_size - 1,
            value=2,
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
        "grid_size": grid_size,
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