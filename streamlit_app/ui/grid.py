"""Grid-world display with emoji visualization (global centered version)."""
from __future__ import annotations

import streamlit as st

__all__ = ["render_grid"]

def render_grid(grid_size: int, current_state: int, goal_pos: int, path: list[int], show_path: bool = True) -> None:
    """Render 1D grid world with emoji indicators and globally centered text."""
    st.subheader("Grid World")
    
    cols = st.columns(grid_size)
    for i, col in enumerate(cols):
        with col:
            display_str = str(i)
            if i == current_state:
                display_str += "\n\n🐶"
            elif i == goal_pos:
                display_str += "\n\n🍖"
            else:
                display_str += "\n\n⬜"
            st.info(display_str)
            # Insert global centering CSS (original behaviour)
            st.markdown(
                """<style>div[data-testid='stVerticalBlock'] > div > div {text-align: center;}</style>""",
                unsafe_allow_html=True,
            )
    
    # Only show path when actively training
    if show_path:
        st.write(f"**Current Path:** {path}")
