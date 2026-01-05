"""Grid-world display with emoji visualization (global centered version)."""
from __future__ import annotations

import streamlit as st

__all__ = ["render_grid_1d", "render_grid_2d"]

def render_grid_1d(start_pos: int, end_pos: int, current_state: int, goal_pos: int, path: list[int], show_path: bool = True, show_dog: bool = True) -> None:
    """Render 1D grid with emoji indicators and globally centered text.
    
    Args:
        start_pos: Starting position of the grid (can be negative)
        end_pos: Ending position of the grid (inclusive)
        current_state: Current position of the agent
        goal_pos: Position of the goal
        path: List of positions visited
        show_path: Whether to display the path below the grid
        show_dog: Whether to display the dog emoji at current position
    """
    st.subheader("1D Grid")
    
    # Calculate positions from start_pos to end_pos (inclusive)
    positions = list(range(start_pos, end_pos + 1))
    grid_size = len(positions)
    
    cols = st.columns(grid_size)
    for i, col in enumerate(cols):
        pos = positions[i]  # Actual position value (can be negative)
        with col:
            display_str = str(pos)
            if show_dog and pos == current_state and pos == goal_pos:
                # Dog reached the goal - show both
                display_str += "\n\n🐶🍖"
            elif show_dog and pos == current_state:
                display_str += "\n\n🐶"
            elif pos == goal_pos:
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


def render_grid_2d(
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    current_state: tuple[int, int],
    goal_pos: tuple[int, int],
    path: list[tuple[int, int]],
    show_path: bool = True,
    show_dog: bool = True,
) -> None:
    """Render 2D grid with emoji indicators using Cartesian coordinates.
    
    Args:
        x_start: Starting X coordinate (left edge)
        x_end: Ending X coordinate (right edge, inclusive)
        y_start: Starting Y coordinate (bottom edge)
        y_end: Ending Y coordinate (top edge, inclusive)
        current_state: (x, y) tuple of current position
        goal_pos: (x, y) tuple of goal position
        path: List of (x, y) tuples showing the path taken
        show_path: Whether to display the path below the grid
        show_dog: Whether to display the dog emoji at current position
    """
    st.subheader("2D Grid")
    
    # Calculate grid dimensions
    x_positions = list(range(x_start, x_end + 1))
    y_positions = list(range(y_start, y_end + 1))
    num_cols = len(x_positions)
    
    # Render grid with (0,0) at bottom-left: iterate Y from highest to lowest
    for y in reversed(y_positions):
        cols = st.columns(num_cols)
        for i, x in enumerate(x_positions):
            with cols[i]:
                pos = (x, y)
                display_str = f"({x},{y})"
                if show_dog and pos == current_state and pos == goal_pos:
                    # Dog reached the goal - show both
                    display_str += "\n\n🐶🍖"
                elif show_dog and pos == current_state:
                    display_str += "\n\n🐶"
                elif pos == goal_pos:
                    display_str += "\n\n🍖"
                else:
                    display_str += "\n\n⬜"
                st.info(display_str)
    
    # Insert global centering CSS
    st.markdown(
        """<style>div[data-testid='stVerticalBlock'] > div > div {text-align: center;}</style>""",
        unsafe_allow_html=True,
    )
    
    # Only show path when actively training
    if show_path:
        st.write(f"**Current Path:** {path}")
