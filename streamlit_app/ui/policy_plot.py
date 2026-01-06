"""Policy visualization using vector field diagrams."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

__all__ = ["render_policy_1d", "render_policy_2d"]

# Action to direction mapping for 2D
ACTIONS_2D_VECTORS = {
    "U": (0, 1),  # Up: +Y
    "D": (0, -1),  # Down: -Y
    "L": (-1, 0),  # Left: -X
    "R": (1, 0),  # Right: +X
}

# Diagonal arrow mapping for multiple optimal actions (normalized vectors)
DIAGONAL_VECTORS = {
    frozenset(["U", "R"]): (1, 1),  # Up-Right
    frozenset(["U", "L"]): (-1, 1),  # Up-Left
    frozenset(["D", "R"]): (1, -1),  # Down-Right
    frozenset(["D", "L"]): (-1, -1),  # Down-Left
}


def _q_table_hash(q_table: pd.DataFrame) -> str:
    """Create a hash string from Q-table for caching."""
    return str(q_table.values.tobytes())


@st.cache_data(hash_funcs={pd.DataFrame: _q_table_hash})
def _create_policy_figure_1d(
    q_table: pd.DataFrame,
    start_pos: int,
    end_pos: int,
    goal_pos: int,
):
    """Create 1D policy matplotlib figure (cached)."""
    positions = list(range(start_pos, end_pos + 1))
    n_positions = len(positions)

    fig, ax = plt.subplots(figsize=(max(6, n_positions * 0.8), 2))

    # Draw grid background (no borders)
    for i, pos in enumerate(positions):
        color = "gold" if pos == goal_pos else "lightblue"
        ax.add_patch(
            plt.Rectangle(
                (i - 0.4, -0.3),
                0.8,
                0.6,
                facecolor=color,
                edgecolor="none",
                linewidth=0,
            )
        )
        ax.text(i, -0.5, str(pos), ha="center", va="top", fontsize=9)

    # Draw arrows for best action at each position
    for i, pos in enumerate(positions):
        if pos == goal_pos:
            # Goal position - use star marker instead of emoji for better compatibility
            ax.plot(i, 0, "*", color="black", markersize=20, markeredgewidth=0)
            continue

        # Get Q-values for this position
        if pos in q_table.index:
            q_left = q_table.at[pos, "L"]
            q_right = q_table.at[pos, "R"]

            # Determine best action
            if q_left == q_right == 0:
                # No learning yet - show dot
                ax.plot(i, 0, "o", color="gray", markersize=8)
            elif q_left > q_right:
                # Arrow pointing left - make it more visible
                ax.annotate(
                    "",
                    xy=(i - 0.3, 0),
                    xytext=(i + 0.2, 0),
                    arrowprops=dict(
                        arrowstyle="->", color="darkblue", lw=3, mutation_scale=20
                    ),
                )
            elif q_right > q_left:
                # Arrow pointing right - make it more visible
                ax.annotate(
                    "",
                    xy=(i + 0.3, 0),
                    xytext=(i - 0.2, 0),
                    arrowprops=dict(
                        arrowstyle="->", color="darkblue", lw=3, mutation_scale=20
                    ),
                )
            else:
                # Equal Q-values - show bidirectional or dot
                ax.plot(i, 0, "o", color="purple", markersize=8)

    ax.set_xlim(-0.6, n_positions - 0.4)
    ax.set_ylim(-0.7, 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    # ax.set_title("Greedy Policy: Arrow shows best action at each position")

    plt.tight_layout()
    return fig


@st.cache_data(hash_funcs={pd.DataFrame: _q_table_hash})
def _create_policy_figure_2d(
    q_table: pd.DataFrame,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    goal_pos: tuple[int, int],
):
    """Create 2D policy matplotlib figure (cached)."""
    x_positions = list(range(x_start, x_end + 1))
    y_positions = list(range(y_start, y_end + 1))
    n_x = len(x_positions)
    n_y = len(y_positions)

    fig, ax = plt.subplots(figsize=(max(4, n_x * 1.2), max(4, n_y * 1.2)))

    for x in x_positions:
        for y in y_positions:
            pos = (x, y)
            pos_str = str(pos)

            # Draw grid cell background (no borders)
            color = "gold" if pos == goal_pos else "lightblue"
            ax.add_patch(
                plt.Rectangle(
                    (x - 0.45, y - 0.45),
                    0.9,
                    0.9,
                    facecolor=color,
                    edgecolor="none",
                    linewidth=0,
                )
            )

            if pos == goal_pos:
                # Goal position - use star marker instead of emoji for better compatibility
                ax.plot(x, y, "*", color="black", markersize=15, markeredgewidth=0)
                continue

            # Get Q-values for this position
            if pos_str in q_table.index:
                q_values = {a: q_table.at[pos_str, a] for a in ["U", "D", "L", "R"]}
                max_q = max(q_values.values())

                if max_q == 0 and all(v == 0 for v in q_values.values()):
                    # No learning yet - show dot
                    ax.plot(x, y, "o", color="gray", markersize=8)
                else:
                    # Find best action(s)
                    best_actions = [a for a, v in q_values.items() if v == max_q]

                    if len(best_actions) == 1:
                        # Single best action - draw arrow using annotate (consistent with 1D)
                        dx, dy = ACTIONS_2D_VECTORS[best_actions[0]]
                        # Calculate arrow start (slightly back from center) and end (in direction)
                        arrow_length = 0.3
                        start_x = x - dx * 0.2
                        start_y = y - dy * 0.2
                        end_x = x + dx * arrow_length
                        end_y = y + dy * arrow_length
                        ax.annotate(
                            "",
                            xy=(end_x, end_y),
                            xytext=(start_x, start_y),
                            arrowprops=dict(
                                arrowstyle="->",
                                color="darkblue",
                                lw=3,
                                mutation_scale=20,
                            ),
                        )
                    else:
                        # Multiple best actions - show diagonal arrow if applicable
                        best_actions_set = frozenset(best_actions)
                        diagonal_vector = DIAGONAL_VECTORS.get(best_actions_set)
                        if diagonal_vector:
                            # Draw diagonal arrow using annotate (same style as single arrows)
                            dx, dy = diagonal_vector
                            # Normalize diagonal vector to unit length for consistent arrow size
                            norm = math.sqrt(dx**2 + dy**2)
                            dx_norm = dx / norm
                            dy_norm = dy / norm
                            arrow_length = 0.3
                            start_x = x - dx_norm * 0.2
                            start_y = y - dy_norm * 0.2
                            end_x = x + dx_norm * arrow_length
                            end_y = y + dy_norm * arrow_length
                            ax.annotate(
                                "",
                                xy=(end_x, end_y),
                                xytext=(start_x, start_y),
                                arrowprops=dict(
                                    arrowstyle="->",
                                    color="darkblue",
                                    lw=3,
                                    mutation_scale=20,
                                ),
                            )
                        else:
                            # Fallback to dot for other combinations
                            ax.plot(x, y, "o", color="purple", markersize=8)

    # Set axis properties
    ax.set_xlim(x_start - 0.6, x_end + 0.6)
    ax.set_ylim(y_start - 0.6, y_end + 0.6)
    ax.set_xticks(x_positions)
    ax.set_yticks(y_positions)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.grid(False)
    # ax.set_title("Greedy Policy: Arrows show best action at each position")

    plt.tight_layout()
    return fig


def render_policy_1d(
    q_table: pd.DataFrame,
    start_pos: int,
    end_pos: int,
    goal_pos: int,
) -> None:
    """Render 1D policy as a vector field showing optimal actions.

    Args:
        q_table: DataFrame with index as positions, columns as ["L", "R"]
        start_pos: Starting position of the grid
        end_pos: Ending position of the grid (inclusive)
        goal_pos: Goal position (no arrow shown here)
    """
    st.markdown(
        "Read from the Q-matrix, arrows show the optimal action at each position. (A dot means no learning yet.)"
    )
    fig = _create_policy_figure_1d(q_table, start_pos, end_pos, goal_pos)
    st.pyplot(fig)
    plt.close(fig)


def render_policy_2d(
    q_table: pd.DataFrame,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    goal_pos: tuple[int, int],
) -> None:
    """Render 2D policy as a vector field showing optimal actions.

    Args:
        q_table: DataFrame with index as "(x, y)" strings, columns as ["U", "D", "L", "R"]
        x_start, x_end: X-axis range (inclusive)
        y_start, y_end: Y-axis range (inclusive)
        goal_pos: (x, y) tuple of goal position
    """
    st.markdown(
        """
        Read from the Q-matrix, arrows show the optimal action at each position. (A dot means no learning yet.)<br>
        When there are two optimal actions, the vector sum of the two arrows is shown. (e.g. ➡️+⬆️=↗️)
        """,
        unsafe_allow_html=True,
    )
    fig = _create_policy_figure_2d(q_table, x_start, x_end, y_start, y_end, goal_pos)
    st.pyplot(fig)
    plt.close(fig)
