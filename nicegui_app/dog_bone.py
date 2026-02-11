from __future__ import annotations

"""NiceGUI implementation of the Dog & Bone Q-learning demo.

This module reuses the original training logic from `streamlit_app.state`
but provides a NiceGUI-based UI which mirrors the Streamlit page:

- Two tabs: 1D Grid and 2D Grid
- Environment and Q-learning parameter controls
- Reset / initialize model
- Step-by-step training, optional simple autoplay, and fast-forward
- Q-matrix table and basic logs
"""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st  # type: ignore[import-not-found]
from nicegui import ui

from streamlit_app.state import (
    get_display_state,
    init_session_state,
    init_session_state_2d,
    is_in_playback_mode,
    jump_to_latest,
    reset_episode,
    reset_episode_2d,
    rewind_checkpoint,
    forward_checkpoint,
    jump_to_start,
    step_agent,
    step_agent_2d,
    run_batch_training,
    run_batch_training_2d,
)


TAB_1D = "tab1"
TAB_2D = "tab2"


def _build_config_1d(
    start_pos: ui.number,  # type: ignore[type-arg]
    end_pos: ui.number,  # type: ignore[type-arg]
    goal_pos: ui.number,  # type: ignore[type-arg]
    start_mode: ui.radio,  # type: ignore[type-arg]
    fixed_start_pos: ui.number | None,  # type: ignore[type-arg]
    alpha: ui.slider,  # type: ignore[type-arg]
    gamma: ui.slider,  # type: ignore[type-arg]
    epsilon: ui.slider,  # type: ignore[type-arg]
    reward_val: ui.number,  # type: ignore[type-arg]
) -> dict[str, Any]:
    """Create the config dict expected by `streamlit_app.state` for 1D."""
    s = int(start_pos.value)
    e = int(end_pos.value)
    if e <= s:
        e = s + 1
    g = int(goal_pos.value)
    g = max(s, min(e, g))
    mode = start_mode.value or "Randomised"
    fixed = s
    if mode == "Fixed" and fixed_start_pos is not None:
        fixed = int(fixed_start_pos.value)
        fixed = max(s, min(e, fixed))

    return {
        "tab_id": TAB_1D,
        "start_pos": s,
        "end_pos": e,
        "goal_pos": g,
        "start_mode": mode,
        "fixed_start_pos": fixed,
        "alpha": float(alpha.value),
        "gamma": float(gamma.value),
        "epsilon": float(epsilon.value),
        "reward_val": float(reward_val.value),
    }


def _build_config_2d(
    x_start: ui.number,  # type: ignore[type-arg]
    x_end: ui.number,  # type: ignore[type-arg]
    y_start: ui.number,  # type: ignore[type-arg]
    y_end: ui.number,  # type: ignore[type-arg]
    goal_x: ui.number,  # type: ignore[type-arg]
    goal_y: ui.number,  # type: ignore[type-arg]
    start_mode: ui.radio,  # type: ignore[type-arg]
    fixed_start_x: ui.number | None,  # type: ignore[type-arg]
    fixed_start_y: ui.number | None,  # type: ignore[type-arg]
    alpha: ui.slider,  # type: ignore[type-arg]
    gamma: ui.slider,  # type: ignore[type-arg]
    epsilon: ui.slider,  # type: ignore[type-arg]
    reward_val: ui.number,  # type: ignore[type-arg]
) -> dict[str, Any]:
    """Create the config dict expected by `streamlit_app.state` for 2D."""
    xs = int(x_start.value)
    xe = int(x_end.value)
    if xe <= xs:
        xe = xs + 1
    ys = int(y_start.value)
    ye = int(y_end.value)
    if ye <= ys:
        ye = ys + 1

    gx = int(goal_x.value)
    gy = int(goal_y.value)
    gx = max(xs, min(xe, gx))
    gy = max(ys, min(ye, gy))

    mode = start_mode.value or "Randomised"
    fx, fy = xs, ys
    if mode == "Fixed" and fixed_start_x is not None and fixed_start_y is not None:
        fx = max(xs, min(xe, int(fixed_start_x.value)))
        fy = max(ys, min(ye, int(fixed_start_y.value)))

    return {
        "tab_id": TAB_2D,
        "x_start": xs,
        "x_end": xe,
        "y_start": ys,
        "y_end": ye,
        "goal_x": gx,
        "goal_y": gy,
        "start_mode": mode,
        "fixed_start_x": fx,
        "fixed_start_y": fy,
        "alpha": float(alpha.value),
        "gamma": float(gamma.value),
        "epsilon": float(epsilon.value),
        "reward_val": float(reward_val.value),
    }


def _render_grid_1d(container: ui.column, config: dict[str, Any], display_state: dict) -> None:
    """Render a simple 1D grid similar to the Streamlit version."""
    container.clear()

    start_pos = config["start_pos"]
    end_pos = config["end_pos"]
    goal_pos = config["goal_pos"]
    current_state = display_state["current_state"]
    path = display_state["current_path"]
    ready = display_state.get("ready_for_episode", True)
    is_terminal = display_state.get("is_terminal", True)
    episode_completed_via_step = display_state.get("episode_completed_via_step", False)

    show_dog = (
        (config["start_mode"] == "Fixed")
        or (not ready)
        or (ready and is_terminal and episode_completed_via_step)
    )
    show_final_path = ready and is_terminal and episode_completed_via_step

    positions = list(range(start_pos, end_pos + 1))
    cells = []
    for pos in positions:
        label = str(pos)
        if show_dog and pos == current_state and pos == goal_pos:
            label += "<br/>🐶🦴"
        elif show_dog and pos == current_state:
            label += "<br/>🐶"
        elif pos == goal_pos:
            label += "<br/>🍖"
        else:
            label += "<br/>⬜"
        cells.append(
            f"<div style='flex:1; padding:8px; text-align:center; border:1px solid #ddd; border-radius:4px;'>{label}</div>"
        )

    html = (
        "<div style='display:flex; gap:8px; justify-content:center; margin-top:8px;'>"
        + "".join(cells)
        + "</div>"
    )
    with container:
        ui.label("1D Grid").classes("font-semibold")
        ui.html(html)
        if not ready or show_final_path:
            ui.label(f"Current Path: {path}").classes("text-sm text-gray-700")


def _render_grid_2d(
    container: ui.column,
    config: dict[str, Any],
    display_state: dict,
) -> None:
    """Render a simple 2D grid similar to the Streamlit version."""
    container.clear()

    xs = config["x_start"]
    xe = config["x_end"]
    ys = config["y_start"]
    ye = config["y_end"]
    goal = (config["goal_x"], config["goal_y"])
    current_state = display_state["current_state"]
    path = display_state["current_path"]
    ready = display_state.get("ready_for_episode", True)
    is_terminal = display_state.get("is_terminal", True)
    episode_completed_via_step = display_state.get("episode_completed_via_step", False)

    show_dog = (
        (config["start_mode"] == "Fixed")
        or (not ready)
        or (ready and is_terminal and episode_completed_via_step)
    )
    show_final_path = ready and is_terminal and episode_completed_via_step

    x_positions = list(range(xs, xe + 1))
    y_positions = list(range(ys, ye + 1))

    rows_html = []
    for y in reversed(y_positions):
        row_cells = []
        for x in x_positions:
            pos = (x, y)
            label = f"({x},{y})"
            if show_dog and pos == current_state and pos == goal:
                label += "<br/>🐶🦴"
            elif show_dog and pos == current_state:
                label += "<br/>🐶"
            elif pos == goal:
                label += "<br/>🍖"
            else:
                label += "<br/>⬜"
            row_cells.append(
                f"<td style='padding:6px 10px; text-align:center; border:1px solid #ddd;'>{label}</td>"
            )
        rows_html.append("<tr>" + "".join(row_cells) + "</tr>")

    table_html = (
        "<table style='border-collapse:collapse; margin-top:8px;'>"
        + "".join(rows_html)
        + "</table>"
    )

    with container:
        ui.label("2D Grid").classes("font-semibold")
        ui.html(table_html)
        if not ready or show_final_path:
            ui.label(f"Current Path: {path}").classes("text-sm text-gray-700")


def _render_q_table(container: ui.column, q_table: pd.DataFrame, title: str) -> None:
    container.clear()
    with container:
        ui.label(title).classes("font-semibold")
        records = q_table.reset_index().rename(columns={"index": "state"}).to_dict(
            orient="records"
        )
        if records:
            fields = list(records[0].keys())
        else:
            fields = ["state"]
        columns = [{"label": f, "field": f} for f in fields]
        ui.table(columns=columns, rows=records).classes("text-xs")


def _render_step_log(container: ui.column, step_log: list[dict[str, Any]]) -> None:
    container.clear()
    with container:
        ui.label("Bellman Log (latest steps)").classes("font-semibold")
        if not step_log:
            ui.label("No steps taken yet.").classes("text-sm text-gray-600")
            return
        recent = step_log[-50:]
        rows = []
        for e in recent:
            new_q = e.get("New Q")
            if isinstance(new_q, (int, float)):
                new_q_str = f"{new_q:.4f}"
            else:
                new_q_str = str(new_q)
            rows.append(
                {
                    "Episode": e.get("Episode"),
                    "Step": e.get("Step"),
                    "State": e.get("State (s)"),
                    "Action": e.get("Action (a)"),
                    "Next state": e.get("Next state"),
                    "New Q": new_q_str,
                }
            )
        cols = [
            {"label": "Episode", "field": "Episode"},
            {"label": "Step", "field": "Step"},
            {"label": "State", "field": "State"},
            {"label": "Action", "field": "Action"},
            {"label": "Next state", "field": "Next state"},
            {"label": "New Q", "field": "New Q"},
        ]
        ui.table(columns=cols, rows=rows).classes("text-xs")


def _render_q_history_summary(
    container: ui.column,
    q_history_plot: list[dict[str, Any]],
    steps_per_episode: list[int],
) -> None:
    """Simple textual summary instead of full Streamlit charts."""
    container.clear()
    with container:
        ui.label("Training Summary").classes("font-semibold")
        if not q_history_plot:
            ui.label("No Q-value history yet.").classes("text-sm text-gray-600")
            return
        last = q_history_plot[-1]
        episode = last.get("Episode", 0)
        ui.label(f"Episodes completed: {episode}").classes("text-sm")
        if steps_per_episode:
            ui.label(
                f"Last episode steps: {steps_per_episode[-1]}"
            ).classes("text-sm text-gray-700")


def _render_policy_1d(
    container: ui.column,
    q_table: pd.DataFrame,
    start_pos: int,
    end_pos: int,
    goal_pos: int,
) -> None:
    """Render a simple 1D optimal action visualization using arrows."""
    container.clear()
    with container:
        ui.label("Optimal Actions (1D)").classes("font-semibold")
        positions = list(range(start_pos, end_pos + 1))
        cells: list[str] = []
        for pos in positions:
            if pos == goal_pos:
                label = f"{pos}<br/>★"
            elif pos in q_table.index:
                q_left = q_table.at[pos, "L"]
                q_right = q_table.at[pos, "R"]
                if q_left == 0 and q_right == 0:
                    arrow = "·"
                elif q_left > q_right:
                    arrow = "←"
                elif q_right > q_left:
                    arrow = "→"
                else:
                    arrow = "·"
                label = f"{pos}<br/>{arrow}"
            else:
                label = f"{pos}<br/>·"
            cells.append(
                f"<div style='flex:1; padding:8px; text-align:center; border:1px solid #ddd; border-radius:4px;'>{label}</div>"
            )
        html = (
            "<div style='display:flex; gap:8px; justify-content:center; margin-top:8px;'>"
            + "".join(cells)
            + "</div>"
        )
        ui.html(html)


def _render_policy_2d(
    container: ui.column,
    q_table: pd.DataFrame,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    goal_pos: tuple[int, int],
) -> None:
    """Render a simple 2D optimal action visualization using arrow characters."""
    container.clear()
    with container:
        ui.label("Optimal Actions (2D)").classes("font-semibold")
        x_positions = list(range(x_start, x_end + 1))
        y_positions = list(range(y_start, y_end + 1))
        rows_html: list[str] = []
        for y in reversed(y_positions):
            row_cells: list[str] = []
            for x in x_positions:
                pos = (x, y)
                pos_str = str(pos)
                if pos == goal_pos:
                    label = f"{pos_str}<br/>★"
                elif pos_str in q_table.index:
                    q_vals = {a: q_table.at[pos_str, a] for a in q_table.columns}
                    max_q = max(q_vals.values())
                    if max_q == 0 and all(v == 0 for v in q_vals.values()):
                        arrow = "·"
                    else:
                        best_actions = [a for a, v in q_vals.items() if v == max_q]
                        # Map primary action to arrow; if multiple, show '+'
                        if "U" in best_actions:
                            arrow = "↑"
                        elif "D" in best_actions:
                            arrow = "↓"
                        elif "R" in best_actions:
                            arrow = "→"
                        elif "L" in best_actions:
                            arrow = "←"
                        else:
                            arrow = "·"
                    label = f"{pos_str}<br/>{arrow}"
                else:
                    label = f"{pos_str}<br/>·"
                row_cells.append(
                    f"<td style='padding:6px 10px; text-align:center; border:1px solid #ddd;'>{label}</td>"
                )
            rows_html.append("<tr>" + "".join(row_cells) + "</tr>")
        table_html = (
            "<table style='border-collapse:collapse; margin-top:8px;'>"
            + "".join(rows_html)
            + "</table>"
        )
        ui.html(table_html)


def _render_q_history_chart(q_history_plot: list[dict[str, Any]]) -> None:
    """Render evolving Q-values chart using matplotlib."""
    import io
    import base64

    ui.label("📈 Evolving Q-Values").classes("font-semibold")
    if not q_history_plot:
        ui.label("No Q-value history yet.").classes("text-sm text-gray-600")
        return

    hist_df = pd.DataFrame(q_history_plot)
    if hist_df.empty or "Episode" not in hist_df.columns:
        ui.label("No Q-value history yet.").classes("text-sm text-gray-600")
        return

    hist_df = hist_df.set_index("Episode")
    if len(hist_df) > 100:
        step = max(1, len(hist_df) // 100)
        hist_df = hist_df.iloc[::step]

    fig, ax = plt.subplots(figsize=(6, 3))
    hist_df.plot(ax=ax, legend=False)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Q-value")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Convert figure to base64 image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode()
    buf.close()
    plt.close(fig)
    
    ui.image(f"data:image/png;base64,{img_data}").classes("w-full")


def _render_steps_chart(
    steps_per_episode: list[int], q_history_plot: list[dict[str, Any]] | None
) -> None:
    """Render steps per episode chart similar to Streamlit version."""
    import io
    import base64

    ui.label("📊 Steps per Episode").classes("font-semibold")
    if not steps_per_episode:
        ui.label("No completed episodes yet.").classes("text-sm text-gray-600")
        return

    if q_history_plot:
        episode_numbers = [
            entry.get("Episode") for entry in q_history_plot if "Episode" in entry
        ]
        if not episode_numbers:
            ui.label("No episode data yet.").classes("text-sm text-gray-600")
            return
        filtered_data: list[dict[str, Any]] = []
        for ep_num in episode_numbers:
            if ep_num == 0:
                continue
            steps_idx = ep_num - 1
            if 0 <= steps_idx < len(steps_per_episode):
                filtered_data.append(
                    {"Episode": ep_num, "Steps": steps_per_episode[steps_idx]}
                )
        if not filtered_data:
            ui.label("No matching episode data yet.").classes("text-sm text-gray-600")
            return
        df = pd.DataFrame(filtered_data).set_index("Episode")
    else:
        df = pd.DataFrame(
            {
                "Episode": range(1, len(steps_per_episode) + 1),
                "Steps": steps_per_episode,
            }
        ).set_index("Episode")

    fig, ax = plt.subplots(figsize=(6, 3))
    df.plot(ax=ax, legend=False)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Convert figure to base64 image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode()
    buf.close()
    plt.close(fig)
    
    ui.image(f"data:image/png;base64,{img_data}").classes("w-full")


def render_page() -> None:
    """Render the complete Dog & Bone page inside the current NiceGUI route."""

    ui.label("The Dog & The Bone").classes("text-2xl font-bold")
    ui.markdown(
        """
In this demo, we use Q-learning to help the dog 🐶 learn how to find the bone 🍖!
Change the parameters to see how they affect learning, and use the controls to step through the process.
"""
    )
    ui.separator()
    with ui.tabs().classes("w-full") as tabs:
        tab_1d = ui.tab("1D Grid")
        tab_2d = ui.tab("2D Grid")

    with ui.tab_panels(tabs, value=tab_1d).classes("w-full"):
        with ui.tab_panel(tab_1d):
            _render_tab_1d()
        with ui.tab_panel(tab_2d):
            _render_tab_2d()


def _render_tab_1d() -> None:
    with ui.column().classes("w-full gap-4"):
        # Controls
        with ui.expansion("🐶 Environment Settings", value=True).classes("w-full"):
            with ui.row().classes("w-full gap-4"):
                start_pos = ui.number(
                    "Start Position", value=0, min=-10, max=10, step=1
                ).classes("w-1/5")
                end_pos = ui.number(
                    "End Position", value=5, min=-10, max=10, step=1
                ).classes("w-1/5")
                goal_pos = ui.number(
                    "Goal Position (Bone)", value=4, min=-10, max=10, step=1
                ).classes("w-1/5")
                start_mode = (
                    ui.radio(
                        ["Randomised", "Fixed"],
                        value="Randomised",
                    )
                    .props('inline label="Starting position for each episode"')
                    .classes("w-1/5")
                )
                fixed_start_pos = ui.number(
                    "Fixed Start Position", value=0, min=-10, max=10, step=1
                ).classes("w-1/5")

        with ui.expansion("🧠 Q-Learning Parameters", value=True).classes("w-full"):
            with ui.row().classes("w-full gap-4"):
                alpha = (
                    ui.slider(min=0.0, max=1.0, step=0.01, value=0.5)
                    .props('label="α (Learning Rate)"')
                    .classes("w-1/4")
                )
                gamma = (
                    ui.slider(min=0.0, max=1.0, step=0.01, value=0.9)
                    .props('label="γ (Discount Factor)"')
                    .classes("w-1/4")
                )
                epsilon = (
                    ui.slider(
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=0.2,
                    )
                    .props('label="ε (Exploration Rate)"')
                    .classes("w-1/4")
                )
                reward_val = ui.number(
                    "Reward Value r", value=1.0, step=0.1
                ).classes("w-1/4")

        grid_container = ui.column().classes("w-full gap-2")
        left_controls = ui.column().classes("w-full gap-2")
        right_info = ui.column().classes("w-full gap-2")

        with ui.row().classes("w-full gap-8"):
            with ui.column().classes("w-1/2 gap-4"):
                _render_tab_1d_controls(
                    start_pos,
                    end_pos,
                    goal_pos,
                    start_mode,
                    fixed_start_pos,
                    alpha,
                    gamma,
                    epsilon,
                    reward_val,
                    grid_container,
                    left_controls,
                )
            with ui.column().classes("w-1/2 gap-4"):
                _render_tab_1d_info(right_info)


def _render_tab_1d_controls(
    start_pos: ui.number,  # type: ignore[type-arg]
    end_pos: ui.number,  # type: ignore[type-arg]
    goal_pos: ui.number,  # type: ignore[type-arg]
    start_mode: ui.radio,  # type: ignore[type-arg]
    fixed_start_pos: ui.number,  # type: ignore[type-arg]
    alpha: ui.slider,  # type: ignore[type-arg]
    gamma: ui.slider,  # type: ignore[type-arg]
    epsilon: ui.slider,  # type: ignore[type-arg]
    reward_val: ui.number,  # type: ignore[type-arg]
    grid_container: ui.column,
    controls_container: ui.column,
) -> None:
    """Training controls and Q-matrix for 1D."""

    def build_config() -> dict[str, Any]:
        return _build_config_1d(
            start_pos,
            end_pos,
            goal_pos,
            start_mode,
            fixed_start_pos,
            alpha,
            gamma,
            epsilon,
            reward_val,
        )

    # Ensure session state initialized
    config = build_config()
    if f"{TAB_1D}_q_table" not in st.session_state:
        init_session_state(config)

    # Containers for dynamic content
    q_table_container = ui.column().classes("w-full gap-2")
    policy_container = ui.column().classes("w-full gap-2")
    summary_container = ui.column().classes("w-full gap-2")

    def refresh_view() -> None:
        cfg = build_config()
        display_state = get_display_state(cfg)
        in_playback = is_in_playback_mode(cfg)

        _render_grid_1d(grid_container, cfg, display_state)
        _render_q_table(q_table_container, display_state["q_table"], "Current Q-Matrix")
        _render_policy_1d(
            policy_container,
            display_state["q_table"],
            cfg["start_pos"],
            cfg["end_pos"],
            cfg["goal_pos"],
        )
        _render_q_history_summary(
            summary_container,
            display_state["q_history_plot"],
            display_state["steps_per_episode"],
        )
        # Simple status indicator
        controls_container.clear()
        with controls_container:
            episodes = display_state["total_episodes"]
            ready = display_state.get("ready_for_episode", True)
            if ready:
                ui.label(
                    f"Number of Episodes Trained: {episodes}"
                    if episodes > 0
                    else "No episodes trained yet."
                ).classes("text-sm")
            else:
                ui.label(
                    f"Current Training Episode: {episodes + 1}"
                ).classes("text-sm")

            with ui.row().classes("gap-2"):
                ui.button(
                    "Reset / Initialize Model",
                    on_click=lambda: (_on_reset(build_config()), refresh_view()),
                ).props("color=primary")
            ui.separator()
            with ui.row().classes("gap-2"):
                btn_step = ui.button(
                    "👟 Take Next Step",
                    on_click=lambda: (_on_step(build_config()), refresh_view()),
                )
                if in_playback:
                    btn_step.props("disable")
                episodes_fast = ui.number(
                    "Fast-forward episodes", value=1, min=1, step=1
                ).classes("w-32")

                def on_fast_forward() -> None:
                    run_batch_training(int(episodes_fast.value), build_config())
                    refresh_view()

                btn_ff = ui.button(
                    "⏩ Fast Forward",
                    on_click=on_fast_forward,
                )
                if in_playback:
                    btn_ff.props("disable")

            ui.separator()
            ui.label("Playback Controls").classes("text-sm font-semibold")
            with ui.row().classes("gap-2"):
                ui.button(
                    "⏮️ First",
                    on_click=lambda: (jump_to_start(build_config()), refresh_view()),
                )
                ui.button(
                    "◀️ Previous",
                    on_click=lambda: (rewind_checkpoint(build_config()), refresh_view()),
                )
                ui.button(
                    "Next ▶️",
                    on_click=lambda: (forward_checkpoint(build_config()), refresh_view()),
                )
                ui.button(
                    "Last ⏭️",
                    on_click=lambda: (jump_to_latest(build_config()), refresh_view()),
                )

    def _on_reset(cfg: dict[str, Any]) -> None:
        init_session_state(cfg)

    def _on_step(cfg: dict[str, Any]) -> None:
        # If we're in playback, jump back to latest live state first
        if is_in_playback_mode(cfg):
            jump_to_latest(cfg)
        step_agent(cfg)

    # Initial render
    refresh_view()
    ui.separator()
    q_table_container  # keep visible
    policy_container  # keep visible
    summary_container  # keep visible


def _render_tab_1d_info(info_container: ui.column) -> None:
    """Right-hand side info: step log and simple summaries."""
    # We read directly from session state to avoid extra config plumbing.
    if f"{TAB_1D}_q_table" not in st.session_state:
        info_container.clear()
        with info_container:
            ui.label("No training data yet.").classes("text-sm text-gray-600")
        return

    cfg = {"tab_id": TAB_1D}
    display_state = get_display_state(cfg)
    _render_step_log(info_container, display_state["step_log"])
    with info_container:
        ui.separator()
        with ui.expansion("📈 Evolving Q-Values", value=False).classes("w-full"):
            _render_q_history_chart(display_state["q_history_plot"])
        with ui.expansion("📊 Steps per Episode", value=False).classes("w-full"):
            _render_steps_chart(
                display_state["steps_per_episode"],
                display_state["q_history_plot"],
            )


def _render_tab_2d() -> None:
    with ui.column().classes("w-full gap-4"):
        # Controls
        with ui.expansion("🐶 Environment Settings", value=True).classes("w-full"):
            with ui.row().classes("w-full gap-4"):
                x_start = ui.number(
                    "X Start", value=0, min=-10, max=10, step=1
                ).classes("w-1/4")
                x_end = ui.number(
                    "X End", value=3, min=-10, max=10, step=1
                ).classes("w-1/4")
                y_start = ui.number(
                    "Y Start", value=0, min=-10, max=10, step=1
                ).classes("w-1/4")
                y_end = ui.number(
                    "Y End", value=3, min=-10, max=10, step=1
                ).classes("w-1/4")
            with ui.row().classes("w-full gap-4"):
                goal_x = ui.number(
                    "Goal position (X axis)", value=2, min=-10, max=10, step=1
                ).classes("w-1/3")
                goal_y = ui.number(
                    "Goal position (Y axis)", value=2, min=-10, max=10, step=1
                ).classes("w-1/3")
                start_mode = (
                    ui.radio(
                        ["Randomised", "Fixed"],
                        value="Randomised",
                    )
                    .props('inline label="Starting position for each episode"')
                    .classes("w-1/3")
                )
            with ui.row().classes("w-full gap-4"):
                fixed_start_x = ui.number(
                    "Start X", value=0, min=-10, max=10, step=1
                ).classes("w-1/2")
                fixed_start_y = ui.number(
                    "Start Y", value=0, min=-10, max=10, step=1
                ).classes("w-1/2")

        with ui.expansion("🧠 Q-Learning Parameters", value=True).classes("w-full"):
            with ui.row().classes("w-full gap-4"):
                alpha = (
                    ui.slider(min=0.0, max=1.0, step=0.01, value=0.5)
                    .props('label="α (Learning Rate)"')
                    .classes("w-1/4")
                )
                gamma = (
                    ui.slider(min=0.0, max=1.0, step=0.01, value=0.9)
                    .props('label="γ (Discount Factor)"')
                    .classes("w-1/4")
                )
                epsilon = (
                    ui.slider(
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=0.2,
                    )
                    .props('label="ε (Exploration Rate)"')
                    .classes("w-1/4")
                )
                reward_val = ui.number(
                    "Reward Value r", value=1.0, step=0.1
                ).classes("w-1/4")

        grid_container = ui.column().classes("w-full gap-2")
        left_controls = ui.column().classes("w-full gap-2")
        right_info = ui.column().classes("w-full gap-2")

        with ui.row().classes("w-full gap-8"):
            with ui.column().classes("w-1/2 gap-4"):
                _render_tab_2d_controls(
                    x_start,
                    x_end,
                    y_start,
                    y_end,
                    goal_x,
                    goal_y,
                    start_mode,
                    fixed_start_x,
                    fixed_start_y,
                    alpha,
                    gamma,
                    epsilon,
                    reward_val,
                    grid_container,
                    left_controls,
                )
            with ui.column().classes("w-1/2 gap-4"):
                _render_tab_2d_info(right_info)


def _render_tab_2d_controls(
    x_start: ui.number,  # type: ignore[type-arg]
    x_end: ui.number,  # type: ignore[type-arg]
    y_start: ui.number,  # type: ignore[type-arg]
    y_end: ui.number,  # type: ignore[type-arg]
    goal_x: ui.number,  # type: ignore[type-arg]
    goal_y: ui.number,  # type: ignore[type-arg]
    start_mode: ui.radio,  # type: ignore[type-arg]
    fixed_start_x: ui.number,  # type: ignore[type-arg]
    fixed_start_y: ui.number,  # type: ignore[type-arg]
    alpha: ui.slider,  # type: ignore[type-arg]
    gamma: ui.slider,  # type: ignore[type-arg]
    epsilon: ui.slider,  # type: ignore[type-arg]
    reward_val: ui.number,  # type: ignore[type-arg]
    grid_container: ui.column,
    controls_container: ui.column,
) -> None:
    """Training controls and Q-matrix for 2D."""

    def build_config() -> dict[str, Any]:
        return _build_config_2d(
            x_start,
            x_end,
            y_start,
            y_end,
            goal_x,
            goal_y,
            start_mode,
            fixed_start_x,
            fixed_start_y,
            alpha,
            gamma,
            epsilon,
            reward_val,
        )

    cfg = build_config()
    if f"{TAB_2D}_q_table" not in st.session_state:
        init_session_state_2d(cfg)

    q_table_container = ui.column().classes("w-full gap-2")
    policy_container = ui.column().classes("w-full gap-2")
    summary_container = ui.column().classes("w-full gap-2")

    def refresh_view() -> None:
        cfg_inner = build_config()
        display_state = get_display_state(cfg_inner)
        in_playback = is_in_playback_mode(cfg_inner)

        _render_grid_2d(grid_container, cfg_inner, display_state)
        _render_q_table(q_table_container, display_state["q_table"], "Current Q-Matrix")
        _render_policy_2d(
            policy_container,
            display_state["q_table"],
            cfg_inner["x_start"],
            cfg_inner["x_end"],
            cfg_inner["y_start"],
            cfg_inner["y_end"],
            (cfg_inner["goal_x"], cfg_inner["goal_y"]),
        )
        _render_q_history_summary(
            summary_container,
            display_state["q_history_plot"],
            display_state["steps_per_episode"],
        )

        controls_container.clear()
        with controls_container:
            episodes = display_state["total_episodes"]
            ready = display_state.get("ready_for_episode", True)
            if ready:
                ui.label(
                    f"Number of Episodes Trained: {episodes}"
                    if episodes > 0
                    else "No episodes trained yet."
                ).classes("text-sm")
            else:
                ui.label(
                    f"Current Training Episode: {episodes + 1}"
                ).classes("text-sm")

            with ui.row().classes("gap-2"):
                ui.button(
                    "Reset / Initialize Model",
                    on_click=lambda: (init_session_state_2d(build_config()), refresh_view()),
                ).props("color=primary")
            ui.separator()
            with ui.row().classes("gap-2"):
                btn_step = ui.button(
                    "👟 Take Next Step",
                    on_click=lambda: (_on_step(build_config()), refresh_view()),
                )
                if in_playback:
                    btn_step.props("disable")
                episodes_fast = ui.number(
                    "Fast-forward episodes", value=1, min=1, step=1
                ).classes("w-32")

                def on_fast_forward() -> None:
                    run_batch_training_2d(int(episodes_fast.value), build_config())
                    refresh_view()

                btn_ff = ui.button(
                    "⏩ Fast Forward",
                    on_click=on_fast_forward,
                )
                if in_playback:
                    btn_ff.props("disable")

            ui.separator()
            ui.label("Playback Controls").classes("text-sm font-semibold")
            with ui.row().classes("gap-2"):
                ui.button(
                    "⏮️ First",
                    on_click=lambda: (jump_to_start(build_config()), refresh_view()),
                )
                ui.button(
                    "◀️ Previous",
                    on_click=lambda: (rewind_checkpoint(build_config()), refresh_view()),
                )
                ui.button(
                    "Next ▶️",
                    on_click=lambda: (forward_checkpoint(build_config()), refresh_view()),
                )
                ui.button(
                    "Last ⏭️",
                    on_click=lambda: (jump_to_latest(build_config()), refresh_view()),
                )

    def _on_step(cfg_inner: dict[str, Any]) -> None:
        if is_in_playback_mode(cfg_inner):
            jump_to_latest(cfg_inner)
        step_agent_2d(cfg_inner)

    refresh_view()
    ui.separator()
    q_table_container
    policy_container
    summary_container


def _render_tab_2d_info(info_container: ui.column) -> None:
    if f"{TAB_2D}_q_table" not in st.session_state:
        info_container.clear()
        with info_container:
            ui.label("No training data yet.").classes("text-sm text-gray-600")
        return

    cfg = {"tab_id": TAB_2D}
    display_state = get_display_state(cfg)
    _render_step_log(info_container, display_state["step_log"])
    with info_container:
        ui.separator()
        with ui.expansion("📈 Evolving Q-Values", value=False).classes("w-full"):
            _render_q_history_chart(display_state["q_history_plot"])
        with ui.expansion("📊 Steps per Episode", value=False).classes("w-full"):
            _render_steps_chart(
                display_state["steps_per_episode"],
                display_state["q_history_plot"],
            )


