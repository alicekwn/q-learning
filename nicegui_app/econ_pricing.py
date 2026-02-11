from __future__ import annotations

"""NiceGUI implementation of the Economics Pricing Q-learning demo.

This module reuses the original training and battle logic from
`streamlit_app.state_econ` and `streamlit_app.ui.trajectory` while
providing a NiceGUI-based UI with three tabs:

- Game Theory (explanatory markdown)
- Training (controls, price history, Q-matrices, logs)
- Pricing Battle (upload two Q-tables and compare strategies)
"""

from typing import Any
import io

import numpy as np
import pandas as pd
import streamlit as st  # type: ignore[import-not-found]
from nicegui import ui, events

from streamlit_app.state_econ import (
    init_session_state_econ,
    pick_random_starting_prices_econ,
    step_agent_econ,
    run_batch_training_econ,
    run_until_convergence_econ,
    get_display_state_econ,
    is_in_playback_mode_econ,
    rewind_checkpoint_econ,
    forward_checkpoint_econ,
    jump_to_latest_econ,
    jump_to_start_econ,
    calculate_prices,
    demand1,
    demand2,
    profit1,
    profit2,
    flip_q_table_states,
)
from streamlit_app.ui.trajectory import follow_greedy_until_loop_econ


TAB_ID = "econ"


def _build_config_econ(
    k1: ui.number,  # type: ignore[type-arg]
    k2: ui.number,  # type: ignore[type-arg]
    c: ui.number,  # type: ignore[type-arg]
    m: ui.select,  # type: ignore[type-arg]
    start_mode: ui.radio,  # type: ignore[type-arg]
    fixed_start_p1: ui.number | None,  # type: ignore[type-arg]
    fixed_start_p2: ui.number | None,  # type: ignore[type-arg]
    alpha: ui.slider,  # type: ignore[type-arg]
    delta: ui.slider,  # type: ignore[type-arg]
    beta: ui.number,  # type: ignore[type-arg]
    seed: ui.number,  # type: ignore[type-arg]
    check_every: ui.number,  # type: ignore[type-arg]
    stable_required: ui.number,  # type: ignore[type-arg]
    max_periods: ui.number,  # type: ignore[type-arg]
) -> dict[str, Any]:
    """Create the config dict expected by `init_session_state_econ`."""
    env_k1 = float(k1.value)
    env_k2 = float(k2.value)
    env_c = float(c.value)
    env_m = int(m.value or 7)
    mode = start_mode.value or "Randomised"
    fs_p1 = float(fixed_start_p1.value) if fixed_start_p1 is not None else 7.0
    fs_p2 = float(fixed_start_p2.value) if fixed_start_p2 is not None else 7.0

    return {
        "tab_id": TAB_ID,
        "k1": env_k1,
        "k2": env_k2,
        "c": env_c,
        "m": env_m,
        "alpha": float(alpha.value),
        "delta": float(delta.value),
        "beta": float(beta.value),
        "seed": int(seed.value),
        "check_every": int(check_every.value),
        "stable_required": int(stable_required.value),
        "max_periods": int(max_periods.value),
        "start_mode": mode,
        "fixed_start_p1": fs_p1,
        "fixed_start_p2": fs_p2,
    }


def _render_price_history_ui(
    container: ui.column,
    price_history: list[tuple[int, float, float]],
    skipped_steps: list[tuple[int, int]],
    step_count: int,
    starting_prices_picked: bool = True,
) -> None:
    """NiceGUI version of the price history visualization."""
    from streamlit_app.ui import econ_visualization  # type: ignore

    container.clear()
    with container:
        ui.label("Price History").classes("font-semibold")

        # Reuse the core logic from the original function to build the table data,
        # but render via NiceGUI.
        if not starting_prices_picked:
            price_history = [item for item in price_history if item[0] != 0]

        if not price_history:
            ui.label(
                "No price history yet. Pick starting prices to begin training."
            ).classes("text-sm text-gray-600")
            return

        # Build display_items exactly as in econ_visualization.render_price_history
        display_items: list[dict[str, Any]] = []
        all_markers: list[dict[str, Any]] = []

        for skip_start, skip_end in skipped_steps:
            all_markers.append(
                {
                    "type": "skipped",
                    "step_range": (skip_start, skip_end),
                    "step_num": skip_start,
                }
            )
        for step_num, p1, p2 in price_history:
            all_markers.append(
                {"type": "normal", "step_num": step_num, "p1": p1, "p2": p2}
            )

        all_markers.sort(key=lambda x: x["step_num"])

        processed_skipped_ranges: set[tuple[int, int]] = set()
        for marker in all_markers:
            if marker["type"] == "skipped":
                skip_range = marker["step_range"]
                if skip_range not in processed_skipped_ranges:
                    display_items.append(
                        {
                            "type": "skipped",
                            "step_range": skip_range,
                            "step_num": skip_range[0],
                        }
                    )
                    processed_skipped_ranges.add(skip_range)
            else:
                step_num = marker["step_num"]
                is_skipped = any(
                    skip_start <= step_num <= skip_end
                    for skip_start, skip_end in skipped_steps
                )
                if not is_skipped:
                    display_items.append(
                        {
                            "type": "normal",
                            "step_num": step_num,
                            "p1": marker["p1"],
                            "p2": marker["p2"],
                        }
                    )

        if not display_items:
            ui.label("No price history to display.").classes("text-sm text-gray-600")
            return

        display_items_reversed = list(reversed(display_items))
        alice_row: list[str] = []
        bob_row: list[str] = []
        step_row: list[str] = []
        num_visible_cols = 10
        fixed_col_width = "90px"

        for item in display_items_reversed:
            if item["type"] == "skipped":
                skip_start, skip_end = item["step_range"]
                alice_row.append("Skipped")
                bob_row.append("Skipped")
                step_row.append(f"Step {skip_start} - {skip_end}")
            else:
                alice_row.append(f"{item['p1']:.1f}")
                bob_row.append(f"{item['p2']:.1f}")
                step_row.append(f"Step {item['step_num']}")

        table_id = f"price_history_table_{id(display_items)}"
        html_table = f"""
        <div id="{table_id}_container" style="width: 100%; overflow-x: auto; overflow-y: hidden; border: 1px solid #ddd; border-radius: 5px;">
            <table style="border-collapse: collapse; table-layout: fixed;">
                <tbody>
                    <tr>
                        <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">👩🏼‍💼 Alice</td>
                        {''.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in alice_row])}
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">🧑🏼‍💼 Bob</td>
                        {''.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in bob_row])}
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">Step</td>
                        {''.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; font-size: 0.85em; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in step_row])}
                    </tr>
                </tbody>
            </table>
        </div>
        """
        ui.html(html_table)

        if len(display_items) > num_visible_cols:
            ui.label(
                "Scroll horizontally to see all steps. The newest steps appear on the left."
            ).classes("text-xs text-gray-500")
        elif step_count > len(display_items):
            ui.label(f"Showing all recorded steps. (Total steps: {step_count})").classes(
                "text-xs text-gray-500"
            )


def _render_q_tables(
    container: ui.column, q1: pd.DataFrame, q2: pd.DataFrame
) -> None:
    container.clear()
    with container:
        ui.label("Current Q-Matrices").classes("font-semibold")
        with ui.row().classes("w-full gap-4"):
            with ui.column().classes("w-1/2 gap-2"):
                ui.label("Alice – Q1").classes("font-medium")
                rows1 = q1.reset_index().rename(columns={"index": "state"}).to_dict(
                    orient="records"
                )
                if rows1:
                    fields1 = list(rows1[0].keys())
                else:
                    fields1 = ["state"]
                cols1 = [{"label": f, "field": f} for f in fields1]
                ui.table(columns=cols1, rows=rows1).classes("text-xs")
            with ui.column().classes("w-1/2 gap-2"):
                ui.label("Bob – Q2").classes("font-medium")
                rows2 = q2.reset_index().rename(columns={"index": "state"}).to_dict(
                    orient="records"
                )
                if rows2:
                    fields2 = list(rows2[0].keys())
                else:
                    fields2 = ["state"]
                cols2 = [{"label": f, "field": f} for f in fields2]
                ui.table(columns=cols2, rows=rows2).classes("text-xs")

        # CSV export buttons
        with ui.row().classes("w-full gap-4 mt-2"):
            def _download_csv(df: pd.DataFrame, filename: str) -> None:
                csv_bytes = df.to_csv(index=True).encode("utf-8")
                ui.download(data=csv_bytes, filename=filename)

            ui.button(
                "Export Alice Q1 as CSV",
                on_click=lambda: _download_csv(q1, "econ_q1_alice.csv"),
            )
            ui.button(
                "Export Bob Q2 as CSV",
                on_click=lambda: _download_csv(q2, "econ_q2_bob.csv"),
            )


def _render_step_log_econ(container: ui.column, step_log: list[dict[str, Any]]) -> None:
    container.clear()
    with container:
        ui.label("Bellman Log (latest steps)").classes("font-semibold")
        if not step_log:
            ui.label("No steps taken yet.").classes("text-sm text-gray-600")
            return
        recent = step_log[-50:]
        rows: list[dict[str, str]] = []
        for e in recent:
            rows.append(
                {
                    "Step": str(e.get("Step")),
                    "s": str(e.get("state_str") or e.get("state")),
                    "p1": (
                        f"{e.get('p1_next', 0):.1f}"
                        if "p1_next" in e
                        else ""
                    ),
                    "p2": (
                        f"{e.get('p2_next', 0):.1f}"
                        if "p2_next" in e
                        else ""
                    ),
                    "Type": str(
                        e.get("Type 1") or e.get("Type 2") or e.get("Type") or ""
                    ),
                }
            )
        cols = [
            {"label": "Step", "field": "Step"},
            {"label": "State", "field": "s"},
            {"label": "p1", "field": "p1"},
            {"label": "p2", "field": "p2"},
            {"label": "Type", "field": "Type"},
        ]
        ui.table(columns=cols, rows=rows).classes("text-xs")


def _render_training_summary(
    container: ui.column, display_state: dict[str, Any]
) -> None:
    container.clear()
    with container:
        ui.label("Training Status").classes("font-semibold")
        ready = display_state.get("ready_for_training", True)
        step_count = display_state.get("step_count", 0)
        conv = display_state.get("convergence_info")
        ui.label(f"Total steps: {step_count:,}").classes("text-sm")
        ui.label(f"Ready for training: {'Yes' if ready else 'No'}").classes("text-sm")
        if conv and conv.get("converged"):
            ui.label(
                f"Converged at step {conv.get('step', 0):,}."
            ).classes("text-sm text-emerald-700")
        else:
            ui.label("Not yet converged.").classes("text-sm text-gray-700")


def _render_greedy_trajectory(container: ui.column, config: dict[str, Any]) -> None:
    """NiceGUI version of the greedy trajectory computation UI."""
    tab_id = config.get("tab_id", TAB_ID)

    prices = st.session_state.get(f"{tab_id}_prices", [])
    container.clear()
    with container:
        if not prices:
            ui.label("Initialize the model first to compute trajectories.").classes(
                "text-sm text-gray-600"
            )
            return

        prices_display = ", ".join(f"{p:.1f}" for p in prices)
        ui.label("Greedy Trajectory").classes("text-lg font-semibold")
        ui.markdown(
            "Compute the greedy trajectory starting from a given pair of prices.\n\n"
            "The trajectory follows the best-response actions for both players until a cycle is detected.\n\n"
            f"Pick from the action space A = {{{prices_display}}}."
        )

        step_count = st.session_state.get(f"{tab_id}_step_count", 0)
        min_steps_required = 1000
        if step_count < min_steps_required:
            ui.label(
                f"⚠️ The model needs to be trained for at least {min_steps_required:,} steps "
                f"before computing greedy trajectories. Current steps: {step_count:,}."
            ).classes("text-sm text-amber-700")
            return

        with ui.row().classes("w-full gap-4"):
            start_p1_input = ui.number(
                "Starting price for Alice (p₁)",
                value=float(prices[len(prices) // 2]),
                min=float(min(prices)),
                max=float(max(prices)),
                step=0.1,
            ).classes("w-1/2")
            start_p2_input = ui.number(
                "Starting price for Bob (p₂)",
                value=float(prices[len(prices) // 2]),
                min=float(min(prices)),
                max=float(max(prices)),
                step=0.1,
            ).classes("w-1/2")

        def compute_and_store_trajectory() -> None:
            start_p1 = float(start_p1_input.value)
            start_p2 = float(start_p2_input.value)

            tolerance = 5e-2
            p1_valid = any(abs(start_p1 - p) <= tolerance for p in prices)
            p2_valid = any(abs(start_p2 - p) <= tolerance for p in prices)
            if not p1_valid or not p2_valid:
                ui.notify(
                    "Prices must be in the action space; please select valid prices.",
                    type="warning",
                )
                return

            start_p1_n = min(prices, key=lambda p: abs(p - start_p1))
            start_p2_n = min(prices, key=lambda p: abs(p - start_p2))

            Q1 = st.session_state.get(f"{tab_id}_Q1")
            Q2 = st.session_state.get(f"{tab_id}_Q2")
            rng = st.session_state.get(f"{tab_id}_rng")
            if Q1 is None or Q2 is None:
                ui.notify(
                    "Q-tables not initialized. Please initialize the model first.",
                    type="negative",
                )
                return
            if rng is None:
                rng = np.random.default_rng(43)

            traj = follow_greedy_until_loop_econ(
                Q1, Q2, start_p1_n, start_p2_n, prices, rng, max_steps=50000
            )

            st.session_state[f"{tab_id}_trajectory"] = traj
            st.session_state[f"{tab_id}_trajectory_start"] = (start_p1_n, start_p2_n)

        ui.button(
            "Compute Greedy Trajectory",
            on_click=compute_and_store_trajectory,
        ).props("color=primary")

        traj = st.session_state.get(f"{tab_id}_trajectory")
        traj_start = st.session_state.get(f"{tab_id}_trajectory_start")

        if traj and traj_start:
            start_p1_n, start_p2_n = traj_start
            path = traj["path"]
            loop = traj["loop"]
            loop_start = traj["loop_start"]

            ui.markdown("**Trajectory Steps for $p_1, p_2$ :**")
            if path:
                full_path = [
                    {"step_num": 0, "a1_price": start_p1_n, "a2_price": start_p2_n}
                ]
                for i, rec in enumerate(path, start=1):
                    full_path.append(
                        {
                            "step_num": i,
                            "a1_price": rec["a1_price"],
                            "a2_price": rec["a2_price"],
                        }
                    )

                alice_row: list[str] = []
                bob_row: list[str] = []
                step_row: list[str] = []
                num_visible_cols = 10
                fixed_col_width = "90px"

                for rec in full_path:
                    alice_row.append(f"{rec['a1_price']:.1f}")
                    bob_row.append(f"{rec['a2_price']:.1f}")
                    step_row.append(f"Step {rec['step_num']}")

                table_id = f"trajectory_table_{id(path)}"
                html_table = f"""
                <div id="{table_id}_container" style="width: 100%; overflow-x: auto; overflow-y: hidden; border: 1px solid #ddd; border-radius: 5px;">
                    <table style="border-collapse: collapse; table-layout: fixed;">
                        <tbody>
                            <tr>
                                <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 120px; width: 120px;">👩🏼‍💼 Alice p₁</td>
                                {''.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in alice_row])}
                            </tr>
                            <tr>
                                <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 120px; width: 120px;">🧑🏼‍💼 Bob p₂</td>
                                {''.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in bob_row])}
                            </tr>
                            <tr>
                                <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 120px; width: 120px;">Step</td>
                                {''.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; font-size: 0.85em; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in step_row])}
                            </tr>
                        </tbody>
                    </table>
                </div>
                """
                ui.html(html_table)
                if len(full_path) > num_visible_cols:
                    ui.label(
                        "Scroll horizontally to see all steps. The newest steps appear on the left."
                    ).classes("text-xs text-gray-500")

            if loop_start is not None and loop:
                ui.label(
                    f"Cycle detected starting at step {loop_start}, length {len(loop)} steps."
                ).classes("text-sm text-gray-700")
            else:
                ui.label(
                    "No cycle detected within the maximum number of steps."
                ).classes("text-sm text-amber-700")


def render_page() -> None:
    """Render the complete Economics Pricing page."""
    ui.label("Pricing Strategies in Economics").classes("text-2xl font-bold")
    ui.markdown(
        "This page demonstrates Q-learning applied to economics pricing with "
        "two players learning optimal pricing strategies."
    )
    ui.separator()
    with ui.tabs().classes("w-full") as tabs:
        tab_theory = ui.tab("Game Theory")
        tab_training = ui.tab("Training")
        tab_battle = ui.tab("Pricing Battle")

    with ui.tab_panels(tabs, value=tab_theory).classes("w-full"):
        with ui.tab_panel(tab_theory):
            _render_tab_theory()
        with ui.tab_panel(tab_training):
            _render_tab_training()
        with ui.tab_panel(tab_battle):
            _render_tab_battle()


def _render_tab_theory() -> None:
    with ui.column().classes("w-full max-w-4xl gap-4"):
        ui.markdown(
            r"""
### Using game theory to find the equilibrium price and the collusive price

This is a demo version of an economics game. It assumes there are only 2 parties, and uses simplified demand:

\[
q_1 = k_1 - p_1 + k_2 \cdot p_2, \quad q_2 = k_1 - p_2 + k_2 \cdot p_1
\]

\[
\pi_1 = (p_1 - c) \times q_1, \quad \pi_2 = (p_2 - c) \times q_2
\]

The equilibrium price \(p_e\) is:

\[
p_e = \frac{k_1 + c}{2 - k_2}
\]

The collusive price \(p_c\) is:

\[
p_c = \frac{2 k_1 + 2c (1-k_2)}{4(1-k_2)}
\]

The feasible prices for each party lie in \([2p_e-p_c, 2p_c-p_e]\), discretized into \(m\) equally spaced points.
The state space \(S\) contains all pairs \((p_1, p_2)\), so \(|S| = m^2\).
"""
        )


def _render_tab_training() -> None:
    with ui.column().classes("w-full gap-4"):
        # Environment settings and parameters
        with ui.expansion("💰 Environment Settings", value=True).classes("w-full"):
            with ui.row().classes("w-full gap-4"):
                k1 = ui.number(
                    "k₁", value=7.0, min=0.1, max=20.0, step=0.1
                ).classes("w-1/4")
                k2 = ui.number(
                    "k₂", value=0.5, min=0.0, max=1.0, step=0.01
                ).classes("w-1/4")
                c = ui.number(
                    "c (Marginal Cost)", value=2.0, min=0.0, max=10.0, step=0.1
                ).classes("w-1/4")
                m = (
                    ui.select(
                        ["4", "7", "10", "13", "16"],
                        value="7",
                    )
                    .props('label="m (Action space size)"')
                    .classes("w-1/4")
                )

            # Derived prices
            info_label = ui.label("").classes("text-sm text-gray-700")

            start_mode = (
                ui.radio(
                    ["Randomised", "Fixed"],
                    value="Randomised",
                )
                .props('inline label="Starting prices for initialization"')
                .classes("")
            )

            fixed_start_p1 = ui.number(
                "Starting price for Alice (p₁)", value=7.0, step=0.1
            )
            fixed_start_p2 = ui.number(
                "Starting price for Bob (p₂)", value=7.0, step=0.1
            )

            def update_price_info() -> None:
                try:
                    prices, p_e, p_c, _, _ = calculate_prices(
                        float(k1.value),
                        float(k2.value),
                        float(c.value),
                        int(m.value or "7"),
                    )
                    price_start = prices[0]
                    price_end = prices[-1]
                    prices_display = ", ".join(f"{p:.1f}" for p in prices)
                    info_label.text = (
                        f"Equilibrium price p_e = {p_e:.1f} | "
                        f"Collusion price p_c = {p_c:.1f} | "
                        f"Price range: [{price_start:.1f}, {price_end:.1f}] | "
                        f"Action space A = {{{prices_display}}}"
                    )
                except Exception:  # pragma: no cover - defensive
                    info_label.text = "Error calculating prices."

            for control in (k1, k2, c, m):
                control.on("update:model-value", lambda e: update_price_info())
            update_price_info()

        with ui.expansion("🧠 Q-Learning Parameters", value=True).classes("w-full"):
            with ui.row().classes("w-full gap-4"):
                alpha = (
                    ui.slider(
                        min=0.0,
                        max=1.0,
                        step=0.001,
                        value=0.125,
                    )
                    .props('label="α (Learning Rate)"')
                    .classes("w-1/3")
                )
                delta = (
                    ui.slider(
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=0.95,
                    )
                    .props('label="δ (Discount Factor)"')
                    .classes("w-1/3")
                )
                beta = ui.number(
                    "β (Exponential Decay Rate)",
                    value=2e-5,
                    step=1e-6,
                    format="%.6f",
                ).classes("w-1/3")

        with ui.expansion("⚙️ Advanced Parameters", value=False).classes("w-full"):
            with ui.row().classes("w-full gap-4"):
                seed = ui.number(
                    "Random Seed", value=43, min=0, max=1000, step=1
                ).classes("w-1/4")
                check_every = ui.number(
                    "Check Every", value=1000, min=100, max=10000, step=100
                ).classes("w-1/4")
                stable_required = ui.number(
                    "Stable Required",
                    value=100000,
                    min=1000,
                    max=1000000,
                    step=10000,
                ).classes("w-1/4")
                max_periods = ui.number(
                    "Max Periods",
                    value=2000000,
                    min=10000,
                    max=10000000,
                    step=100000,
                ).classes("w-1/4")

        price_history_container = ui.column().classes("w-full gap-2")
        with ui.row().classes("w-full gap-8"):
            left_controls = ui.column().classes("w-1/3 gap-4")
            right_plots = ui.column().classes("w-2/3 gap-4")

        q_tables_container = ui.column().classes("w-full gap-4")
        trajectory_container = ui.column().classes("w-full gap-4")

        def build_config() -> dict[str, Any]:
            return _build_config_econ(
                k1,
                k2,
                c,
                m,
                start_mode,
                fixed_start_p1,
                fixed_start_p2,
                alpha,
                delta,
                beta,
                seed,
                check_every,
                stable_required,
                max_periods,
            )

        cfg_initial = build_config()
        if f"{TAB_ID}_Q1" not in st.session_state:
            init_session_state_econ(cfg_initial)

        def refresh_view() -> None:
            cfg = build_config()
            display_state = get_display_state_econ(cfg)
            in_playback = is_in_playback_mode_econ(cfg)

            _render_price_history_ui(
                price_history_container,
                display_state["price_history"],
                display_state["skipped_steps"],
                display_state["step_count"],
                display_state.get("starting_prices_picked", True),
            )

            right_plots.clear()
            with right_plots:
                _render_step_log_econ(right_plots, display_state["step_log"])
                _render_training_summary(right_plots, display_state)

            _render_q_tables(
                q_tables_container,
                display_state["q_table_1"],
                display_state["q_table_2"],
            )

            # Greedy trajectory UI (similar to Streamlit render_trajectory_econ)
            _render_greedy_trajectory(trajectory_container, cfg)

            left_controls.clear()
            with left_controls:
                ready = display_state.get("ready_for_training", True)
                starting_picked = display_state.get("starting_prices_picked", True)
                ui.label(
                    f"Ready for training: {'Yes' if ready else 'No'}"
                ).classes("text-sm")

                ui.button(
                    "Reset / Initialize Model",
                    on_click=lambda: (init_session_state_econ(build_config()), refresh_view()),
                ).props("color=primary").classes("w-full")

                if not starting_picked:
                    ui.button(
                        "Pick Random Starting Prices",
                        on_click=lambda: (
                            pick_random_starting_prices_econ(build_config()),
                            refresh_view(),
                        ),
                    ).classes("w-full")

                ui.separator()
                ui.label("Training Controls").classes("font-semibold text-sm")

                btn_step = ui.button(
                    "👟 Take Next Step",
                    on_click=lambda: (
                        (_ensure_live(cfg), step_agent_econ(build_config())),
                        refresh_view(),
                    ),
                ).classes("w-full")
                if in_playback:
                    btn_step.props("disable")

                steps_fast = ui.number(
                    "Fast-forward steps", value=1000, min=1, step=100
                ).classes("w-full")

                def on_fast_forward() -> None:
                    run_batch_training_econ(int(steps_fast.value), build_config())
                    refresh_view()

                btn_ff = ui.button(
                    "⏩ Fast Forward",
                    on_click=on_fast_forward,
                ).classes("w-full")
                if in_playback:
                    btn_ff.props("disable")

                btn_conv = ui.button(
                    "Run Until Convergence",
                    on_click=lambda: (
                        run_until_convergence_econ(build_config()),
                        refresh_view(),
                    ),
                ).classes("w-full")
                if in_playback:
                    btn_conv.props("disable")

                ui.separator()
                ui.label("Playback Controls").classes("font-semibold text-sm")
                with ui.row().classes("gap-2"):
                    ui.button(
                        "⏮️ Initial State",
                        on_click=lambda: (
                            jump_to_start_econ(build_config()),
                            refresh_view(),
                        ),
                    )
                    ui.button(
                        "◀️ Prev action",
                        on_click=lambda: (
                            rewind_checkpoint_econ(build_config()),
                            refresh_view(),
                        ),
                    )
                    ui.button(
                        "Next ▶️",
                        on_click=lambda: (
                            forward_checkpoint_econ(build_config()),
                            refresh_view(),
                        ),
                    )
                    ui.button(
                        "Last ⏭️",
                        on_click=lambda: (
                            jump_to_latest_econ(build_config()),
                            refresh_view(),
                        ),
                    )

        def _ensure_live(cfg: dict[str, Any]) -> None:
            if is_in_playback_mode_econ(cfg):
                jump_to_latest_econ(cfg)

        refresh_view()


def _render_tab_battle() -> None:
    with ui.column().classes("w-full gap-4"):
        ui.markdown(
            """
Export your Q-tables to CSV (from the Training tab) and upload them here to
compete in a pricing battle. The winner is the one with the highest average
profit over the prices/profits cycle.
"""
        )

        battle_state: dict[str, Any] = {
            "df1": None,
            "df2": None,
            "q1_perspective": None,
            "q2_perspective": None,
        }

        with ui.row().classes("w-full gap-4"):
            with ui.column().classes("w-1/2 gap-2"):
                player_1_name = ui.input(
                    "Player 1 name", value="Alice"
                ).classes("w-full")

                table1_container = ui.column().classes("w-full gap-2")

                def on_upload1(e: events.UploadEventArguments) -> None:
                    try:
                        content = e.content.read()
                        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
                        battle_state["df1"] = df
                        table1_container.clear()
                        with table1_container:
                            ui.label("Q-table 1 preview").classes("font-semibold")
                            rows = df.reset_index().to_dict(orient="records")
                            if rows:
                                fields = list(rows[0].keys())
                            else:
                                fields = []
                            cols = [{"label": f, "field": f} for f in fields]
                            ui.table(columns=cols, rows=rows[:20]).classes("text-xs")
                        ui.notify("Loaded Q-table 1", type="positive")
                    except Exception as exc:  # pragma: no cover
                        ui.notify(f"Error loading Q-table 1: {exc}", type="negative")

                ui.upload(
                    on_upload=on_upload1, multiple=False
                ).props('label="Upload Q-table 1 (CSV)"')

                q1_perspective = (
                    ui.select(
                        ["Player 1 (Q1)", "Player 2 (Q2)"],
                    )
                    .props('label="Perspective of Q-table 1"')
                )
                q1_perspective.on(
                    "update:model-value",
                    lambda e: battle_state.update({"q1_perspective": e.value}),
                )

            with ui.column().classes("w-1/2 gap-2"):
                player_2_name = ui.input(
                    "Player 2 name", value="Bob"
                ).classes("w-full")
                table2_container = ui.column().classes("w-full gap-2")

                def on_upload2(e: events.UploadEventArguments) -> None:
                    try:
                        content = e.content.read()
                        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
                        battle_state["df2"] = df
                        table2_container.clear()
                        with table2_container:
                            ui.label("Q-table 2 preview").classes("font-semibold")
                            rows = df.reset_index().to_dict(orient="records")
                            if rows:
                                fields = list(rows[0].keys())
                            else:
                                fields = []
                            cols = [{"label": f, "field": f} for f in fields]
                            ui.table(columns=cols, rows=rows[:20]).classes("text-xs")
                        ui.notify("Loaded Q-table 2", type="positive")
                    except Exception as exc:  # pragma: no cover
                        ui.notify(f"Error loading Q-table 2: {exc}", type="negative")

                ui.upload(
                    on_upload=on_upload2, multiple=False
                ).props('label="Upload Q-table 2 (CSV)"')

                q2_perspective = (
                    ui.select(
                        ["Player 1 (Q1)", "Player 2 (Q2)"],
                    )
                    .props('label="Perspective of Q-table 2"')
                )
                q2_perspective.on(
                    "update:model-value",
                    lambda e: battle_state.update({"q2_perspective": e.value}),
                )

        ui.separator()
        ui.label("Environment Parameters").classes("font-semibold")
        with ui.row().classes("w-full gap-4"):
            battle_k1 = ui.number(
                "k₁", value=7.0, min=0.1, max=20.0, step=0.1
            ).classes("w-1/3")
            battle_k2 = ui.number(
                "k₂", value=0.5, min=0.0, max=1.0, step=0.01
            ).classes("w-1/3")
            battle_c = ui.number(
                "c (Marginal Cost)", value=2.0, min=0.0, max=10.0, step=0.1
            ).classes("w-1/3")

        ui.separator()
        ui.label("Starting Prices for Battle").classes("font-semibold")
        with ui.row().classes("w-full gap-4"):
            battle_start_p1 = ui.number(
                "Starting price for Player 1", value=7.0, min=0.1, max=50.0, step=0.1
            ).classes("w-1/2")
            battle_start_p2 = ui.number(
                "Starting price for Player 2", value=7.0, min=0.1, max=50.0, step=0.1
            ).classes("w-1/2")

        result_container = ui.column().classes("w-full gap-4")

        def compute_battle() -> None:
            df1 = battle_state.get("df1")
            df2 = battle_state.get("df2")
            if df1 is None or df2 is None:
                ui.notify(
                    "Please upload both Q-table CSV files before computing the trajectory.",
                    type="warning",
                )
                return

            q1_p = battle_state.get("q1_perspective")
            q2_p = battle_state.get("q2_perspective")
            if q1_p is None or q2_p is None:
                ui.notify(
                    "Please select the perspective for both uploaded Q-tables.",
                    type="warning",
                )
                return

            def extract_prices_from_columns(df: pd.DataFrame) -> list[float]:
                prices: list[float] = []
                for col in df.columns:
                    if "price=" in col:
                        try:
                            prices.append(float(col.replace("price=", "")))
                        except ValueError:
                            pass
                return sorted(prices)

            prices1 = extract_prices_from_columns(df1)
            prices2 = extract_prices_from_columns(df2)

            if not prices1 or not prices2:
                ui.notify(
                    "Could not extract prices from Q-table columns. "
                    "Expected format: 'price=X.X'",
                    type="negative",
                )
                return

            if prices1 != prices2:
                ui.notify(
                    "The two Q-tables use different price sets. Using prices from the first Q-table.",
                    type="warning",
                )
                prices = prices1
            else:
                prices = prices1

            tolerance = 1e-3
            p1_val = float(battle_start_p1.value)
            p2_val = float(battle_start_p2.value)
            p1_valid = any(abs(p1_val - p) < tolerance for p in prices)
            p2_valid = any(abs(p2_val - p) < tolerance for p in prices)
            if not p1_valid or not p2_valid:
                ui.notify(
                    "Starting prices must be close to prices in the action space. "
                    f"Valid prices: {[f'{p:.1f}' for p in prices]}",
                    type="negative",
                )
                return

            original_p1 = p1_val
            original_p2 = p2_val
            p1_val = min(prices, key=lambda p: abs(p - original_p1))
            p2_val = min(prices, key=lambda p: abs(p - original_p2))

            n_actions = len(prices)
            n_states = n_actions * n_actions

            k1_val = float(battle_k1.value)
            k2_val = float(battle_k2.value)
            c_val = float(battle_c.value)

            p_e = (k1_val + c_val) / (2 - k2_val)
            profit_e = (p_e - c_val) * demand1(p_e, p_e, k1_val, k2_val)
            p_c = (2 * k1_val + 2 * c_val * (1 - k2_val)) / (4 * (1 - k2_val))
            profit_c = (p_c - c_val) * demand1(p_c, p_c, k1_val, k2_val)

            if df1.shape != (n_states, n_actions) or df2.shape != (n_states, n_actions):
                ui.notify(
                    f"Q-table dimensions don't match expected size "
                    f"({n_states}, {n_actions}).",
                    type="negative",
                )
                return

            Q1 = df1.values.copy()
            Q2 = df2.values.copy()

            if q1_p == "Player 2 (Q2)":
                Q1 = flip_q_table_states(Q1, prices)
                ui.notify(
                    "Q-table 1 was flipped (uploaded from Player 2 perspective).",
                    type="info",
                )

            if q2_p == "Player 1 (Q1)":
                Q2 = flip_q_table_states(Q2, prices)
                ui.notify(
                    "Q-table 2 was flipped (uploaded from Player 1 perspective).",
                    type="info",
                )

            rng = np.random.default_rng(43)

            traj = follow_greedy_until_loop_econ(
                Q1,
                Q2,
                p1_val,
                p2_val,
                prices,
                rng,
                max_steps=50000,
            )

            # Persist in session state for potential later use
            st.session_state["battle_trajectory"] = traj
            st.session_state["battle_prices"] = prices
            st.session_state["battle_cfg_k1"] = k1_val
            st.session_state["battle_cfg_k2"] = k2_val
            st.session_state["battle_cfg_c"] = c_val
            st.session_state["battle_cfg_start"] = (p1_val, p2_val)
            st.session_state["battle_cfg_p_e"] = p_e
            st.session_state["battle_cfg_p_c"] = p_c
            st.session_state["battle_cfg_profit_e"] = profit_e
            st.session_state["battle_cfg_profit_c"] = profit_c

            _render_battle_results(
                result_container,
                traj,
                prices,
                k1_val,
                k2_val,
                c_val,
                p_e,
                p_c,
                profit_e,
                profit_c,
                p1_val,
                p2_val,
                player_1_name.value,
                player_2_name.value,
            )

        ui.button(
            "Compute Trajectory & Determine Winner",
            on_click=compute_battle,
        ).props("color=primary")

        # If a previous trajectory exists, show it on load as well
        if "battle_trajectory" in st.session_state:
            traj = st.session_state["battle_trajectory"]
            prices = st.session_state["battle_prices"]
            k1_val = st.session_state.get("battle_cfg_k1", 7.0)
            k2_val = st.session_state.get("battle_cfg_k2", 0.5)
            c_val = st.session_state.get("battle_cfg_c", 2.0)
            start_p1, start_p2 = st.session_state.get("battle_cfg_start", (7.0, 7.0))
            p_e = st.session_state.get("battle_cfg_p_e", 0.0)
            p_c = st.session_state.get("battle_cfg_p_c", 0.0)
            profit_e = st.session_state.get("battle_cfg_profit_e", 0.0)
            profit_c = st.session_state.get("battle_cfg_profit_c", 0.0)
            _render_battle_results(
                result_container,
                traj,
                prices,
                k1_val,
                k2_val,
                c_val,
                p_e,
                p_c,
                profit_e,
                profit_c,
                start_p1,
                start_p2,
                player_1_name.value,
                player_2_name.value,
            )


def _render_battle_results(
    container: ui.column,
    traj: dict,
    prices: list[float],
    k1_val: float,
    k2_val: float,
    c_val: float,
    p_e: float,
    p_c: float,
    profit_e: float,
    profit_c: float,
    start_p1: float,
    start_p2: float,
    player_1_name: str,
    player_2_name: str,
) -> None:
    container.clear()
    with container:
        path = traj["path"]
        loop = traj["loop"]
        loop_start = traj["loop_start"]

        ui.markdown(
            rf"""
- Equilibrium price: \(p_e = {p_e:.2f}\)
- Collusion price: \(p_c = {p_c:.2f}\)
- Equilibrium profit: \(\pi_e = {profit_e:.2f}\)
- Collusion profit: \(\pi_c = {profit_c:.2f}\)
"""
        )

        ui.label("Battle Results").classes("text-xl font-semibold")

        if loop_start is not None and loop:
            loop_profits_alice: list[float] = []
            loop_profits_bob: list[float] = []
            loop_prices_alice: list[float] = []
            loop_prices_bob: list[float] = []

            for rec in loop:
                p1 = rec["a1_price"]
                p2 = rec["a2_price"]
                loop_prices_alice.append(p1)
                loop_prices_bob.append(p2)
                pi1 = profit1(p1, p2, c_val, k1_val, k2_val)
                pi2 = profit2(p1, p2, c_val, k1_val, k2_val)
                loop_profits_alice.append(pi1)
                loop_profits_bob.append(pi2)

            avg_profit_alice = (
                sum(loop_profits_alice) / len(loop_profits_alice)
                if loop_profits_alice
                else 0.0
            )
            avg_profit_bob = (
                sum(loop_profits_bob) / len(loop_profits_bob)
                if loop_profits_bob
                else 0.0
            )
            avg_p1 = (
                sum(loop_prices_alice) / len(loop_prices_alice)
                if loop_prices_alice
                else 0.0
            )
            avg_p2 = (
                sum(loop_prices_bob) / len(loop_prices_bob)
                if loop_prices_bob
                else 0.0
            )

            if avg_profit_alice > avg_profit_bob:
                winner = f"👩🏼‍💼 {player_1_name} (Player 1)"
                winner_profit = avg_profit_alice
                loser_profit = avg_profit_bob
            elif avg_profit_bob > avg_profit_alice:
                winner = f"🧑🏼‍💼 {player_2_name} (Player 2)"
                winner_profit = avg_profit_bob
                loser_profit = avg_profit_alice
            else:
                winner = "🤝 Tie"
                winner_profit = avg_profit_alice
                loser_profit = avg_profit_bob

            if winner != "🤝 Tie":
                ui.markdown(f"## 🏆 Winner: {winner}!")
                ui.markdown(f"**Average Profit: {winner_profit:.2f}** (vs {loser_profit:.2f})")
            else:
                ui.markdown(f"## {winner}!")
                ui.markdown(
                    f"**Average Profit: {winner_profit:.2f}** (both players)"
                )

            denominator = profit_c - profit_e
            if abs(denominator) > 1e-10:
                normalized_profit_alice = (avg_profit_alice - profit_e) / denominator
                normalized_profit_bob = (avg_profit_bob - profit_e) / denominator
            else:
                normalized_profit_alice = None
                normalized_profit_bob = None

            with ui.row().classes("w-full gap-4"):
                with ui.column().classes("w-1/2 gap-2"):
                    ui.label(f"👩🏼‍💼 {player_1_name} (Player 1)").classes(
                        "font-semibold"
                    )
                    ui.label(
                        f"Average profit: {avg_profit_alice:.2f}"
                    ).classes("text-sm")
                    ui.label(
                        f"Average price: {avg_p1:.2f}"
                    ).classes("text-sm text-gray-700")
                    if normalized_profit_alice is not None:
                        ui.label(
                            f"Normalised average profit: {normalized_profit_alice:.2f}"
                        ).classes("text-sm text-gray-700")
                with ui.column().classes("w-1/2 gap-2"):
                    ui.label(f"🧑🏼‍💼 {player_2_name} (Player 2)").classes(
                        "font-semibold"
                    )
                    ui.label(
                        f"Average profit: {avg_profit_bob:.2f}"
                    ).classes("text-sm")
                    ui.label(
                        f"Average price: {avg_p2:.2f}"
                    ).classes("text-sm text-gray-700")
                    if normalized_profit_bob is not None:
                        ui.label(
                            f"Normalised average profit: {normalized_profit_bob:.2f}"
                        ).classes("text-sm text-gray-700")

            ui.separator()
            ui.label("Cycle Information").classes("text-lg font-semibold")

            # Build full_path as in the Streamlit implementation
            full_path = [
                {"step_num": 0, "a1_price": start_p1, "a2_price": start_p2}
            ]
            for i, rec in enumerate(path, start=1):
                full_path.append(
                    {
                        "step_num": i,
                        "a1_price": rec["a1_price"],
                        "a2_price": rec["a2_price"],
                    }
                )

            alice_row = [f"{rec['a1_price']:.1f}" for rec in full_path]
            bob_row = [f"{rec['a2_price']:.1f}" for rec in full_path]
            step_row = [f"Step {rec['step_num']}" for rec in full_path]

            num_visible_cols = 10
            fixed_col_width = "90px"

            table_id = f"battle_trajectory_table_{id(path)}"
            html_table = f"""
            <div id="{table_id}_container" style="width: 100%; overflow-x: auto; overflow-y: hidden; border: 1px solid #ddd; border-radius: 5px;">
                <table style="border-collapse: collapse; table-layout: fixed;">
                    <tbody>
                        <tr>
                            <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">👩🏼‍💼 {player_1_name}</td>
                            {''.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in alice_row])}
                        </tr>
                        <tr>
                            <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">🧑🏼‍💼 {player_2_name}</td>
                            {''.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in bob_row])}
                        </tr>
                        <tr>
                            <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">Step</td>
                            {''.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; font-size: 0.85em; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in step_row])}
                        </tr>
                    </tbody>
                </table>
            </div>
            """
            ui.html(html_table)
            if len(full_path) > num_visible_cols:
                ui.label(
                    "Scroll horizontally to see all steps. The newest steps appear on the left."
                ).classes("text-xs text-gray-500")

        else:
            ui.label(
                "No cycle detected within max steps. Consider adjusting starting prices or increasing max steps."
            ).classes("text-sm text-amber-700")

