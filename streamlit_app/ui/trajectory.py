"""Trajectory computation and display for Economics Pricing Q-learning demo."""

from __future__ import annotations
import numpy as np
import streamlit as st
from streamlit_app.state_econ import (
    profit1,
    profit2,
    state_index,
    index_to_state,
    argmax_tie,
)

__all__ = ["render_trajectory_econ"]


def greedy_successor_econ(
    Q1: np.ndarray,
    Q2: np.ndarray,
    s: int,
    prices: list[float],
    rng: np.random.Generator,
) -> tuple[int, int, float, float, int]:
    """From state index s, take each firm's greedy action and return:
    (a1, a2, p1_next, p2_next, s_next)
    """

    a1 = argmax_tie(Q1[s], rng)
    a2 = argmax_tie(Q2[s], rng)
    p1_next = prices[a1]
    p2_next = prices[a2]
    s_next = state_index(p1_next, p2_next, prices)
    return a1, a2, p1_next, p2_next, s_next


def follow_greedy_until_loop_econ(
    Q1: np.ndarray,
    Q2: np.ndarray,
    start_p1: float,
    start_p2: float,
    prices: list[float],
    rng: np.random.Generator,
    max_steps: int = 100000,
) -> dict:
    """Follow the greedy map from (start_p1, start_p2) until a cycle is detected.

    Returns a dict with the full path and the detected cycle.
    """

    # Start state
    s = state_index(start_p1, start_p2, prices)

    # Remember when we first saw each state (for loop detection)
    first_seen_at = {}  # state_idx -> time step when first visited
    path = []  # list of dicts describing transitions

    for t in range(max_steps):
        if s in first_seen_at:
            loop_start = first_seen_at[s]
            loop_path = path[loop_start:]  # the cycle
            return {
                "path": path,  # all transitions until loop
                "loop_start": loop_start,  # index in path where loop begins
                "loop": loop_path,  # the loop transitions
            }

        # Mark first visit to this state
        first_seen_at[s] = t

        # Decode current state's (p1, p2) for display
        cur_p1, cur_p2 = index_to_state(s, prices)

        # Take greedy actions and move to successor
        a1, a2, p1_next, p2_next, s_next = greedy_successor_econ(Q1, Q2, s, prices, rng)

        # Record transition
        path.append(
            {
                "t": t,
                "state": s,
                "state_str": f"s({cur_p1:.1f},{cur_p2:.1f})",
                "a1_idx": int(a1),
                "a2_idx": int(a2),
                "a1_price": p1_next,  # "max movement read from Q1"
                "a2_price": p2_next,  # "max movement read from Q2"
                "next_state": s_next,
                "next_state_str": f"s({p1_next:.1f},{p2_next:.1f})",
            }
        )

        # Advance
        s = s_next

    # If we hit max_steps without finding a loop
    return {"path": path, "loop_start": None, "loop": []}


def render_trajectory_econ(config: dict) -> None:  # noqa: ARG001
    """Render trajectory computation UI for economics pricing.

    Args:
        config: Configuration dict with 'tab_id' key
        display_state: Display state dict with Q-tables and prices
    """
    tab_id = config.get("tab_id", "default")

    # Get available prices from session state
    prices = st.session_state.get(f"{tab_id}_prices", [])
    if not prices:
        st.info("Initialize the model first to compute trajectories.")
        return
    prices_display = [f"{p:.1f}" for p in prices]

    st.subheader("Greedy Trajectory")
    st.markdown(
        rf"""Compute the greedy trajectory starting from a given pair of prices.
        <br>
        The trajectory follows the best-response actions for both players until a cycle is detected.<br>
        <br>
        Pick from the action space $A$ = {{{', '.join(prices_display)}}}.
        """,
        unsafe_allow_html=True,
    )
    # Check if Q-tables have been trained (need at least 1000 steps)
    step_count = st.session_state.get(f"{tab_id}_step_count", 0)
    min_steps_required = 1000
    if step_count < min_steps_required:
        st.warning(
            f"⚠️ The model needs to be trained for at least {min_steps_required:,} steps before computing greedy trajectories. "
            f"Current steps: {step_count:,}. Please train the model first."
        )
        return

    # Input fields for starting prices
    col1, col2 = st.columns(2)
    with col1:
        start_p1 = st.number_input(
            r"Starting price for Alice ($p_1$):",
            min_value=float(min(prices)),
            max_value=float(max(prices)),
            value=float(prices[len(prices) // 2]),
            step=0.1,
            key=f"{tab_id}_traj_p1",
            help="Starting price for player 1 (Alice)",
        )
    with col2:
        start_p2 = st.number_input(
            r"Starting price for Bob ($p_2$):",
            min_value=float(min(prices)),
            max_value=float(max(prices)),
            value=float(prices[len(prices) // 2]),
            step=0.1,
            key=f"{tab_id}_traj_p2",
            help="Starting price for player 2 (Bob)",
        )

    # Validate and normalize prices to exact values in PRICES list (handles floating-point precision)
    tolerance = 5e-2
    p1_valid = any(abs(start_p1 - p) <= tolerance for p in prices)
    p2_valid = any(abs(start_p2 - p) <= tolerance for p in prices)

    if not p1_valid or not p2_valid:
        st.warning(
            f"⚠️ Prices must be in the action space: {{{', '.join([f'{p:.1f}' for p in prices])}}}. "
            f"Please select valid prices."
        )
        return

    # Normalize to exact values in PRICES to avoid floating-point precision issues
    start_p1 = min(prices, key=lambda p: abs(p - start_p1))
    start_p2 = min(prices, key=lambda p: abs(p - start_p2))

    # Button to compute trajectory
    if st.button("Compute Greedy Trajectory", key=f"{tab_id}_compute_traj"):
        # Get Q-tables from session state
        Q1 = st.session_state.get(f"{tab_id}_Q1")
        Q2 = st.session_state.get(f"{tab_id}_Q2")
        rng = st.session_state.get(f"{tab_id}_rng")

        if Q1 is None or Q2 is None:
            st.error("Q-tables not initialized. Please initialize the model first.")
            return

        if rng is None:
            rng = np.random.default_rng(43)

        # Compute trajectory
        with st.spinner("Computing trajectory..."):
            traj = follow_greedy_until_loop_econ(
                Q1, Q2, start_p1, start_p2, prices, rng, max_steps=50000
            )

        # Store in session state for display
        st.session_state[f"{tab_id}_trajectory"] = traj
        st.session_state[f"{tab_id}_trajectory_start"] = (start_p1, start_p2)
        st.rerun()

    # Display trajectory if available
    traj = st.session_state.get(f"{tab_id}_trajectory")
    traj_start = st.session_state.get(f"{tab_id}_trajectory_start")

    if traj and traj_start:
        # Check if the trajectory matches current starting prices
        if traj_start != (start_p1, start_p2):
            # Clear old trajectory if starting prices changed
            st.session_state[f"{tab_id}_trajectory"] = None
            st.session_state[f"{tab_id}_trajectory_start"] = None
        else:
            # Display trajectory
            path = traj["path"]
            loop = traj["loop"]
            loop_start = traj["loop_start"]

            st.markdown(r"**Trajectory Steps for $p_1, p_2$ :**")

            # Display path in table format (similar to price history)
            if path:
                # Add step 0 with starting prices at the beginning
                # Renumber path steps to 1, 2, 3... (since path starts from t=0)
                full_path = [
                    {
                        "step_num": 0,
                        "a1_price": start_p1,
                        "a2_price": start_p2,
                    }
                ]
                # Add path steps, renumbering them starting from 1
                for i, rec in enumerate(path, start=1):
                    full_path.append(
                        {
                            "step_num": i,
                            "a1_price": rec["a1_price"],
                            "a2_price": rec["a2_price"],
                        }
                    )

                # Build table data: rows are Alice, Bob, Step; columns are trajectory steps
                alice_row = []
                bob_row = []
                step_row = []

                # Fixed column width: calculate width to show 10 columns consistently
                num_visible_cols = 10
                fixed_col_width = "90px"  # Fixed width per column

                for rec in full_path:
                    alice_row.append(f"{rec['a1_price']:.1f}")
                    bob_row.append(f"{rec['a2_price']:.1f}")
                    step_row.append(f"Step {rec['step_num']}")

                # Create HTML table with horizontal scroll, no header, fixed column width
                table_id = f"trajectory_table_{id(path)}"
                html_table = f"""
                <div id="{table_id}_container" style="width: 100%; overflow-x: auto; overflow-y: hidden; border: 1px solid #ddd; border-radius: 5px;">
                    <table style="border-collapse: collapse; table-layout: fixed;">
                        <tbody>
                            <tr>
                                <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">👩🏼‍💼 Alice p₁</td>
                                {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in alice_row])}
                            </tr>
                            <tr>
                                <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">🧑🏼‍💼 Bob p₂</td>
                                {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in bob_row])}
                            </tr>
                            <tr>
                                <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">Step</td>
                                {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; font-size: 0.85em; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in step_row])}
                            </tr>
                        </tbody>
                    </table>
                </div>
                """

                st.markdown(html_table, unsafe_allow_html=True)

                if len(full_path) > num_visible_cols:
                    st.caption(
                        "Scroll horizontally to see all steps. The newest steps appear on the left."
                    )

            # Trajectory table for profit steps
            st.markdown(r"**Trajectory Steps for $\pi_1, \pi_2$ :**")

            # Get profit calculation parameters from session state
            c = st.session_state.get(f"{tab_id}_cfg_c", 0.0)
            k1 = st.session_state.get(f"{tab_id}_cfg_k1", 1.0)
            k2 = st.session_state.get(f"{tab_id}_cfg_k2", 1.0)
            p_e = st.session_state.get(f"{tab_id}_cfg_p_e", 0.0)
            p_c = st.session_state.get(f"{tab_id}_cfg_p_c", 0.0)
            profit_e = st.session_state.get(f"{tab_id}_cfg_profit_e", 0.0)
            profit_c = st.session_state.get(f"{tab_id}_cfg_profit_c", 0.0)

            # Calculate average profits in the loop
            average_profit_alice = 0.0
            average_profit_bob = 0.0
            if loop_start is not None and loop:
                loop_profits_alice = []
                loop_profits_bob = []
                loop_prices_alice = []
                loop_prices_bob = []
                for rec in loop:
                    p1 = rec["a1_price"]
                    p2 = rec["a2_price"]
                    loop_prices_alice.append(p1)
                    loop_prices_bob.append(p2)
                    pi1 = profit1(p1, p2, c, k1, k2)
                    pi2 = profit2(p1, p2, c, k1, k2)
                    loop_profits_alice.append(pi1)
                    loop_profits_bob.append(pi2)
                average_p1 = (
                    sum(loop_prices_alice) / len(loop_prices_alice)
                    if loop_prices_alice
                    else 0.0
                )
                average_p2 = (
                    sum(loop_prices_bob) / len(loop_prices_bob)
                    if loop_prices_bob
                    else 0.0
                )
                average_profit_alice = (
                    sum(loop_profits_alice) / len(loop_profits_alice)
                    if loop_profits_alice
                    else 0.0
                )
                average_profit_bob = (
                    sum(loop_profits_bob) / len(loop_profits_bob)
                    if loop_profits_bob
                    else 0.0
                )

            if path:
                # Calculate profits for each step in full_path
                alice_profit_row = []
                bob_profit_row = []
                step_row_profit = []

                for rec in full_path:
                    p1 = rec["a1_price"]
                    p2 = rec["a2_price"]
                    pi1 = profit1(p1, p2, c, k1, k2)
                    pi2 = profit2(p1, p2, c, k1, k2)
                    alice_profit_row.append(f"{pi1:.2f}")
                    bob_profit_row.append(f"{pi2:.2f}")
                    step_row_profit.append(f"Step {rec['step_num']}")

                # Create HTML table for profits with horizontal scroll, fixed column width
                table_id_profit = f"trajectory_profit_table_{id(path)}"
                html_table_profit = f"""
                <div id="{table_id_profit}_container" style="width: 100%; overflow-x: auto; overflow-y: hidden; border: 1px solid #ddd; border-radius: 5px;">
                    <table style="border-collapse: collapse; table-layout: fixed;">
                        <tbody>
                            <tr>
                                <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">👩🏼‍💼 Alice π₁</td>
                                {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in alice_profit_row])}
                            </tr>
                            <tr>
                                <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">🧑🏼‍💼 Bob π₂</td>
                                {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in bob_profit_row])}
                            </tr>
                            <tr>
                                <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">Step</td>
                                {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; font-size: 0.85em; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in step_row_profit])}
                            </tr>
                        </tbody>
                    </table>
                </div>
                """

                st.markdown(html_table_profit, unsafe_allow_html=True)

                if len(full_path) > num_visible_cols:
                    st.caption(
                        "💡 Scroll horizontally to see all steps. The newest steps appear on the right."
                    )

            # Display cycle if detected
            if loop_start is not None and loop:
                st.markdown(
                    f"""Cycle detected: starts at step {loop_start}, ends at step {loop_start + len(loop)-1}, length = {len(loop)}<br>
                    <br>
                    The average price for Alice is {average_p1:.1f}, and for Bob is {average_p2:.1f};<br>
                    and the average profit for Alice is {average_profit_alice:.2f}, and for Bob is {average_profit_bob:.2f}.<br>
                    <br>
                    Remember that the equilibrium price is {p_e:.1f}, and the collusion price is {p_c:.1f}; 
                    <br>
                    and the profit of nash equilibrium is {profit_e:.2f}, and the profit of collusion is {profit_c:.2f}.<br>
                    """,
                    unsafe_allow_html=True,
                )
                # Calculate normalized profits
                denominator = profit_c - profit_e
                if abs(denominator) > 1e-10:  # Avoid division by zero
                    normalized_profit_alice = (
                        average_profit_alice - profit_e
                    ) / denominator
                    normalized_profit_bob = (
                        average_profit_bob - profit_e
                    ) / denominator
                    st.markdown(
                        rf"""
                        The normalised profit is calculated as the difference between the average profit and the equilibrium profit, 
                        divided by the difference between collusion profit and equilibrium profit: <br>
                        $\Delta = \dfrac{{\pi_{{\text{{avg}}}} - \pi_{{\text{{equilibrium}}}}}}{{\pi_{{\text{{collusion}}}} - \pi_{{\text{{equilibrium}}}}}}$. <br>
                        <br>
                        Hence the normalised profit <br>
                        for Alice is: $\Delta_{{\text{{Alice}}}} = \dfrac{{{average_profit_alice:.2f} - {profit_e:.2f}}}{{{profit_c:.2f} - {profit_e:.2f}}} = {normalized_profit_alice:.2f}$, <br>
                        and for Bob is: $\Delta_{{\text{{Bob}}}} = \dfrac{{{average_profit_bob:.2f} - {profit_e:.2f}}}{{{profit_c:.2f} - {profit_e:.2f}}} = {normalized_profit_bob:.2f}$.<br>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        r"""
                        The normalised profit cannot be calculated because the denominator 
                        (collusion profit - equilibrium profit) is zero.
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info(
                    "No cycle detected within max_steps. Consider increasing max_steps."
                )
