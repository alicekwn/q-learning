"""Economics Pricing Example with Q-learning"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root (containing `streamlit_app`) is on sys.path
ROOT = Path(__file__).resolve().parent.parent.parent  # project root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import streamlit as st
import pandas as pd
from streamlit_app.ui.controls import parameters_econ
from streamlit_app.ui.training import render_training_controls_econ
from streamlit_app.ui.bellman import render_bellman_log_econ
from streamlit_app.ui.econ_visualization import render_price_history
from streamlit_app.ui.trajectory import (
    render_trajectory_econ,
    follow_greedy_until_loop_econ,
)
from streamlit_app.state_econ import (
    init_session_state_econ,
    step_agent_econ,
    run_batch_training_econ,
    run_until_convergence_econ,
    get_display_state_econ,
    is_in_playback_mode_econ,
    profit1,
    profit2,
    demand1,
    flip_q_table_states,
)

st.set_page_config(page_title="Pricing Strategies in Economics", layout="wide")

st.title("Pricing Strategies in Economics")
st.info(
    "This page demonstrates Q-learning applied to economics pricing with two players learning optimal pricing strategies."
)

tab_1, tab_2, tab_3 = st.tabs(["Game Theory", "Training", "Pricing Battle"])

with tab_1:
    st.markdown(
        r"""
        ### Using game theory to find the equilibrium price and the collusive price

        This is a demo version of economics games scenario. It assumes there are only 2 parties, and we used a simplified equation for demand:

        $q_1=k_1 - p_1 + k_2 \cdot p_2$, $\quad q_2=k_1 - p_2 + k_2 \cdot p_1$ 

        $\pi_1 = (p_1 - c) \times q_1$, $\quad \pi_2 = (p_2 - c) \times q_2$.

        The equilibrium price $p_e$ is calculated as 

        $p_e=\frac{k_1 + c}{2 - k_2}$.

        The collusive price $p_c$ is calculated as 
        
        $p_c=\frac{2 k_1 + 2c (1-k_2)}{4(1-k_2)}$.

        And the equilibrium profit $\pi_e$ is calculated as $\pi_e = (p_e - c) \times q_1$.

        The collusive profit $\pi_c$ is calculated as $\pi_c = (p_c - c) \times q_1$.

        For the feasible prices which each party can take is within the interval $[p_e-\xi(p_c-p_e), p_c+\xi(p_c-p_e)]$.

        To simplify the problem, we assume $\xi=1$, then the feasible prices would be within the interval $[2p_e-p_c,2p_c-p_e]$.

        Note that $m\in \{4+3\sigma, \sigma=0,1,2,...\}$, for these 4 prices: $2p_e-p_c, p_e, p_c, 2p_c-p_e$ to be included in $A$.

        The action space $A$ would be the set of all possible prices within the interval, by choosing $m$ equally spaced points inside the interval.

        $A=\{2p_e-p_c, 2p_e-p_c + \delta, 2p_e-p_c + 2\delta, ..., 2p_c-p_e\}$, where $\delta = \frac{(2p_c-p_e) - (2p_e-p_c)}{m-1} = \frac{3(p_c-p_e)}{m-1}$.

        

        The state space $S$ would be the set of all possible combinations of prices chosen by the two players, i.e., $S=\{(p_1,p_2) | p_1,p_2 \in A\}$.

        The number of states $|S|=m\times m = m^2$.
        """
    )

    with st.expander("Example", expanded=True):
        st.markdown(
            r"""
            When $c=2, k_1= 7, k_2=0.5$, $p_e=6, p_c=8$, the feasible prices would be within the interval $[4,10]$.

            Assume there are $m=7$ equally spaced points in the interval, then the action space for each player would be 

            $A=\{4,5,6,7,8,9,10\}$.

            The state space would be $S=\{(4,4),(4,5),(4,6),...,(4,10),(5,4),(5,5),(5,6),...,(5,10),(6,4),(6,5),......(10,10)\}$, 

            where $|S|=7\times 7 = 49$.
            """
        )

with tab_2:

    st.header("Training of Q-value update in Economics Pricing Scenario")
    st.markdown(
        "This section demonstrates how the Q-value updates in this economics pricing scenario. "
        "You can step through the training process and observe how both players learn optimal pricing strategies."
    )

    # --- A. TOP PANEL: Controls ---
    config_demo = parameters_econ("training")

    # Reset button
    col_reset, col_spacer = st.columns([1, 5])
    with col_reset:
        if st.button(
            "Reset / Initialize Model",
            type="primary",
            key="demo_reset",
            help="After changing any of the settings / parameters above, click the button to reset the model and start training.",
        ):
            init_session_state_econ(config_demo)
            st.rerun()

    st.markdown("---")

    # Initialize on first load
    if f"{config_demo.get('tab_id', 'training')}_Q1" not in st.session_state:
        init_session_state_econ(config_demo)

    display_state = get_display_state_econ(config_demo)
    in_playback = is_in_playback_mode_econ(config_demo)

    # --- B. PRICE HISTORY VISUALIZATION ---
    render_price_history(
        display_state["price_history"],
        display_state["skipped_steps"],
        display_state["step_count"],
        display_state.get("starting_prices_picked", True),
    )

    st.markdown("---")

    # --- C. CONTROLS & DISPLAY AREA ---
    col_left, col_right = st.columns([1, 2])

    with col_left:
        # Training controls
        render_training_controls_econ(
            config_demo,
            display_state,
            in_playback,
            step_agent_econ,
            run_batch_training_econ,
            run_until_convergence_econ,
        )

    with col_right:
        # Bellman logs
        render_bellman_log_econ(display_state["step_log"])

    st.markdown("---")

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        with st.expander(r"Current Q-Matrix (Alice - $Q_1$)", expanded=True):
            st.dataframe(
                display_state["q_table_1"].style.highlight_max(
                    axis=1, color="lightgreen"
                ),
                width="stretch",
                height=400,
            )

    with col_q2:
        with st.expander(r"Current Q-Matrix (Bob - $Q_2$)", expanded=True):
            st.dataframe(
                display_state["q_table_2"].style.highlight_max(
                    axis=1, color="lightgreen"
                ),
                width="stretch",
                height=400,
            )

    st.markdown("---")

    # Trajectory section
    render_trajectory_econ(config_demo)


with tab_3:
    st.header("Pricing Battle")
    st.markdown(
        """
        Export your Q-tables to a csv file (from the Demo page) and upload it to this pricing battle page, and compete with other players. The winner is the one with the highest average profit of the prices/profits cycle.
        
        **Important:** When uploading a Q-table, you must specify whether it was trained from Player 1's perspective (Q1) or Player 2's perspective (Q2). 
        The state encoding s(p1, p2) depends on which player's perspective the Q-table was trained from. If you upload a Q-table with the wrong perspective, 
        the system will automatically flip the state indices to match the battle role.
        """
    )
    # Initialize perspective variables (will be set when files are uploaded)
    q1_perspective = None
    q2_perspective = None

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        # upload 1st Q-table
        player_1_name = st.text_input(
            "Enter your name", value="Alice", key="player_1_name"
        )
        uploaded_file1 = st.file_uploader(
            "Upload your 1st Q-tables csv file", type="csv", key="q_table_upload_1"
        )
        if uploaded_file1 is not None:
            # Reset file pointer and read
            uploaded_file1.seek(0)
            df1_display = pd.read_csv(uploaded_file1)
            st.dataframe(df1_display)
            q1_perspective = st.selectbox(
                "This Q-table is from the perspective of:",
                ["Player 1 (Q1)", "Player 2 (Q2)"],
                key="q1_perspective",
                help="Select whether this Q-table was trained from Player 1's perspective (Q1) or Player 2's perspective (Q2). This will be used to correctly interpret the state encoding.",
            )

    with col_q2:
        # upload 2nd Q-table
        player_2_name = st.text_input(
            "Enter your name", value="Bob", key="player_2_name"
        )
        uploaded_file2 = st.file_uploader(
            "Upload your 2nd Q-tables csv file", type="csv", key="q_table_upload_2"
        )
        if uploaded_file2 is not None:
            # Reset file pointer and read
            uploaded_file2.seek(0)
            df2_display = pd.read_csv(uploaded_file2)
            st.dataframe(df2_display)
            q2_perspective = st.selectbox(
                "This Q-table is from the perspective of:",
                ["Player 1 (Q1)", "Player 2 (Q2)"],
                key="q2_perspective",
                help="Select whether this Q-table was trained from Player 1's perspective (Q1) or Player 2's perspective (Q2). This will be used to correctly interpret the state encoding.",
            )

    st.markdown("---")

    # Environment parameters for the battle (must match the Q-tables)
    st.subheader("Environment Parameters")
    st.info(
        "⚠️ These parameters must match the ones used to train the uploaded Q-tables."
    )

    col_k1, col_k2, col_c = st.columns(3)
    with col_k1:
        battle_k1 = st.number_input(
            r"$k_1$",
            min_value=0.1,
            max_value=20.0,
            value=7.0,
            step=0.1,
            key="battle_k1",
            help="Parameter in demand functions: q1 = k1 - p1 + k2 * p2",
        )
    with col_k2:
        battle_k2 = st.number_input(
            r"$k_2$",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            key="battle_k2",
            help="Cross-price parameter in demand functions",
        )
    with col_c:
        battle_c = st.number_input(
            r"$c$ (Marginal Cost)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            key="battle_c",
            help="Marginal cost for both players",
        )

    st.markdown("---")

    # Starting prices for the battle
    st.subheader("Starting Prices for Battle")
    col_start_p1, col_start_p2 = st.columns(2)
    with col_start_p1:
        battle_start_p1 = st.number_input(
            "Starting price for Player 1",
            min_value=0.1,
            max_value=50.0,
            value=7.0,
            step=0.1,
            key="battle_start_p1",
        )
    with col_start_p2:
        battle_start_p2 = st.number_input(
            "Starting price for Player 2",
            min_value=0.1,
            max_value=50.0,
            value=7.0,
            step=0.1,
            key="battle_start_p2",
        )

    st.markdown("---")

    if st.button(
        "Compute Trajectory & Determine Winner",
        key="compute_trajectory",
        type="primary",
    ):
        # Check if both files are uploaded
        if uploaded_file1 is None or uploaded_file2 is None:
            st.error(
                "❌ Please upload both Q-table CSV files before computing the trajectory."
            )
            st.stop()

        try:
            # Reset file pointers and load Q-tables from CSV
            uploaded_file1.seek(0)
            uploaded_file2.seek(0)
            df1 = pd.read_csv(uploaded_file1, index_col=0)
            df2 = pd.read_csv(uploaded_file2, index_col=0)

            # Extract prices from column names (format: "price=X.X")
            def extract_prices_from_columns(df):
                prices = []
                for col in df.columns:
                    if "price=" in col:
                        price_str = col.replace("price=", "")
                        try:
                            prices.append(float(price_str))
                        except ValueError:
                            pass
                return sorted(prices)

            prices1 = extract_prices_from_columns(df1)
            prices2 = extract_prices_from_columns(df2)

            # Validate that both Q-tables use the same prices
            if len(prices1) == 0 or len(prices2) == 0:
                st.error(
                    "❌ Could not extract prices from Q-table columns. Expected format: 'price=X.X'"
                )
                st.stop()

            if prices1 != prices2:
                st.warning(
                    "⚠️ The two Q-tables use different price sets. Using prices from the first Q-table."
                )
                st.info(f"Q-table 1 prices: {[f'{p:.1f}' for p in prices1]}")
                st.info(f"Q-table 2 prices: {[f'{p:.1f}' for p in prices2]}")
                prices = prices1
            else:
                prices = prices1

            # Validate starting prices are in prices and normalize to exact values, to avoid floating-point precision issues
            tolerance = 1e-3
            p1_valid = any(abs(battle_start_p1 - p) < tolerance for p in prices)
            p2_valid = any(abs(battle_start_p2 - p) < tolerance for p in prices)

            if not p1_valid or not p2_valid:
                st.error(
                    f"❌ Starting prices must be within tolerance of prices in the action space. "
                    f"Valid prices: {[f'{p:.1f}' for p in prices]}"
                )
                st.stop()

            # Store original values for comparison
            original_p1 = battle_start_p1
            original_p2 = battle_start_p2

            # Always normalize to exact values from 'prices' to avoid floating-point precision issues
            battle_start_p1 = min(prices, key=lambda p: abs(p - original_p1))
            battle_start_p2 = min(prices, key=lambda p: abs(p - original_p2))

            # Only show info if values were actually adjusted
            if (
                abs(battle_start_p1 - original_p1) > 1e-10
                or abs(battle_start_p2 - original_p2) > 1e-10
            ):
                st.info(
                    f"ℹ️ Starting prices normalized to: p1={battle_start_p1:.1f}, p2={battle_start_p2:.1f}"
                )

            # Validate Q-table dimensions
            n_actions = len(prices)
            n_states = n_actions * n_actions

            # Calculate equilibrium and collusion prices and profits
            # p_e = (k1 + c) / (2 - k2)
            p_e = (battle_k1 + battle_c) / (2 - battle_k2)
            # profit_e = (p_e - c) * demand1(p_e, p_e, k1, k2)
            profit_e = (p_e - battle_c) * demand1(p_e, p_e, battle_k1, battle_k2)
            # p_c = (2*k1 + 2*c*(1-k2)) / (4*(1-k2))
            p_c = (2 * battle_k1 + 2 * battle_c * (1 - battle_k2)) / (
                4 * (1 - battle_k2)
            )
            # profit_c = (p_c - c) * demand1(p_c, p_c, k1, k2)
            profit_c = (p_c - battle_c) * demand1(p_c, p_c, battle_k1, battle_k2)

            if df1.shape != (n_states, n_actions) or df2.shape != (n_states, n_actions):
                st.error(
                    f"❌ Q-table dimensions don't match expected size. "
                    f"Expected: ({n_states}, {n_actions}), "
                    f"Got: Q-table 1: {df1.shape}, Q-table 2: {df2.shape}"
                )
                st.stop()

            # Check if perspective selection was made for both Q-tables
            if q1_perspective is None or q2_perspective is None:
                st.error(
                    "❌ Please select the perspective for both uploaded Q-tables before computing the trajectory."
                )
                st.stop()

            # Convert DataFrames to numpy arrays
            Q1 = df1.values.copy()
            Q2 = df2.values.copy()

            # Apply state flipping based on perspective vs battle role
            # Q1 is used for Player 1: flip if uploaded Q-table is from Player 2's perspective
            if q1_perspective == "Player 2 (Q2)":
                Q1 = flip_q_table_states(Q1, prices)
                st.info(
                    "ℹ️ Q-table 1 was flipped because it was from Player 2's perspective but is being used for Player 1."
                )

            # Q2 is used for Player 2: flip if uploaded Q-table is from Player 1's perspective
            if q2_perspective == "Player 1 (Q1)":
                Q2 = flip_q_table_states(Q2, prices)
                st.info(
                    "ℹ️ Q-table 2 was flipped because it was from Player 1's perspective but is being used for Player 2."
                )

            # Initialize random number generator
            rng = np.random.default_rng(43)

            # Compute trajectory
            with st.spinner("🔄 Computing trajectory and detecting cycle..."):
                traj = follow_greedy_until_loop_econ(
                    Q1,
                    Q2,
                    battle_start_p1,
                    battle_start_p2,
                    prices,
                    rng,
                    max_steps=50000,
                )

            # Store in session state for potential display (use different keys to avoid widget conflicts)
            st.session_state["battle_trajectory"] = traj
            st.session_state["battle_prices"] = prices
            st.session_state["battle_cfg_k1"] = battle_k1
            st.session_state["battle_cfg_k2"] = battle_k2
            st.session_state["battle_cfg_c"] = battle_c
            st.session_state["battle_cfg_start"] = (battle_start_p1, battle_start_p2)
            st.session_state["battle_cfg_p_e"] = p_e
            st.session_state["battle_cfg_p_c"] = p_c
            st.session_state["battle_cfg_profit_e"] = profit_e
            st.session_state["battle_cfg_profit_c"] = profit_c

            st.rerun()

        except Exception as e:
            st.error(f"❌ Error computing trajectory: {e}")
            st.exception(e)

    # Display results if trajectory has been computed
    if "battle_trajectory" in st.session_state:
        traj = st.session_state["battle_trajectory"]
        prices = st.session_state["battle_prices"]
        # Read stored config values (stored when trajectory was computed)
        battle_k1 = st.session_state.get("battle_cfg_k1", 7.0)
        battle_k2 = st.session_state.get("battle_cfg_k2", 0.5)
        battle_c = st.session_state.get("battle_cfg_c", 2.0)
        battle_start = st.session_state.get("battle_cfg_start", (7.0, 7.0))
        battle_start_p1, battle_start_p2 = battle_start
        p_e = st.session_state.get("battle_cfg_p_e", 0.0)
        p_c = st.session_state.get("battle_cfg_p_c", 0.0)
        profit_e = st.session_state.get("battle_cfg_profit_e", 0.0)
        profit_c = st.session_state.get("battle_cfg_profit_c", 0.0)

        path = traj["path"]
        loop = traj["loop"]
        loop_start = traj["loop_start"]

        st.markdown("---")
        st.markdown(
            rf"""
            - Equilibrium price: $p_e = {p_e:.2f}$
            - Collusion price: $p_c = {p_c:.2f}$
            - Equilibrium profit: $\pi_e = {profit_e:.2f}$
            - Collusion profit: $\pi_c = {profit_c:.2f}$
            """
        )

        st.subheader("🏆 Battle Results")

        if loop_start is not None and loop:
            # Calculate average profits in the cycle
            loop_profits_alice = []
            loop_profits_bob = []
            loop_prices_alice = []
            loop_prices_bob = []

            for rec in loop:
                p1 = rec["a1_price"]
                p2 = rec["a2_price"]
                loop_prices_alice.append(p1)
                loop_prices_bob.append(p2)
                pi1 = profit1(p1, p2, battle_c, battle_k1, battle_k2)
                pi2 = profit2(p1, p2, battle_c, battle_k1, battle_k2)
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
                sum(loop_prices_bob) / len(loop_prices_bob) if loop_prices_bob else 0.0
            )

            # Determine winner
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

            # Display winner with style
            if winner != "🤝 Tie":
                st.success(f"## 🏆 Winner: {winner}!")
                st.markdown(
                    f"""
                    **Average Profit: {winner_profit:.2f}** (vs {loser_profit:.2f})
                    """
                )
            else:
                st.info(f"## {winner}!")
                st.markdown(rf"**Average Profit: {winner_profit:.2f}** (both players)")

            # Calculate normalized profits
            denominator = profit_c - profit_e
            if abs(denominator) > 1e-10:  # Avoid division by zero
                normalized_profit_alice = (avg_profit_alice - profit_e) / denominator
                normalized_profit_bob = (avg_profit_bob - profit_e) / denominator
            else:
                normalized_profit_alice = None
                normalized_profit_bob = None

            # Display detailed results
            col_result1, col_result2 = st.columns(2)

            with col_result1:
                st.metric(
                    f"👩🏼‍💼 {player_1_name} (Player 1)",
                    f"{avg_profit_alice:.2f}",
                    delta=(
                        f"{avg_profit_alice - avg_profit_bob:.2f}"
                        if avg_profit_alice != avg_profit_bob
                        else None
                    ),
                    help=rf"Average price: $\bar{{p}}_1 = {avg_p1:.2f}$, average profit: $\bar{{\pi}}_1 = {avg_profit_alice:.2f}$",
                )
                if normalized_profit_alice is not None:
                    st.metric(
                        "Normalised average profit",
                        f"{normalized_profit_alice:.2f}",
                        help=r"Normalised average profit: $\Delta_1 = \dfrac{{\bar{{\pi}}_1 - \pi_e}}{{\pi_c - \pi_e}}$",
                    )
                else:
                    st.metric(
                        "Normalised average profit",
                        "N/A",
                        help="Cannot calculate normalized profit: denominator (collusion profit - equilibrium profit) is zero",
                    )

            with col_result2:
                st.metric(
                    f"🧑🏼‍💼 {player_2_name} (Player 2)",
                    f"{avg_profit_bob:.2f}",
                    delta=(
                        f"{avg_profit_bob - avg_profit_alice:.2f}"
                        if avg_profit_bob != avg_profit_alice
                        else None
                    ),
                    help=rf"Average price: $\bar{{p}}_2 = {avg_p2:.2f}$, average profit: $\bar{{\pi}}_2 = {avg_profit_bob:.2f}$",
                )
                if normalized_profit_bob is not None:
                    st.metric(
                        "Normalised average profit",
                        f"{normalized_profit_bob:.2f}",
                        help=r"Normalised average profit: $\Delta_2 = \dfrac{{\bar{{\pi}}_2 - \pi_e}}{{\pi_c - \pi_e}}$",
                    )
                else:
                    st.metric(
                        "Normalised average profit",
                        "N/A",
                        help="Cannot calculate normalized profit: denominator (collusion profit - equilibrium profit) is zero",
                    )

            # Display cycle information
            st.markdown("---")
            st.markdown("### 📊 Cycle Information")

            # Display price trajectory (same format as tab_2)
            st.markdown(r"**Trajectory Steps for $p_1, p_2$ :**")

            if path:
                # Add step 0 with starting prices at the beginning
                full_path = [
                    {
                        "step_num": 0,
                        "a1_price": battle_start_p1,
                        "a2_price": battle_start_p2,
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
                table_id = f"battle_trajectory_table_{id(path)}"
                html_table = f"""
                <div id="{table_id}_container" style="width: 100%; overflow-x: auto; overflow-y: hidden; border: 1px solid #ddd; border-radius: 5px;">
                    <table style="border-collapse: collapse; table-layout: fixed;">
                        <tbody>
                            <tr>
                                <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">👩🏼‍💼 {player_1_name}</td>
                                {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in alice_row])}
                            </tr>
                            <tr>
                                <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">🧑🏼‍💼 {player_2_name}</td>
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
                        "💡 Scroll horizontally to see all steps. The newest steps appear on the left."
                    )

            # Trajectory table for profit steps (same format as tab_2)
            st.markdown(r"**Trajectory Steps for $\pi_1, \pi_2$ :**")

            if path:
                # Calculate profits for each step in full_path
                alice_profit_row = []
                bob_profit_row = []
                step_row_profit = []

                for rec in full_path:
                    p1 = rec["a1_price"]
                    p2 = rec["a2_price"]
                    pi1 = profit1(p1, p2, battle_c, battle_k1, battle_k2)
                    pi2 = profit2(p1, p2, battle_c, battle_k1, battle_k2)
                    alice_profit_row.append(f"{pi1:.2f}")
                    bob_profit_row.append(f"{pi2:.2f}")
                    step_row_profit.append(f"Step {rec['step_num']}")

                # Create HTML table for profits with horizontal scroll, fixed column width
                table_id_profit = f"battle_trajectory_profit_table_{id(path)}"
                html_table_profit = f"""
                <div id="{table_id_profit}_container" style="width: 100%; overflow-x: auto; overflow-y: hidden; border: 1px solid #ddd; border-radius: 5px;">
                    <table style="border-collapse: collapse; table-layout: fixed;">
                        <tbody>
                            <tr>
                                <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">👩🏼‍💼 {player_1_name}</td>
                                {' '.join([f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; width: {fixed_col_width}; min-width: {fixed_col_width};">{val}</td>' for val in alice_profit_row])}
                            </tr>
                            <tr>
                                <td style="padding: 10px; border-right: 2px solid #ddd; position: sticky; left: 0; background-color: white; z-index: 9; font-weight: bold; min-width: 100px; width: 100px;">🧑🏼‍💼 {player_2_name}</td>
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
            if (
                normalized_profit_alice is not None
                and normalized_profit_bob is not None
            ):
                st.markdown(
                    rf"""
                    - **Cycle detected**: starts at step {loop_start}, length = {len(loop)} steps
                    - **Average price for {player_1_name} ($\bar{{p}}_1$)**: {avg_p1:.2f}
                    - **Average price for {player_2_name} ($\bar{{p}}_2$)**: {avg_p2:.2f}
                    - **Average profit for {player_1_name} ($\bar{{\pi}}_1$)**: {avg_profit_alice:.2f}
                    - **Average profit for {player_2_name} ($\bar{{\pi}}_2$)**: {avg_profit_bob:.2f}
                    - Equilibrium profit: $\pi_e = {profit_e:.2f}$
                    - Collusion profit: $\pi_c = {profit_c:.2f}$
                    - **Normalised average profit for {player_1_name} ($\Delta_1$)**= $\dfrac{{{avg_profit_alice:.2f}-{profit_e:.2f}}}{{{profit_c:.2f} - {profit_e:.2f}}}$ = {normalized_profit_alice:.2f}
                    - **Normalised average profit for {player_2_name} ($\Delta_2$)**= $\dfrac{{{avg_profit_bob:.2f}-{profit_e:.2f}}}{{{profit_c:.2f} - {profit_e:.2f}}}$ = {normalized_profit_bob:.2f}
                    """
                )
            else:
                st.markdown(
                    rf"""
                    - **Cycle detected**: starts at step {loop_start}, length = {len(loop)} steps
                    - **Average price for {player_1_name} ($p_1$)**: {avg_p1:.2f}
                    - **Average price for {player_2_name} ($p_2$)**: {avg_p2:.2f}
                    - **Average profit for {player_1_name} ($\pi_1$)**: {avg_profit_alice:.2f}
                    - **Average profit for {player_2_name} ($\pi_2$)**: {avg_profit_bob:.2f}
                    - **Normalised profit**: Cannot be calculated (denominator is zero)
                    """
                )
        else:
            st.warning(
                "⚠️ No cycle detected within max_steps. Consider adjusting starting prices or increasing max_steps."
            )
