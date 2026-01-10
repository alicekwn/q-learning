"""Session state management for Economics Pricing Q-learning demo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# Performance: Limit checkpoint history to prevent memory bloat
MAX_CHECKPOINTS = 50

__all__ = [
    "init_session_state_econ",
    "pick_random_starting_prices_econ",
    "step_agent_econ",
    "run_batch_training_econ",
    "run_until_convergence_econ",
    "get_display_state_econ",
    "is_in_playback_mode_econ",
    "save_checkpoint_econ",
    "rewind_checkpoint_econ",
    "forward_checkpoint_econ",
    "jump_to_latest_econ",
    "jump_to_start_econ",
]


# ---------- Economics Helper Functions ----------


def demand1(p1: float, p2: float, k1: float, k2: float) -> float:
    """Demand function for player 1: q1 = k1 - p1 + k2 * p2."""
    return max(0.0, k1 - p1 + k2 * p2)


def demand2(p1: float, p2: float, k1: float, k2: float) -> float:
    """Demand function for player 2: q2 = k1 - p2 + k2 * p1."""
    return max(0.0, k1 - p2 + k2 * p1)


def profit1(p1: float, p2: float, c: float, k1: float, k2: float) -> float:
    """Profit function for player 1: π1 = (p1 - c) * q1."""
    return (p1 - c) * demand1(p1, p2, k1, k2)


def profit2(p1: float, p2: float, c: float, k1: float, k2: float) -> float:
    """Profit function for player 2: π2 = (p2 - c) * q2."""
    return (p2 - c) * demand2(p1, p2, k1, k2)


def calculate_prices(
    k1: float, k2: float, c: float, m: int
) -> tuple[list[float], float, float, float, float]:
    """Calculate equilibrium price, collusion price, and action space.

    Returns:
        (PRICES list, p_e (equilibrium), p_c (collusion), profit_e (equilibrium profit), profit_c (collusion profit))
    """
    # Calculate equilibrium price: p_e = (k1 + c) / (2 - k2)
    p_e = (k1 + c) / (2 - k2)

    # Calculate equilibrium profit: π_e = (p_e - c) * demand1(p_e, p_e, k1, k2)
    profit_e = (p_e - c) * demand1(p_e, p_e, k1, k2)

    # Calculate collusion price: p_c = (2*k1 + 2*c*(1-k2)) / (4*(1-k2))
    p_c = (2 * k1 + 2 * c * (1 - k2)) / (4 * (1 - k2))

    # Calculate collusion profit: π_c = (p_c - c) * demand1(p_c, p_c, k1, k2)
    profit_c = (p_c - c) * demand1(p_c, p_c, k1, k2)

    # Feasible price interval: [2p_e - p_c, 2p_c - p_e]
    price_start = 2 * p_e - p_c
    price_end = 2 * p_c - p_e

    # Generate m equally spaced points and round to 1 decimal place to avoid floating point precision issues
    PRICES = np.round(np.linspace(price_start, price_end, m), 3).tolist()

    return PRICES, p_e, p_c, profit_e, profit_c


def state_index(p1: float, p2: float, PRICES: list[float]) -> int:
    """Flattens a 2D (p1, p2) into a single row index."""
    price_to_idx = {p: i for i, p in enumerate(PRICES)}
    N_ACTIONS = len(PRICES)
    return price_to_idx[p1] * N_ACTIONS + price_to_idx[p2]


def index_to_state(s: int, PRICES: list[float]) -> tuple[float, float]:
    """Inverse map from index -> (p1, p2)."""
    N_ACTIONS = len(PRICES)
    i = s // N_ACTIONS
    j = s % N_ACTIONS
    return PRICES[i], PRICES[j]


def epsilon_at(step: int, beta: float) -> float:
    """Exponential decay exploration rate: ε_t = exp(-beta * t)."""
    return float(np.exp(-beta * step))


def argmax_tie(x: np.ndarray, rng: np.random.Generator) -> int:
    """When there are multiple max values, choose a random one."""
    m = np.max(x)
    idxs = np.flatnonzero(np.isclose(x, m))
    return int(rng.choice(idxs))


def greedy_map(Q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Returns an array of length N_STATES with best-action indices."""
    N_STATES = Q.shape[0]
    return np.array([argmax_tie(Q[s_], rng) for s_ in range(N_STATES)], dtype=int)


# ---------- State Management Functions ----------


def init_session_state_econ(config: dict) -> None:
    """Initialize or reset all session state for economics pricing (tab-scoped)."""
    tab_id = config.get("tab_id", "default")
    k1 = config["k1"]
    k2 = config["k2"]
    c = config["c"]
    m = config["m"]
    alpha = config["alpha"]
    delta = config["delta"]
    beta = config["beta"]
    seed = config.get("seed", 43)

    # Calculate prices and key prices
    PRICES, p_e, p_c, profit_e, profit_c = calculate_prices(k1, k2, c, m)
    N_ACTIONS = len(PRICES)
    N_STATES = N_ACTIONS * N_ACTIONS

    # Store configuration
    st.session_state[f"{tab_id}_PRICES"] = PRICES
    st.session_state[f"{tab_id}_N_ACTIONS"] = N_ACTIONS
    st.session_state[f"{tab_id}_N_STATES"] = N_STATES
    # Store config values under cfg_ keys to avoid widget key collisions
    st.session_state[f"{tab_id}_cfg_k1"] = k1
    st.session_state[f"{tab_id}_cfg_k2"] = k2
    st.session_state[f"{tab_id}_cfg_c"] = c
    st.session_state[f"{tab_id}_cfg_m"] = m
    st.session_state[f"{tab_id}_cfg_alpha"] = alpha
    st.session_state[f"{tab_id}_cfg_delta"] = delta
    st.session_state[f"{tab_id}_cfg_beta"] = beta
    st.session_state[f"{tab_id}_cfg_p_e"] = p_e
    st.session_state[f"{tab_id}_cfg_p_c"] = p_c
    st.session_state[f"{tab_id}_cfg_profit_e"] = profit_e
    st.session_state[f"{tab_id}_cfg_profit_c"] = profit_c
    st.session_state[f"{tab_id}_cfg_seed"] = seed

    # Initialize random number generator
    rng = np.random.default_rng(seed)
    st.session_state[f"{tab_id}_rng"] = rng

    # Initialize Q-tables (both start at zero)
    Q1 = np.zeros((N_STATES, N_ACTIONS))
    Q2 = np.zeros((N_STATES, N_ACTIONS))
    st.session_state[f"{tab_id}_Q1"] = Q1
    st.session_state[f"{tab_id}_Q2"] = Q2

    # Create Q-tables as DataFrames for display
    states = [f"s({p1:.1f},{p2:.1f})" for p1 in PRICES for p2 in PRICES]
    actions = [f"price={p:.1f}" for p in PRICES]
    st.session_state[f"{tab_id}_q_table_1"] = pd.DataFrame(
        Q1, index=states, columns=actions
    )
    st.session_state[f"{tab_id}_q_table_2"] = pd.DataFrame(
        Q2, index=states, columns=actions
    )

    # Pick starting state based on mode
    start_mode = config.get("start_mode", "Randomised")
    if start_mode == "Fixed":
        p1_start = config.get("fixed_start_p1", PRICES[0])
        p2_start = config.get("fixed_start_p2", PRICES[0])
        # Validate prices are in PRICES (round to nearest if close)
        p1_start = min(PRICES, key=lambda x: abs(x - p1_start))
        p2_start = min(PRICES, key=lambda x: abs(x - p2_start))
        starting_prices_picked = True
    else:
        # Randomised: don't set starting prices yet - user must click button
        p1_start = None
        p2_start = None
        starting_prices_picked = False

    # Current state tracking
    if starting_prices_picked:
        s_start = state_index(p1_start, p2_start, PRICES)
        st.session_state[f"{tab_id}_current_p1"] = p1_start
        st.session_state[f"{tab_id}_current_p2"] = p2_start
        st.session_state[f"{tab_id}_current_state"] = s_start
        # Price history: list of (step_num, p1, p2) tuples
        st.session_state[f"{tab_id}_price_history"] = [(0, p1_start, p2_start)]
    else:
        # No starting prices yet - set to None/empty
        st.session_state[f"{tab_id}_current_p1"] = None
        st.session_state[f"{tab_id}_current_p2"] = None
        st.session_state[f"{tab_id}_current_state"] = None
        st.session_state[f"{tab_id}_price_history"] = []

    # Track whether starting prices have been picked
    st.session_state[f"{tab_id}_starting_prices_picked"] = starting_prices_picked

    # Skipped steps during fast-forward: list of (start_step, end_step) tuples
    st.session_state[f"{tab_id}_skipped_steps"] = []

    # Step counter
    st.session_state[f"{tab_id}_step_count"] = 0

    # Convergence tracking
    st.session_state[f"{tab_id}_prev_pi1"] = greedy_map(Q1, rng)
    st.session_state[f"{tab_id}_prev_pi2"] = greedy_map(Q2, rng)
    st.session_state[f"{tab_id}_stable_count"] = 0
    st.session_state[f"{tab_id}_cfg_check_every"] = config.get("check_every", 1000)
    st.session_state[f"{tab_id}_cfg_stable_required"] = config.get(
        "stable_required", 100000
    )
    st.session_state[f"{tab_id}_cfg_max_periods"] = config.get("max_periods", 2000000)

    # Convergence info
    st.session_state[f"{tab_id}_convergence_info"] = None

    # Logging
    st.session_state[f"{tab_id}_step_log"] = []  # Detailed step log
    st.session_state[f"{tab_id}_checkpoints"] = []  # User action checkpoints
    st.session_state[f"{tab_id}_playback_index"] = (
        -1
    )  # -1 = live, 0+ = checkpoint index

    # Ready state
    st.session_state[f"{tab_id}_ready_for_training"] = True

    # Save initial checkpoint
    save_checkpoint_econ(config, "init", {"description": "Initial state"})


def pick_random_starting_prices_econ(config: dict) -> None:
    """Pick random starting prices for randomized mode (tab-scoped)."""
    tab_id = config.get("tab_id", "default")

    # Get PRICES from session state
    PRICES = st.session_state.get(f"{tab_id}_PRICES")
    if not PRICES:
        return

    # Pick random starting prices using np.random.choice directly (not seeded rng) for true randomization
    p1_start = float(np.random.choice(PRICES))
    p2_start = float(np.random.choice(PRICES))

    # Set starting state
    s_start = state_index(p1_start, p2_start, PRICES)
    st.session_state[f"{tab_id}_current_p1"] = p1_start
    st.session_state[f"{tab_id}_current_p2"] = p2_start
    st.session_state[f"{tab_id}_current_state"] = s_start

    # Add to price history
    st.session_state[f"{tab_id}_price_history"] = [(0, p1_start, p2_start)]

    # Mark as picked
    st.session_state[f"{tab_id}_starting_prices_picked"] = True

    # Save checkpoint
    save_checkpoint_econ(
        config, "pick_starting_prices", {"p1": p1_start, "p2": p2_start}
    )


def step_agent_econ(config: dict) -> None:
    """Perform one Q-learning step for both players with logging (tab-scoped)."""
    tab_id = config.get("tab_id", "default")

    # Check if starting prices have been picked
    starting_prices_picked = st.session_state.get(
        f"{tab_id}_starting_prices_picked", True
    )
    if not starting_prices_picked:
        # Cannot step if starting prices haven't been picked
        return

    # Get configuration
    PRICES = st.session_state[f"{tab_id}_PRICES"]
    N_ACTIONS = st.session_state[f"{tab_id}_N_ACTIONS"]
    k1 = st.session_state[f"{tab_id}_cfg_k1"]
    k2 = st.session_state[f"{tab_id}_cfg_k2"]
    c = st.session_state[f"{tab_id}_cfg_c"]
    alpha = st.session_state[f"{tab_id}_cfg_alpha"]
    delta = st.session_state[f"{tab_id}_cfg_delta"]
    beta = st.session_state[f"{tab_id}_cfg_beta"]
    rng = st.session_state[f"{tab_id}_rng"]

    # Get current state
    s = st.session_state[f"{tab_id}_current_state"]
    p1, p2 = index_to_state(s, PRICES)

    # Get Q-tables
    Q1 = st.session_state[f"{tab_id}_Q1"]
    Q2 = st.session_state[f"{tab_id}_Q2"]

    # Increment step count
    step_count = st.session_state[f"{tab_id}_step_count"] + 1
    st.session_state[f"{tab_id}_step_count"] = step_count

    # Calculate epsilon for this step
    eps = epsilon_at(step_count, beta)

    # ε-greedy action selection for both players
    if rng.random() < eps:
        a1 = rng.integers(0, N_ACTIONS)
        decision_type_1 = "Exploratory (Random)"
    else:
        a1 = argmax_tie(Q1[s], rng)
        decision_type_1 = "Max Value (Greedy)"

    if rng.random() < eps:
        a2 = rng.integers(0, N_ACTIONS)
        decision_type_2 = "Exploratory (Random)"
    else:
        a2 = argmax_tie(Q2[s], rng)
        decision_type_2 = "Max Value (Greedy)"

    # Compute next state
    p1_next = PRICES[a1]
    p2_next = PRICES[a2]
    s_next = state_index(p1_next, p2_next, PRICES)

    # Calculate demands and profits (rewards)
    q1 = demand1(p1_next, p2_next, k1, k2)
    q2 = demand2(p1_next, p2_next, k1, k2)
    pi1 = profit1(p1_next, p2_next, c, k1, k2)
    pi2 = profit2(p1_next, p2_next, c, k1, k2)

    # Q-learning updates
    old_val_1 = Q1[s, a1]
    max_next_q1 = np.max(Q1[s_next])
    td_target_1 = pi1 + delta * max_next_q1
    new_val_1 = (1 - alpha) * old_val_1 + alpha * td_target_1

    old_val_2 = Q2[s, a2]
    max_next_q2 = np.max(Q2[s_next])
    td_target_2 = pi2 + delta * max_next_q2
    new_val_2 = (1 - alpha) * old_val_2 + alpha * td_target_2

    # Update Q-tables
    Q1[s, a1] = new_val_1
    Q2[s, a2] = new_val_2
    st.session_state[f"{tab_id}_Q1"] = Q1
    st.session_state[f"{tab_id}_Q2"] = Q2

    # Update DataFrame representations
    states = [f"s({p1:.1f},{p2:.1f})" for p1 in PRICES for p2 in PRICES]
    actions = [f"price={p:.1f}" for p in PRICES]
    st.session_state[f"{tab_id}_q_table_1"] = pd.DataFrame(
        Q1, index=states, columns=actions
    )
    st.session_state[f"{tab_id}_q_table_2"] = pd.DataFrame(
        Q2, index=states, columns=actions
    )

    # Find best actions for next state (for display)
    max_next_a1_idx = argmax_tie(Q1[s_next], rng)
    max_next_a2_idx = argmax_tie(Q2[s_next], rng)
    max_next_p1 = PRICES[max_next_a1_idx]
    max_next_p2 = PRICES[max_next_a2_idx]

    # Log step details for Q1
    eq_str_1 = (
        rf"Q_1(s({p1:.1f},{p2:.1f}), a_1={p1_next:.1f} = {old_val_1:.4f} + {alpha} * "
        f"[{pi1:.4f} + {delta} * {max_next_q1:.4f} - {old_val_1:.4f}] = **{new_val_1:.4f}**"
    )

    step_entry_1 = {
        "Player": "Alice (Q1)",
        "Step": step_count,
        "State (s)": f"s({p1:.1f},{p2:.1f})",
        "Action (a1)": f"{p1_next:.1f}",
        "Action (a2)": f"{p2_next:.1f}",
        "Next state": f"s({p1_next:.1f},{p2_next:.1f})",
        "Next action (a1)": f"{max_next_p1:.1f}",
        "Next action (a2)": f"{max_next_p2:.1f}",
        "Max next Q1": max_next_q1,
        "Type": decision_type_1,
        "Equation": eq_str_1,
        "New Q": new_val_1,
        "Reward (π1)": pi1,
        "p1": p1_next,
        "p2": p2_next,
        "c": c,
        "q1": q1,
    }

    # Log step details for Q2
    eq_str_2 = (
        rf"Q_2(s({p1:.1f},{p2:.1f}), a_2={p2_next:.1f}) = {old_val_2:.4f} + {alpha} * "
        f"[{pi2:.4f} + {delta} * {max_next_q2:.4f} - {old_val_2:.4f}] = **{new_val_2:.4f}**"
    )

    step_entry_2 = {
        "Player": "Bob (Q2)",
        "Step": step_count,
        "State (s)": f"s({p1:.1f},{p2:.1f})",
        "Action (a1)": f"{p1_next:.1f}",
        "Action (a2)": f"{p2_next:.1f}",
        "Next state": f"s({p1_next:.1f},{p2_next:.1f})",
        "Next action (a1)": f"{max_next_p1:.1f}",
        "Next action (a2)": f"{max_next_p2:.1f}",
        "Max next Q2": max_next_q2,
        "Type": decision_type_2,
        "Equation": eq_str_2,
        "New Q": new_val_2,
        "Reward (π2)": pi2,
        "p1": p1_next,
        "p2": p2_next,
        "c": c,
        "q2": q2,
    }

    st.session_state[f"{tab_id}_step_log"].append(step_entry_1)
    st.session_state[f"{tab_id}_step_log"].append(step_entry_2)

    # Update current state
    st.session_state[f"{tab_id}_current_p1"] = p1_next
    st.session_state[f"{tab_id}_current_p2"] = p2_next
    st.session_state[f"{tab_id}_current_state"] = s_next

    # Update price history (only for step-by-step, not fast-forward)
    st.session_state[f"{tab_id}_price_history"].append((step_count, p1_next, p2_next))

    # Check convergence (every check_every steps)
    check_every = st.session_state[f"{tab_id}_cfg_check_every"]
    if step_count % check_every == 0:
        current_pi1 = greedy_map(Q1, rng)
        current_pi2 = greedy_map(Q2, rng)
        prev_pi1 = st.session_state[f"{tab_id}_prev_pi1"]
        prev_pi2 = st.session_state[f"{tab_id}_prev_pi2"]

        if np.array_equal(current_pi1, prev_pi1) and np.array_equal(
            current_pi2, prev_pi2
        ):
            stable_count = st.session_state[f"{tab_id}_stable_count"] + check_every
            st.session_state[f"{tab_id}_stable_count"] = stable_count

            stable_required = st.session_state[f"{tab_id}_cfg_stable_required"]
            if stable_count >= stable_required:
                # Converged!
                st.session_state[f"{tab_id}_convergence_info"] = {
                    "converged": True,
                    "periods_run": step_count,
                    "stable_periods": stable_count,
                    "epsilon_final": eps,
                }
                st.session_state[f"{tab_id}_ready_for_training"] = True
        else:
            st.session_state[f"{tab_id}_stable_count"] = 0
            st.session_state[f"{tab_id}_prev_pi1"] = current_pi1
            st.session_state[f"{tab_id}_prev_pi2"] = current_pi2

    # Save checkpoint
    save_checkpoint_econ(config, "step", {"step": step_count})


def run_batch_training_econ(steps_to_run: int, config: dict) -> None:
    """Fast-forward training for N steps (tab-scoped)."""
    tab_id = config.get("tab_id", "default")

    # Check if starting prices have been picked
    starting_prices_picked = st.session_state.get(
        f"{tab_id}_starting_prices_picked", True
    )
    if not starting_prices_picked:
        # Cannot run batch training if starting prices haven't been picked
        return

    # Get configuration
    PRICES = st.session_state[f"{tab_id}_PRICES"]
    N_ACTIONS = st.session_state[f"{tab_id}_N_ACTIONS"]
    k1 = st.session_state[f"{tab_id}_cfg_k1"]
    k2 = st.session_state[f"{tab_id}_cfg_k2"]
    c = st.session_state[f"{tab_id}_cfg_c"]
    alpha = st.session_state[f"{tab_id}_cfg_alpha"]
    delta = st.session_state[f"{tab_id}_cfg_delta"]
    beta = st.session_state[f"{tab_id}_cfg_beta"]
    rng = st.session_state[f"{tab_id}_rng"]
    check_every = st.session_state[f"{tab_id}_cfg_check_every"]
    stable_required = st.session_state[f"{tab_id}_cfg_stable_required"]
    max_periods = st.session_state[f"{tab_id}_cfg_max_periods"]

    # Get current state
    Q1 = st.session_state[f"{tab_id}_Q1"]
    Q2 = st.session_state[f"{tab_id}_Q2"]
    s = st.session_state[f"{tab_id}_current_state"]
    step_count = st.session_state[f"{tab_id}_step_count"]

    # Track starting step for skipped range
    start_step = step_count + 1

    # Run steps
    for _ in range(steps_to_run):
        step_count += 1

        # Check max periods
        if step_count > max_periods:
            break

        # Get current prices
        p1, p2 = index_to_state(s, PRICES)

        # Calculate epsilon
        eps = epsilon_at(step_count, beta)

        # ε-greedy action selection
        if rng.random() < eps:
            a1 = rng.integers(0, N_ACTIONS)
        else:
            a1 = argmax_tie(Q1[s], rng)

        if rng.random() < eps:
            a2 = rng.integers(0, N_ACTIONS)
        else:
            a2 = argmax_tie(Q2[s], rng)

        # Compute next state
        p1_next = PRICES[a1]
        p2_next = PRICES[a2]
        s_next = state_index(p1_next, p2_next, PRICES)

        # Calculate profits
        pi1 = profit1(p1_next, p2_next, c, k1, k2)
        pi2 = profit2(p1_next, p2_next, c, k1, k2)

        # Q-learning updates
        Q1[s, a1] = (1 - alpha) * Q1[s, a1] + alpha * (pi1 + delta * np.max(Q1[s_next]))
        Q2[s, a2] = (1 - alpha) * Q2[s, a2] + alpha * (pi2 + delta * np.max(Q2[s_next]))

        s = s_next

        # Check convergence
        if step_count % check_every == 0:
            current_pi1 = greedy_map(Q1, rng)
            current_pi2 = greedy_map(Q2, rng)
            prev_pi1 = st.session_state[f"{tab_id}_prev_pi1"]
            prev_pi2 = st.session_state[f"{tab_id}_prev_pi2"]

            if np.array_equal(current_pi1, prev_pi1) and np.array_equal(
                current_pi2, prev_pi2
            ):
                stable_count = st.session_state[f"{tab_id}_stable_count"] + check_every
                st.session_state[f"{tab_id}_stable_count"] = stable_count

                if stable_count >= stable_required:
                    # Converged!
                    st.session_state[f"{tab_id}_convergence_info"] = {
                        "converged": True,
                        "periods_run": step_count,
                        "stable_periods": stable_count,
                        "epsilon_final": eps,
                    }
                    break
            else:
                st.session_state[f"{tab_id}_stable_count"] = 0
                st.session_state[f"{tab_id}_prev_pi1"] = current_pi1
                st.session_state[f"{tab_id}_prev_pi2"] = current_pi2

    # Update state
    st.session_state[f"{tab_id}_Q1"] = Q1
    st.session_state[f"{tab_id}_Q2"] = Q2
    st.session_state[f"{tab_id}_current_state"] = s
    p1_final, p2_final = index_to_state(s, PRICES)
    st.session_state[f"{tab_id}_current_p1"] = p1_final
    st.session_state[f"{tab_id}_current_p2"] = p2_final
    st.session_state[f"{tab_id}_step_count"] = step_count

    # Update DataFrame representations
    states = [f"s({p1:.1f},{p2:.1f})" for p1 in PRICES for p2 in PRICES]
    actions = [f"price={p:.1f}" for p in PRICES]
    st.session_state[f"{tab_id}_q_table_1"] = pd.DataFrame(
        Q1, index=states, columns=actions
    )
    st.session_state[f"{tab_id}_q_table_2"] = pd.DataFrame(
        Q2, index=states, columns=actions
    )

    # Record skipped steps
    end_step = step_count
    if end_step > start_step:
        st.session_state[f"{tab_id}_skipped_steps"].append((start_step, end_step))

    # Save checkpoint
    save_checkpoint_econ(
        config, "batch", {"steps": steps_to_run, "steps_run": end_step - start_step + 1}
    )


def run_until_convergence_econ(config: dict) -> None:
    """Fast-forward until convergence (stable_required periods) (tab-scoped)."""
    tab_id = config.get("tab_id", "default")

    # Check if starting prices have been picked
    starting_prices_picked = st.session_state.get(
        f"{tab_id}_starting_prices_picked", True
    )
    if not starting_prices_picked:
        # Cannot run until convergence if starting prices haven't been picked
        return

    # Get configuration
    PRICES = st.session_state[f"{tab_id}_PRICES"]
    N_ACTIONS = st.session_state[f"{tab_id}_N_ACTIONS"]
    k1 = st.session_state[f"{tab_id}_cfg_k1"]
    k2 = st.session_state[f"{tab_id}_cfg_k2"]
    c = st.session_state[f"{tab_id}_cfg_c"]
    alpha = st.session_state[f"{tab_id}_cfg_alpha"]
    delta = st.session_state[f"{tab_id}_cfg_delta"]
    beta = st.session_state[f"{tab_id}_cfg_beta"]
    rng = st.session_state[f"{tab_id}_rng"]
    check_every = st.session_state[f"{tab_id}_cfg_check_every"]
    stable_required = st.session_state[f"{tab_id}_cfg_stable_required"]
    max_periods = st.session_state[f"{tab_id}_cfg_max_periods"]

    # Get current state
    Q1 = st.session_state[f"{tab_id}_Q1"]
    Q2 = st.session_state[f"{tab_id}_Q2"]
    s = st.session_state[f"{tab_id}_current_state"]
    step_count = st.session_state[f"{tab_id}_step_count"]

    # Track starting step for skipped range
    start_step = step_count + 1

    # Run until convergence
    while step_count < max_periods:
        step_count += 1

        # Get current prices
        p1, p2 = index_to_state(s, PRICES)

        # Calculate epsilon
        eps = epsilon_at(step_count, beta)

        # ε-greedy action selection
        if rng.random() < eps:
            a1 = rng.integers(0, N_ACTIONS)
        else:
            a1 = argmax_tie(Q1[s], rng)

        if rng.random() < eps:
            a2 = rng.integers(0, N_ACTIONS)
        else:
            a2 = argmax_tie(Q2[s], rng)

        # Compute next state
        p1_next = PRICES[a1]
        p2_next = PRICES[a2]
        s_next = state_index(p1_next, p2_next, PRICES)

        # Calculate profits
        pi1 = profit1(p1_next, p2_next, c, k1, k2)
        pi2 = profit2(p1_next, p2_next, c, k1, k2)

        # Q-learning updates
        Q1[s, a1] = (1 - alpha) * Q1[s, a1] + alpha * (pi1 + delta * np.max(Q1[s_next]))
        Q2[s, a2] = (1 - alpha) * Q2[s, a2] + alpha * (pi2 + delta * np.max(Q2[s_next]))

        s = s_next

        # Check convergence
        if step_count % check_every == 0:
            current_pi1 = greedy_map(Q1, rng)
            current_pi2 = greedy_map(Q2, rng)
            prev_pi1 = st.session_state[f"{tab_id}_prev_pi1"]
            prev_pi2 = st.session_state[f"{tab_id}_prev_pi2"]

            if np.array_equal(current_pi1, prev_pi1) and np.array_equal(
                current_pi2, prev_pi2
            ):
                stable_count = st.session_state[f"{tab_id}_stable_count"] + check_every
                st.session_state[f"{tab_id}_stable_count"] = stable_count

                if stable_count >= stable_required:
                    # Converged!
                    st.session_state[f"{tab_id}_convergence_info"] = {
                        "converged": True,
                        "periods_run": step_count,
                        "stable_periods": stable_count,
                        "epsilon_final": eps,
                    }
                    break
            else:
                st.session_state[f"{tab_id}_stable_count"] = 0
                st.session_state[f"{tab_id}_prev_pi1"] = current_pi1
                st.session_state[f"{tab_id}_prev_pi2"] = current_pi2

    # If we hit max_periods without converging
    if step_count >= max_periods:
        st.session_state[f"{tab_id}_convergence_info"] = {
            "converged": False,
            "periods_run": max_periods,
            "stable_periods": st.session_state[f"{tab_id}_stable_count"],
            "epsilon_final": epsilon_at(max_periods, beta),
        }

    # Update state
    st.session_state[f"{tab_id}_Q1"] = Q1
    st.session_state[f"{tab_id}_Q2"] = Q2
    st.session_state[f"{tab_id}_current_state"] = s
    p1_final, p2_final = index_to_state(s, PRICES)
    st.session_state[f"{tab_id}_current_p1"] = p1_final
    st.session_state[f"{tab_id}_current_p2"] = p2_final
    st.session_state[f"{tab_id}_step_count"] = step_count

    # Update DataFrame representations
    states = [f"s({p1:.1f},{p2:.1f})" for p1 in PRICES for p2 in PRICES]
    actions = [f"price={p:.1f}" for p in PRICES]
    st.session_state[f"{tab_id}_q_table_1"] = pd.DataFrame(
        Q1, index=states, columns=actions
    )
    st.session_state[f"{tab_id}_q_table_2"] = pd.DataFrame(
        Q2, index=states, columns=actions
    )

    # Record skipped steps
    end_step = step_count
    if end_step > start_step:
        st.session_state[f"{tab_id}_skipped_steps"].append((start_step, end_step))

    # Mark as ready (converged or maxed out)
    st.session_state[f"{tab_id}_ready_for_training"] = True

    # Save checkpoint
    save_checkpoint_econ(
        config,
        "convergence",
        {"convergence_info": st.session_state[f"{tab_id}_convergence_info"]},
    )


def save_checkpoint_econ(config: dict, action_type: str, metadata: dict = None) -> None:
    """Save a checkpoint after user action (single step or batch training)."""
    tab_id = config.get("tab_id", "default")

    checkpoint = {
        "type": action_type,
        "Q1": st.session_state[f"{tab_id}_Q1"].copy(),
        "Q2": st.session_state[f"{tab_id}_Q2"].copy(),
        "q_table_1": st.session_state[f"{tab_id}_q_table_1"].copy(),
        "q_table_2": st.session_state[f"{tab_id}_q_table_2"].copy(),
        "current_state": st.session_state[f"{tab_id}_current_state"],
        "current_p1": st.session_state[f"{tab_id}_current_p1"],
        "current_p2": st.session_state[f"{tab_id}_current_p2"],
        "price_history": st.session_state[f"{tab_id}_price_history"].copy(),
        "skipped_steps": st.session_state[f"{tab_id}_skipped_steps"].copy(),
        "step_count": st.session_state[f"{tab_id}_step_count"],
        "step_log_count": len(st.session_state[f"{tab_id}_step_log"]),
        "convergence_info": st.session_state.get(f"{tab_id}_convergence_info"),
        "ready_for_training": st.session_state.get(
            f"{tab_id}_ready_for_training", True
        ),
        "starting_prices_picked": st.session_state.get(
            f"{tab_id}_starting_prices_picked", True
        ),
        "metadata": metadata or {},
    }

    checkpoints = st.session_state[f"{tab_id}_checkpoints"]
    checkpoints.append(checkpoint)

    # Limit checkpoint history
    if len(checkpoints) > MAX_CHECKPOINTS:
        st.session_state[f"{tab_id}_checkpoints"] = [checkpoints[0]] + checkpoints[
            -(MAX_CHECKPOINTS - 1) :
        ]


def get_display_state_econ(config: dict) -> dict:
    """Get current display state (live or playback) for economics pricing."""
    tab_id = config.get("tab_id", "default")
    playback_idx = st.session_state.get(f"{tab_id}_playback_index", -1)

    if playback_idx < 0:  # Live mode
        price_history = st.session_state[f"{tab_id}_price_history"]
        step_count = st.session_state[f"{tab_id}_step_count"]

        # Filter price_history to only include steps up to step_count
        # This ensures the latest step in price_history matches step_count
        filtered_price_history = [
            item for item in price_history if item[0] <= step_count
        ]

        return {
            "q_table_1": st.session_state[f"{tab_id}_q_table_1"],
            "q_table_2": st.session_state[f"{tab_id}_q_table_2"],
            "current_p1": st.session_state[f"{tab_id}_current_p1"],
            "current_p2": st.session_state[f"{tab_id}_current_p2"],
            "current_state": st.session_state[f"{tab_id}_current_state"],
            "price_history": filtered_price_history,
            "skipped_steps": st.session_state[f"{tab_id}_skipped_steps"],
            "step_log": st.session_state[f"{tab_id}_step_log"],
            "step_count": step_count,
            "convergence_info": st.session_state.get(f"{tab_id}_convergence_info"),
            "ready_for_training": st.session_state.get(
                f"{tab_id}_ready_for_training", True
            ),
            "starting_prices_picked": st.session_state.get(
                f"{tab_id}_starting_prices_picked", True
            ),
            "is_live": True,
        }

    # Historical mode
    checkpoints = st.session_state[f"{tab_id}_checkpoints"]
    if not checkpoints or playback_idx >= len(checkpoints):
        st.session_state[f"{tab_id}_playback_index"] = -1
        return get_display_state_econ(config)

    checkpoint = checkpoints[playback_idx]
    step_log_count = checkpoint.get(
        "step_log_count", len(st.session_state[f"{tab_id}_step_log"])
    )
    filtered_step_log = st.session_state[f"{tab_id}_step_log"][:step_log_count]

    # Filter price_history to only include steps up to step_count for this checkpoint
    # This ensures the latest step in price_history matches step_count
    step_count = checkpoint["step_count"]
    price_history = checkpoint["price_history"]
    filtered_price_history = [item for item in price_history if item[0] <= step_count]

    return {
        "q_table_1": checkpoint["q_table_1"],
        "q_table_2": checkpoint["q_table_2"],
        "current_p1": checkpoint["current_p1"],
        "current_p2": checkpoint["current_p2"],
        "current_state": checkpoint["current_state"],
        "price_history": filtered_price_history,
        "skipped_steps": checkpoint["skipped_steps"],
        "step_log": filtered_step_log,
        "step_count": step_count,
        "convergence_info": checkpoint.get("convergence_info"),
        "ready_for_training": checkpoint.get("ready_for_training", True),
        "starting_prices_picked": checkpoint.get("starting_prices_picked", True),
        "action_type": checkpoint["type"],
        "metadata": checkpoint["metadata"],
        "is_live": False,
    }


def is_in_playback_mode_econ(config: dict) -> bool:
    """Check if currently in playback mode."""
    tab_id = config.get("tab_id", "default")
    return st.session_state.get(f"{tab_id}_playback_index", -1) >= 0


def rewind_checkpoint_econ(config: dict) -> None:
    """Rewind to previous user action checkpoint."""
    tab_id = config.get("tab_id", "default")
    checkpoints = st.session_state[f"{tab_id}_checkpoints"]

    if not checkpoints:
        return

    current_idx = st.session_state[f"{tab_id}_playback_index"]

    if current_idx < 0:  # Currently live
        # Store the latest checkpoint index before starting playback
        latest_idx = len(checkpoints) - 1
        st.session_state[f"{tab_id}_latest_checkpoint_index"] = latest_idx
        # Go to the checkpoint before the last one (which is the current live state)
        target_idx = max(0, len(checkpoints) - 2)
        st.session_state[f"{tab_id}_playback_index"] = target_idx
    elif current_idx > 0:
        st.session_state[f"{tab_id}_playback_index"] = current_idx - 1

    # Restore state from checkpoint
    _restore_checkpoint_econ(
        config, checkpoints[st.session_state[f"{tab_id}_playback_index"]]
    )


def forward_checkpoint_econ(config: dict) -> None:
    """Forward to next user action checkpoint."""
    tab_id = config.get("tab_id", "default")
    checkpoints = st.session_state[f"{tab_id}_checkpoints"]

    if not checkpoints:
        return

    current_idx = st.session_state[f"{tab_id}_playback_index"]

    if current_idx < 0:  # Currently live, do nothing
        return

    # Get the latest checkpoint index (stored when playback started, or use last checkpoint)
    latest_idx = st.session_state.get(
        f"{tab_id}_latest_checkpoint_index", len(checkpoints) - 1
    )
    latest_idx = min(latest_idx, len(checkpoints) - 1)

    # If we're at or beyond the latest checkpoint, go to live mode
    if current_idx >= latest_idx:
        st.session_state[f"{tab_id}_playback_index"] = -1
        # Live state is already in session state, no need to restore
    else:
        # Move forward one checkpoint
        st.session_state[f"{tab_id}_playback_index"] = current_idx + 1
        _restore_checkpoint_econ(
            config, checkpoints[st.session_state[f"{tab_id}_playback_index"]]
        )


def jump_to_start_econ(config: dict) -> None:
    """Jump to start of playback (step_count = 0, after starting prices picked)."""
    tab_id = config.get("tab_id", "default")
    checkpoints = st.session_state[f"{tab_id}_checkpoints"]

    if not checkpoints:
        return

    # Find the checkpoint with step_count = 0 and starting_prices_picked = True
    # This should be the checkpoint after "pick_starting_prices" (Randomised mode)
    # or the init checkpoint (Fixed mode where starting prices are already picked)
    # Default to checkpoint 0 if not found
    target_idx = 0
    for i, checkpoint in enumerate(checkpoints):
        step_count = checkpoint.get("step_count", 0)
        starting_prices_picked = checkpoint.get("starting_prices_picked", True)
        # Look for checkpoint where starting prices have been picked and step_count is 0
        if step_count == 0 and starting_prices_picked:
            target_idx = i
            break

    st.session_state[f"{tab_id}_playback_index"] = target_idx
    _restore_checkpoint_econ(config, checkpoints[target_idx])


def jump_to_latest_econ(config: dict) -> None:
    """Jump back to the latest checkpoint (the last action taken).

    If called from training controls (taking a new action), exits playback mode.
    Otherwise, goes to the latest checkpoint.
    """
    tab_id = config.get("tab_id", "default")
    checkpoints = st.session_state.get(f"{tab_id}_checkpoints", [])

    if not checkpoints:
        st.session_state[f"{tab_id}_playback_index"] = -1
        return

    # Get the latest checkpoint index (stored when playback started, or use last checkpoint)
    latest_idx = st.session_state.get(
        f"{tab_id}_latest_checkpoint_index", len(checkpoints) - 1
    )
    latest_idx = min(latest_idx, len(checkpoints) - 1)

    # Check if we're currently at the latest checkpoint
    current_idx = st.session_state.get(f"{tab_id}_playback_index", -1)

    # If already at the latest checkpoint or beyond, or in live mode, exit to live mode
    # This handles the case where user is at latest checkpoint and wants to take new actions
    if current_idx >= latest_idx or current_idx < 0:
        st.session_state[f"{tab_id}_playback_index"] = -1
        return

    # Go to the latest checkpoint and restore it
    st.session_state[f"{tab_id}_playback_index"] = latest_idx
    _restore_checkpoint_econ(config, checkpoints[latest_idx])


def _restore_checkpoint_econ(config: dict, checkpoint: dict) -> None:
    """Internal helper to restore state from a checkpoint."""
    tab_id = config.get("tab_id", "default")
    st.session_state[f"{tab_id}_Q1"] = checkpoint["Q1"]
    st.session_state[f"{tab_id}_Q2"] = checkpoint["Q2"]
    st.session_state[f"{tab_id}_q_table_1"] = checkpoint["q_table_1"]
    st.session_state[f"{tab_id}_q_table_2"] = checkpoint["q_table_2"]
    st.session_state[f"{tab_id}_current_state"] = checkpoint["current_state"]
    st.session_state[f"{tab_id}_current_p1"] = checkpoint["current_p1"]
    st.session_state[f"{tab_id}_current_p2"] = checkpoint["current_p2"]
    st.session_state[f"{tab_id}_price_history"] = checkpoint["price_history"]
    st.session_state[f"{tab_id}_skipped_steps"] = checkpoint["skipped_steps"]
    st.session_state[f"{tab_id}_step_count"] = checkpoint["step_count"]
    if "convergence_info" in checkpoint:
        st.session_state[f"{tab_id}_convergence_info"] = checkpoint["convergence_info"]
    st.session_state[f"{tab_id}_ready_for_training"] = checkpoint.get(
        "ready_for_training", True
    )
    st.session_state[f"{tab_id}_starting_prices_picked"] = checkpoint.get(
        "starting_prices_picked", True
    )
