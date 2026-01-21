import numpy as np
import pandas as pd
from tqdm.auto import tqdm

price_start = 24
price_end = 84
interval = 2  # fix the interval between prices
PRICES = np.arange(price_start, price_end + interval / 2, interval).tolist()
n_actions = len(PRICES)
n_states = n_actions * n_actions  # Number of combinations of (p1,p2)
price_to_idx = {p: i for i, p in enumerate(PRICES)}  # Map price to column index 0-9

# ---------- Helper functions ----------


def state_index(p1, p2):
    """Flattens a 2D (p1, p2) into a single row index."""
    return price_to_idx[p1] * n_actions + price_to_idx[p2]


def index_to_state(s: int):
    """Inverse map from index -> (p1,p2)."""
    i = s // n_actions
    j = s % n_actions
    return PRICES[i], PRICES[j]


def demand1(p1, p2):  # q1
    return max(0.0, 62 - p1 + 0.5 * p2)


def demand2(p1, p2):  # q2
    return max(0.0, 62 - p2 + 0.5 * p1)


def profit1(p1, p2, c=4.0):
    return (p1 - c) * demand1(p1, p2)


def profit2(p1, p2, c=4.0):
    return (p2 - c) * demand2(p1, p2)


def epsilon_at(step, beta):
    return float(np.exp(-beta * step))  # ε_t = exp(-beta * t)


def argmax_tie(x):
    # When there are more than one max, choose a random one
    m = np.max(x)
    idxs = np.flatnonzero(np.isclose(x, m))
    return np.random.choice(idxs)  # choose a random one
    # return int(idxs[-1])


def greedy_map(Q):
    # returns an array of length n_states with best-action indices
    return np.array([argmax_tie(Q[s_]) for s_ in range(n_states)], dtype=int)


# ---------- Game Parameters ----------

nash_price = 44
collusion_price = 64
c = 4.0
nash_profit = profit1(nash_price, nash_price, c=c)
# nash_profit = (nash_price - c) * demand1(nash_price, nash_price)
collusion_profit = profit1(collusion_price, collusion_price, c=c)


# ---------- Q-learning training ----------
def train_episode(
    alpha=0.25,
    delta=0.95,
    beta=2 * 1e-5,
    c=4.0,
    stable_required=100_000,  # need greedy policy unchanged for this long
    check_every=1_000,  # compare policies every K periods
    max_periods=2_000_000,  # hard cap so we won't loop forever
    seed=43,
):
    rng = np.random.default_rng(seed)  # NumPy random generator with a fixed seed
    # Initialise the two Q-tables (one per firm), all value are 0
    Q1 = np.zeros((n_states, n_actions))
    Q2 = np.zeros((n_states, n_actions))

    # Pick a single random starting state
    p1, p2 = rng.choice(PRICES), rng.choice(PRICES)
    s = state_index(p1, p2)

    # For stability checks: Record each firm’s argmax action per row.
    prev_pi1 = greedy_map(Q1)
    prev_pi2 = greedy_map(Q2)
    stable = 0  # stability check counter

    for t in range(1, max_periods + 1):  # loop over periods/steps
        eps = epsilon_at(t, beta=beta)

        # ε-greedy choices with deterministic exploitation
        if rng.random() < eps:
            a1 = rng.integers(0, n_actions)
        else:
            a1 = argmax_tie(Q1[s])

        if rng.random() < eps:
            a2 = rng.integers(0, n_actions)
        else:
            a2 = argmax_tie(Q2[s])

        # Compute next state when an action is chosen
        p1_next, p2_next = (
            PRICES[a1],
            PRICES[a2],
        )  # convert action indices to actual prices
        s_next = state_index(p1_next, p2_next)  # compute the next state

        pi1 = profit1(p1_next, p2_next, c=c)
        pi2 = profit2(p1_next, p2_next, c=c)

        # Q-learning updates
        Q1[s, a1] = (1 - alpha) * Q1[s, a1] + alpha * (pi1 + delta * np.max(Q1[s_next]))
        Q2[s, a2] = (1 - alpha) * Q2[s, a2] + alpha * (pi2 + delta * np.max(Q2[s_next]))

        s = s_next

        # check policy-stability criterion over all states
        # for every check_every periods, build the greedy argmax per state policies for both firms
        if t % check_every == 0:
            current_pi1 = greedy_map(Q1)
            current_pi2 = greedy_map(Q2)
            if np.array_equal(current_pi1, prev_pi1) and np.array_equal(
                current_pi2, prev_pi2
            ):
                stable += check_every
            else:
                stable = 0
                prev_pi1, prev_pi2 = current_pi1, current_pi2

            if stable >= stable_required:
                return (
                    Q1,
                    Q2,
                    {
                        "converged": True,
                        "periods_run": t,
                        "stable_periods": stable,
                        "epsilon_final": eps,
                    },
                )

    return (
        Q1,
        Q2,
        {
            "converged": False,
            "periods_run": max_periods,
            "stable_periods": stable,
            "epsilon_final": epsilon_at(max_periods, beta=beta),
        },
    )

    # ---- cycle detection utilities ----


def greedy_successor(Q1, Q2, s):
    """
    From state index s, take each firm's greedy action and return:
    (a1, a2, p1_next, p2_next, s_next)
    """
    a1 = argmax_tie(Q1[s])
    a2 = argmax_tie(Q2[s])
    p1_next = PRICES[a1]
    p2_next = PRICES[a2]
    s_next = state_index(p1_next, p2_next)
    return a1, a2, p1_next, p2_next, s_next


def follow_greedy_until_loop(
    Q1, Q2, start_p1, start_p2, max_steps=100000, verbose=True
):
    """
    Follow the greedy map (both players best-respond) from s(start_p1,start_p2),
    printing the chosen actions ("max movements") and stopping on the first repeat.
    Returns a dict with the full path and the detected cycle.
    """
    # start state
    s = state_index(start_p1, start_p2)

    # remember when we first saw each state (for loop detection)
    first_seen_at = {}  # state_idx -> time step when first visited
    path = []  # list of dicts describing transitions

    for t in range(max_steps):
        if s in first_seen_at:
            loop_start = first_seen_at[s]
            loop_path = path[loop_start:]  # the cycle
            if verbose:
                print("\n=== CYCLE DETECTED ===")
                print(f"Cycle starts at step {loop_start}, length = {len(loop_path)}")
                # print the cycle compactly
                for k, rec in enumerate(loop_path, start=0):
                    print(
                        f"[L{k}] s={rec['state_str']}  "
                        f"→ (a1:{rec['a1_price']}, a2:{rec['a2_price']})  "
                        f"→ s'={rec['next_state_str']}"
                    )
            return {
                "path": path,  # all transitions until loop
                "loop_start": loop_start,  # index in path where loop begins
                "loop": loop_path,  # the loop transitions
            }

        # mark first visit to this state
        first_seen_at[s] = t

        # decode current state's (p1,p2) for printing
        cur_p1, cur_p2 = index_to_state(s)

        # take greedy actions and move to successor
        a1, a2, p1_next, p2_next, s_next = greedy_successor(Q1, Q2, s)

        # record transition
        path.append(
            {
                "t": t,
                "state": s,
                "state_str": f"s({cur_p1},{cur_p2})",
                "a1_idx": int(a1),
                "a2_idx": int(a2),
                "a1_price": p1_next,  # "max movement read from Q1"
                "a2_price": p2_next,  # "max movement read from Q2"
                "next_state": s_next,
                "next_state_str": f"s({p1_next},{p2_next})",
            }
        )

        if verbose:
            print(
                f"t={t:>6} | s= s({cur_p1},{cur_p2}) "
                f"| max(Q1)->{p1_next:>4}  max(Q2)->{p2_next:>4} "
                f"| s' = s({p1_next},{p2_next})"
            )

        # advance
        s = s_next

    # if we hit max_steps without finding a loop
    if verbose:
        print("\nNo loop detected within max_steps; consider increasing max_steps.")
    return {"path": path, "loop_start": None, "loop": []}


def get_cycle_stats(Q1, Q2, start_p1, start_p2, c=4.0, max_steps=100000):
    """
    Run follow_greedy_until_loop from (start_p1, start_p2),
    and compute statistics for the detected cycle:

    - cycle_length
    - avg_price_1, avg_price_2, avg_price_both
    - avg_profit_1, avg_profit_2, avg_total_profit
    """
    traj = follow_greedy_until_loop(
        Q1, Q2, start_p1=start_p1, start_p2=start_p2, max_steps=max_steps, verbose=False
    )

    loop = traj["loop"]
    if not loop:
        # No loop detected within max_steps
        return None

    # Prices along the cycle
    p1_prices = np.array([rec["a1_price"] for rec in loop], dtype=float)
    p2_prices = np.array([rec["a2_price"] for rec in loop], dtype=float)

    # Profits along the cycle
    p1_profits = np.array(
        [profit1(p1, p2, c=c) for p1, p2 in zip(p1_prices, p2_prices)], dtype=float
    )
    p2_profits = np.array(
        [profit2(p1, p2, c=c) for p1, p2 in zip(p1_prices, p2_prices)], dtype=float
    )
    total_profits = p1_profits + p2_profits

    cycle_length = len(loop)

    return {
        "cycle_length": cycle_length,
        "avg_price_1": float(p1_prices.mean()),
        "avg_price_2": float(p2_prices.mean()),
        "avg_price_both": float((p1_prices.mean() + p2_prices.mean()) / 2.0),
        "avg_delta_1": (float(p1_profits.mean()) - nash_profit)
        / (collusion_profit - nash_profit),
        "avg_delta_2": (float(p2_profits.mean()) - nash_profit)
        / (collusion_profit - nash_profit),
        "avg_delta_both": (
            float(p1_profits.mean()) + float(p2_profits.mean()) - 2 * nash_profit
        )
        / (2 * (collusion_profit - nash_profit)),
        "avg_profit_1": float(p1_profits.mean()),
        "avg_profit_2": float(p2_profits.mean()),
        "avg_total_profit": float(total_profits.mean()),
        "loop_path": [(float(p1), float(p2)) for p1, p2 in zip(p1_prices, p2_prices)],
        "loop_profits": [
            (float(p1), float(p2)) for p1, p2 in zip(p1_profits, p2_profits)
        ],
    }


def run_many_episodes(
    n_runs=3,
    alpha=0.125,
    delta=0.95,
    beta=2e-5,
    c=4.0,
    stable_required=100_000,
    check_every=1_000,
    max_periods=2_000_000,
    start_states=((nash_price, nash_price), (collusion_price, collusion_price)),
    seed_base=43,
):
    """
    Train Q-learning n_runs times. For each run:
      - train_episode with a different seed
      - for each (start_p1, start_p2) in start_states:
            follow greedy policy until loop
            compute cycle length and price/profit averages

    Returns a pandas DataFrame with one row per (run, start_state).
    """
    global all_stats  # Make it global so it persists if interrupted
    all_stats = []

    # tqdm wraps the range and shows a progress bar
    try:
        for k in tqdm(range(n_runs), desc="Training episodes"):
            seed = seed_base + k

            Q1, Q2, info = train_episode(
                alpha=alpha,
                delta=delta,
                beta=beta,
                c=c,
                stable_required=stable_required,
                check_every=check_every,
                max_periods=max_periods,
                seed=seed,
            )

            for sp1, sp2 in start_states:
                stats = get_cycle_stats(Q1, Q2, start_p1=sp1, start_p2=sp2, c=c)
                if stats is None:
                    # no loop detected from this start within max_steps
                    continue

                stats["run"] = k
                stats["start_p1"] = sp1
                stats["start_p2"] = sp2
                stats["converged"] = info["converged"]
                stats["periods_run"] = info["periods_run"]
                all_stats.append(stats)

    except KeyboardInterrupt:
        print(f"\nInterrupted! Saving {len(all_stats)} completed runs...")
        df = pd.DataFrame(all_stats)
        df.to_csv(f"interrupted_results_{len(all_stats)}_runs.csv", index=False)
        print(f"Partial results saved to interrupted_results_{len(all_stats)}_runs.csv")
        return df

    df = pd.DataFrame(all_stats)
    return df


# ====== Run many episodes and collect cycle stats ======
beta = 1e-6
results = run_many_episodes(
    n_runs=400,
    alpha=0.125,
    delta=0.95,
    beta=beta,
    c=4.0,
    stable_required=100_000,
    check_every=1_000,
    max_periods=2_000_000,
    start_states=[(nash_price, nash_price), (collusion_price, collusion_price)],
    seed_base=43,
)

print("\n=== Summary DataFrame ===")
print(results.head())
results.to_csv(f"results_{n_actions}_{beta}.csv", index=False)
