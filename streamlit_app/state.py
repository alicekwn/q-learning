"""Session state management for Q-learning demo."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from qlearning import LineGrid, RectangularGrid, QLearningAgent

# Performance: Limit checkpoint history to prevent memory bloat
MAX_CHECKPOINTS = 50

__all__ = [
    "init_session_state",
    "get_start_state",
    "reset_episode",
    "record_q_history",
    "record_episode",
    "step_agent",
    "run_batch_training",
    "rewind_checkpoint",
    "forward_checkpoint",
    "jump_to_latest",
    "get_display_state",
    "is_in_playback_mode",
    "save_checkpoint",
    # 2D functions
    "init_session_state_2d",
    "get_start_state_2d",
    "reset_episode_2d",
    "record_q_history_2d",
    "step_agent_2d",
    "run_batch_training_2d",
]

ACTIONS_1D = ["L", "R"]  # Keep for backward compatibility with DataFrame columns


def init_session_state(config: dict) -> None:
    """Initialize or reset all session state for new environment (tab-scoped)."""
    tab_id = config.get("tab_id", "default")
    start_pos = config["start_pos"]
    end_pos = config["end_pos"]
    goal_pos = config["goal_pos"]
    reward_val = config["reward_val"]
    
    # Create states from start_pos to end_pos (inclusive)
    states = list(range(start_pos, end_pos + 1))
    
    # Create environment and agent
    env = LineGrid(states, goal_pos, reward_val)
    agent = QLearningAgent(env, config["alpha"], config["gamma"], config["epsilon"])
    
    # Store in session state with tab prefix
    st.session_state[f"{tab_id}_env"] = env
    st.session_state[f"{tab_id}_agent"] = agent
    
    # Q-table as DataFrame for display compatibility (index uses actual position values)
    st.session_state[f"{tab_id}_q_table"] = pd.DataFrame(0.0, index=states, columns=ACTIONS_1D)
    
    # Episode tracking
    start_s = get_start_state(
        config["start_mode"], config["fixed_start_pos"], start_pos, end_pos, goal_pos
    )
    st.session_state[f"{tab_id}_current_state"] = start_s
    st.session_state[f"{tab_id}_current_path"] = [start_s]
    st.session_state[f"{tab_id}_is_terminal"] = (start_s == goal_pos)
    st.session_state[f"{tab_id}_episode_start"] = start_s
    st.session_state[f"{tab_id}_ready_for_episode"] = True  # Ready to start training
    
    # Logging
    st.session_state[f"{tab_id}_history_log"] = []  # Episode-level log
    st.session_state[f"{tab_id}_step_log"] = []  # Detailed step log (persists across episodes)
    st.session_state[f"{tab_id}_q_history_plot"] = []
    st.session_state[f"{tab_id}_checkpoints"] = []  # User action checkpoints for rewind
    st.session_state[f"{tab_id}_total_episodes"] = 0
    st.session_state[f"{tab_id}_playback_index"] = -1  # -1 = live, 0+ = checkpoint index
    st.session_state[f"{tab_id}_steps_per_episode"] = []  # Steps taken in each completed episode
    
    record_q_history(config)
    
    # Save initial checkpoint (Q-matrix all zeros) so user can rewind to beginning
    save_checkpoint(config, "init", {"description": "Initial state"})


def get_start_state(mode: str, fixed_pos: int, start_pos: int, end_pos: int, goal_pos: int) -> int:
    """Get starting state based on mode (Fixed/Randomized)."""
    if mode == "Randomized":
        possible_starts = [i for i in range(start_pos, end_pos + 1) if i != goal_pos]
        if not possible_starts:
            return start_pos
        return int(np.random.choice(possible_starts))
    return fixed_pos


def reset_episode(config: dict) -> None:
    """Reset agent position for new episode (keep Q-table) - tab-scoped.
    
    Sets ready_for_episode = False to indicate episode training has begun.
    Saves initial checkpoint so user can rewind to episode start.
    """
    tab_id = config.get("tab_id", "default")
    start_s = get_start_state(
        config["start_mode"],
        config["fixed_start_pos"],
        config["start_pos"],
        config["end_pos"],
        config["goal_pos"]
    )
    st.session_state[f"{tab_id}_current_state"] = start_s
    st.session_state[f"{tab_id}_current_path"] = [start_s]
    st.session_state[f"{tab_id}_is_terminal"] = (start_s == config["goal_pos"])
    st.session_state[f"{tab_id}_episode_start"] = start_s  # Track episode start
    st.session_state[f"{tab_id}_ready_for_episode"] = False  # Now training this episode
    # Don't clear step_log - keep history across episodes
    
    # Save initial checkpoint for this episode (allows rewind to start)
    save_checkpoint(config, "episode_start", {"episode": st.session_state[f"{tab_id}_total_episodes"] + 1})


def record_q_history(config: dict) -> None:
    """Snapshot Q-table for plotting - tab-scoped."""
    tab_id = config.get("tab_id", "default")
    q_table = st.session_state[f"{tab_id}_q_table"]
    snapshot = {}
    for s in q_table.index:
        for a in ACTIONS_1D:
            snapshot[f"Q({s},{a})"] = q_table.at[s, a]
    snapshot['Episode'] = st.session_state[f"{tab_id}_total_episodes"]
    st.session_state[f"{tab_id}_q_history_plot"].append(snapshot)


def record_episode(config: dict, steps_taken: int, start_state: int, end_state: int) -> None:
    """Log completed episode data - tab-scoped."""
    tab_id = config.get("tab_id", "default")
    episode_num = st.session_state[f"{tab_id}_total_episodes"]
    
    episode_entry = {
        "Episode": episode_num,
        "Steps": steps_taken,
        "Start": start_state,
        "End": end_state,
        "Terminal": end_state == config["goal_pos"]
    }
    st.session_state[f"{tab_id}_history_log"].append(episode_entry)


def save_checkpoint(config: dict, action_type: str, metadata: dict = None) -> None:
    """Save a checkpoint after user action (single step or batch training)."""
    tab_id = config.get("tab_id", "default")
    
    checkpoint = {
        "type": action_type,  # "step" or "batch"
        "q_table": st.session_state[f"{tab_id}_q_table"].copy(),
        "q_history_plot": st.session_state[f"{tab_id}_q_history_plot"].copy(),
        "current_state": st.session_state[f"{tab_id}_current_state"],
        "current_path": st.session_state[f"{tab_id}_current_path"].copy(),
        "is_terminal": st.session_state[f"{tab_id}_is_terminal"],
        "ready_for_episode": st.session_state[f"{tab_id}_ready_for_episode"],
        "total_episodes": st.session_state[f"{tab_id}_total_episodes"],
        "steps_per_episode": st.session_state[f"{tab_id}_steps_per_episode"].copy(),
        "metadata": metadata or {}
    }
    
    checkpoints = st.session_state[f"{tab_id}_checkpoints"]
    checkpoints.append(checkpoint)
    
    # Limit checkpoint history to prevent memory bloat (keep first init + recent checkpoints)
    if len(checkpoints) > MAX_CHECKPOINTS:
        st.session_state[f"{tab_id}_checkpoints"] = [checkpoints[0]] + checkpoints[-(MAX_CHECKPOINTS - 1):]


def step_agent(config: dict) -> None:
    """Perform one Q-learning step with logging - tab-scoped."""
    tab_id = config.get("tab_id", "default")
    state = st.session_state[f"{tab_id}_current_state"]
    q_df = st.session_state[f"{tab_id}_q_table"]
    start_pos = config["start_pos"]
    end_pos = config["end_pos"]
    goal_pos = config["goal_pos"]
    alpha = config["alpha"]
    gamma = config["gamma"]
    epsilon = config["epsilon"]
    reward_val = config["reward_val"]
    
    # 1. Choose Action (Epsilon-Greedy)
    if np.random.rand() < epsilon:
        action = np.random.choice(ACTIONS_1D)
        decision_type = "Exploratory (Random)"
    else:
        current_q = q_df.loc[state]
        max_q = current_q.max()
        best_actions = current_q[current_q == max_q].index.tolist()
        action = np.random.choice(best_actions)
        decision_type = "Max Value (Greedy)"
    
    # 2. Environment interaction
    move = -1 if action == "L" else 1
    next_state = max(start_pos, min(end_pos, state + move))
    done = (next_state == goal_pos)
    r = reward_val if done else 0.0
    
    # 3. Bellman update
    old_val = q_df.at[state, action]
    max_next_q = 0.0 if done else q_df.loc[next_state].max()
    td_target = r + gamma * max_next_q
    new_val = old_val + alpha * (td_target - old_val)
    
    # Update tables
    st.session_state[f"{tab_id}_q_table"].at[state, action] = new_val
    st.session_state[f"{tab_id}_agent"].Q[(state, action)] = new_val
    
    # 4. Log step details (for detailed view if needed)
    eq_str = (
        f"Q({state}, {action}) = {old_val:.2f} + {alpha} * "
        f"[{r} + {gamma} * {max_next_q:.2f} - {old_val:.2f}] = **{new_val:.4f}**"
    )
    
    # Count steps within current episode
    episode_num = st.session_state[f"{tab_id}_total_episodes"] + 1
    episode_step_count = sum(1 for log in st.session_state[f"{tab_id}_step_log"] if log["Episode"] == episode_num) + 1
    
    step_entry = {
        "Episode": episode_num,  # Current episode (1-indexed)
        "Step": episode_step_count,  # Step within this episode
        "State": state,
        "Action": action,
        "Type": decision_type,
        "Equation": eq_str,
        "New Q": new_val
    }
    st.session_state[f"{tab_id}_step_log"].append(step_entry)
    
    # 5. Move agent
    st.session_state[f"{tab_id}_current_state"] = next_state
    st.session_state[f"{tab_id}_current_path"].append(next_state)
    st.session_state[f"{tab_id}_is_terminal"] = done
    st.session_state[f"{tab_id}_ready_for_episode"] = False  # Now mid-episode
    
    if done:
        # Log episode completion
        episode_start = st.session_state[f"{tab_id}_episode_start"]
        steps_taken = sum(1 for log in st.session_state[f"{tab_id}_step_log"] if log["Episode"] == episode_num)
        record_episode(config, steps_taken, episode_start, next_state)
        
        # Record steps for this completed episode
        st.session_state[f"{tab_id}_steps_per_episode"].append(steps_taken)
        
        st.session_state[f"{tab_id}_total_episodes"] += 1
        st.session_state[f"{tab_id}_ready_for_episode"] = True  # Ready for next episode
        record_q_history(config)
    
    # Save checkpoint for this user action
    save_checkpoint(config, "step", {"episode": episode_num, "step": episode_step_count})


def run_batch_training(episodes_to_run: int, config: dict) -> None:
    """Fast-forward training for N episodes with progress bar - tab-scoped."""
    tab_id = config.get("tab_id", "default")
    
    # Should only be called from ready state, but safety check
    if not st.session_state.get(f"{tab_id}_ready_for_episode", True):
        # Force to ready state if somehow called mid-episode
        st.session_state[f"{tab_id}_is_terminal"] = True
        st.session_state[f"{tab_id}_ready_for_episode"] = True
    
    progress_bar = st.progress(0)
    start_pos = config["start_pos"]
    end_pos = config["end_pos"]
    goal_pos = config["goal_pos"]
    alpha = config["alpha"]
    gamma = config["gamma"]
    epsilon = config["epsilon"]
    reward_val = config["reward_val"]
    
    for i in range(episodes_to_run):
        # Internal reset for batch training (doesn't change ready flag)
        if st.session_state[f"{tab_id}_is_terminal"] or st.session_state[f"{tab_id}_current_state"] == goal_pos:
            start_s = get_start_state(
                config["start_mode"], config["fixed_start_pos"], start_pos, end_pos, goal_pos
            )
            st.session_state[f"{tab_id}_current_state"] = start_s
            st.session_state[f"{tab_id}_current_path"] = [start_s]
            st.session_state[f"{tab_id}_is_terminal"] = False
        
        curr_s = st.session_state[f"{tab_id}_current_state"]
        episode_start = curr_s  # Track start of this batch episode
        steps = 0
        
        while curr_s != goal_pos and steps < 100:
            # Epsilon-greedy
            if np.random.rand() < epsilon:
                a = np.random.choice(ACTIONS_1D)
            else:
                qs = st.session_state[f"{tab_id}_q_table"].loc[curr_s]
                max_q = qs.max()
                a = np.random.choice(qs[qs == max_q].index.tolist())
            
            # Step
            move = -1 if a == "L" else 1
            next_s = max(start_pos, min(end_pos, curr_s + move))
            done = (next_s == goal_pos)
            r = reward_val if done else 0.0
            
            # Update
            old_v = st.session_state[f"{tab_id}_q_table"].at[curr_s, a]
            max_next = 0.0 if done else st.session_state[f"{tab_id}_q_table"].loc[next_s].max()
            new_v = old_v + alpha * (r + gamma * max_next - old_v)
            st.session_state[f"{tab_id}_q_table"].at[curr_s, a] = new_v
            st.session_state[f"{tab_id}_agent"].Q[(curr_s, a)] = new_v
            
            curr_s = next_s
            steps += 1
        
        # End episode
        st.session_state[f"{tab_id}_current_state"] = curr_s
        st.session_state[f"{tab_id}_is_terminal"] = True
        
        # Log episode data
        record_episode(config, steps, episode_start, curr_s)
        
        # Record steps for this completed episode
        st.session_state[f"{tab_id}_steps_per_episode"].append(steps)
        
        st.session_state[f"{tab_id}_total_episodes"] += 1
        progress_bar.progress((i + 1) / episodes_to_run)
    
    # Record Q-history only once after all batch episodes complete (performance optimization)
    record_q_history(config)
    
    # After batch, set to ready state (terminal, ready for next action)
    st.session_state[f"{tab_id}_is_terminal"] = True
    st.session_state[f"{tab_id}_ready_for_episode"] = True  # Ready for next action
    
    # Save single checkpoint for entire batch
    save_checkpoint(config, "batch", {"episodes": episodes_to_run})


def is_in_playback_mode(config: dict) -> bool:
    """Check if viewing historical state (not live)."""
    tab_id = config.get("tab_id", "default")
    return st.session_state[f"{tab_id}_playback_index"] >= 0


def get_display_state(config: dict) -> dict:
    """Get current display state (live or historical checkpoint)."""
    tab_id = config.get("tab_id", "default")
    playback_idx = st.session_state[f"{tab_id}_playback_index"]
    
    if playback_idx < 0:  # Live mode
        return {
            "q_table": st.session_state[f"{tab_id}_q_table"],
            "q_history_plot": st.session_state[f"{tab_id}_q_history_plot"],
            "current_state": st.session_state[f"{tab_id}_current_state"],
            "current_path": st.session_state[f"{tab_id}_current_path"],
            "step_log": st.session_state[f"{tab_id}_step_log"],
            "total_episodes": st.session_state[f"{tab_id}_total_episodes"],
            "ready_for_episode": st.session_state.get(f"{tab_id}_ready_for_episode", True),
            "is_terminal": st.session_state.get(f"{tab_id}_is_terminal", True),
            "steps_per_episode": st.session_state.get(f"{tab_id}_steps_per_episode", []),
            "is_live": True
        }
    
    # Historical mode - viewing a checkpoint
    checkpoints = st.session_state[f"{tab_id}_checkpoints"]
    if not checkpoints or playback_idx >= len(checkpoints):
        # Fallback to live if invalid index
        st.session_state[f"{tab_id}_playback_index"] = -1
        return get_display_state(config)
    
    checkpoint = checkpoints[playback_idx]
    
    return {
        "q_table": checkpoint["q_table"],
        "q_history_plot": checkpoint["q_history_plot"],
        "current_state": checkpoint["current_state"],
        "current_path": checkpoint["current_path"],
        "step_log": st.session_state[f"{tab_id}_step_log"],  # Always show all logs
        "total_episodes": checkpoint["total_episodes"],
        "ready_for_episode": checkpoint.get("ready_for_episode", True),
        "is_terminal": checkpoint.get("is_terminal", True),
        "steps_per_episode": checkpoint.get("steps_per_episode", []),
        "action_type": checkpoint["type"],
        "metadata": checkpoint["metadata"],
        "is_live": False
    }


def rewind_checkpoint(config: dict) -> None:
    """Rewind to previous user action checkpoint."""
    tab_id = config.get("tab_id", "default")
    checkpoints = st.session_state[f"{tab_id}_checkpoints"]
    
    if not checkpoints:
        return
    
    current_idx = st.session_state[f"{tab_id}_playback_index"]
    
    if current_idx < 0:  # Currently live
        # Jump to last checkpoint
        st.session_state[f"{tab_id}_playback_index"] = len(checkpoints) - 1
    elif current_idx > 0:
        # Go back one checkpoint
        st.session_state[f"{tab_id}_playback_index"] = current_idx - 1


def forward_checkpoint(config: dict) -> None:
    """Forward to next user action checkpoint."""
    tab_id = config.get("tab_id", "default")
    checkpoints = st.session_state[f"{tab_id}_checkpoints"]
    
    if not checkpoints:
        return
    
    current_idx = st.session_state[f"{tab_id}_playback_index"]
    
    if current_idx < 0:  # Currently live, do nothing
        return
    
    if current_idx < len(checkpoints) - 1:
        # Move forward one checkpoint
        st.session_state[f"{tab_id}_playback_index"] = current_idx + 1
    else:
        # At last checkpoint, jump to live
        st.session_state[f"{tab_id}_playback_index"] = -1

def jump_to_start(config: dict) -> None:
    """Jump to start of playback."""
    tab_id = config.get("tab_id", "default")
    st.session_state[f"{tab_id}_playback_index"] = 0

def jump_to_latest(config: dict) -> None:
    """Jump back to live (latest) state."""
    tab_id = config.get("tab_id", "default")
    st.session_state[f"{tab_id}_playback_index"] = -1


# ============================================================================
# 2D Grid Functions
# ============================================================================


def init_session_state_2d(config: dict) -> None:
    """Initialize or reset all session state for 2D grid environment (tab-scoped)."""
    tab_id = config.get("tab_id", "default")
    x_start = config["x_start"]
    x_end = config["x_end"]
    y_start = config["y_start"]
    y_end = config["y_end"]
    goal_pos = (config["goal_x"], config["goal_y"])
    reward_val = config["reward_val"]
    
    # Create environment and agent
    env = RectangularGrid(x_start, x_end, y_start, y_end, goal_pos, reward_val)
    agent = QLearningAgent(env, config["alpha"], config["gamma"], config["epsilon"])
    
    # Store in session state with tab prefix
    st.session_state[f"{tab_id}_env"] = env
    st.session_state[f"{tab_id}_agent"] = agent
    
    # Create all states as (x, y) tuples (Cartesian coordinates)
    all_states = [(x, y) for x in range(x_start, x_end + 1) for y in range(y_start, y_end + 1)]
    st.session_state[f"{tab_id}_all_states"] = all_states
    st.session_state[f"{tab_id}_goal_pos"] = goal_pos
    
    # Get actions from environment
    actions_2d = list(env.ACTIONS.keys())
    
    # Q-table as dict keyed by ((x, y), action) - sync with agent.Q
    q_dict = {}
    for state in all_states:
        for action in actions_2d:
            q_dict[(state, action)] = agent.Q[(state, action)]
    st.session_state[f"{tab_id}_q_dict"] = q_dict
    
    # Also create DataFrame representation for display
    q_table = pd.DataFrame(0.0, index=[str(s) for s in all_states], columns=actions_2d)
    st.session_state[f"{tab_id}_q_table"] = q_table
    
    # Episode tracking
    start_s = get_start_state_2d(
        config["start_mode"],
        (config["fixed_start_x"], config["fixed_start_y"]),
        x_start, x_end, y_start, y_end,
        goal_pos
    )
    st.session_state[f"{tab_id}_current_state"] = start_s
    st.session_state[f"{tab_id}_current_path"] = [start_s]
    st.session_state[f"{tab_id}_is_terminal"] = (start_s == goal_pos)
    st.session_state[f"{tab_id}_episode_start"] = start_s
    st.session_state[f"{tab_id}_ready_for_episode"] = True
    
    # Logging
    st.session_state[f"{tab_id}_history_log"] = []
    st.session_state[f"{tab_id}_step_log"] = []
    st.session_state[f"{tab_id}_q_history_plot"] = []
    st.session_state[f"{tab_id}_checkpoints"] = []
    st.session_state[f"{tab_id}_total_episodes"] = 0
    st.session_state[f"{tab_id}_playback_index"] = -1
    st.session_state[f"{tab_id}_steps_per_episode"] = []
    
    record_q_history_2d(config)
    save_checkpoint(config, "init", {"description": "Initial state"})


def get_start_state_2d(
    mode: str,
    fixed_pos: tuple[int, int],
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    goal_pos: tuple[int, int]
) -> tuple[int, int]:
    """Get starting state for 2D grid based on mode (Fixed/Randomized)."""
    if mode == "Randomized":
        possible_starts = [
            (x, y) for x in range(x_start, x_end + 1) for y in range(y_start, y_end + 1)
            if (x, y) != goal_pos
        ]
        if not possible_starts:
            return (x_start, y_start)
        idx = int(np.random.choice(len(possible_starts)))
        return possible_starts[idx]
    return fixed_pos


def reset_episode_2d(config: dict) -> None:
    """Reset agent position for new episode in 2D grid (keep Q-table)."""
    tab_id = config.get("tab_id", "default")
    goal_pos = (config["goal_x"], config["goal_y"])
    
    start_s = get_start_state_2d(
        config["start_mode"],
        (config["fixed_start_x"], config["fixed_start_y"]),
        config["x_start"], config["x_end"],
        config["y_start"], config["y_end"],
        goal_pos
    )
    st.session_state[f"{tab_id}_current_state"] = start_s
    st.session_state[f"{tab_id}_current_path"] = [start_s]
    st.session_state[f"{tab_id}_is_terminal"] = (start_s == goal_pos)
    st.session_state[f"{tab_id}_episode_start"] = start_s
    st.session_state[f"{tab_id}_ready_for_episode"] = False
    
    save_checkpoint(config, "episode_start", {"episode": st.session_state[f"{tab_id}_total_episodes"] + 1})


def record_q_history_2d(config: dict) -> None:
    """Snapshot Q-table for plotting - 2D version."""
    tab_id = config.get("tab_id", "default")
    q_dict = st.session_state[f"{tab_id}_q_dict"]
    all_states = st.session_state[f"{tab_id}_all_states"]
    env = st.session_state[f"{tab_id}_env"]
    actions_2d = list(env.ACTIONS.keys())
    
    snapshot = {}
    for s in all_states:
        for a in actions_2d:
            snapshot[f"Q{s},{a}"] = q_dict[(s, a)]
    snapshot['Episode'] = st.session_state[f"{tab_id}_total_episodes"]
    st.session_state[f"{tab_id}_q_history_plot"].append(snapshot)


def step_agent_2d(config: dict) -> None:
    """Perform one Q-learning step in 2D grid with logging (Cartesian coords)."""
    tab_id = config.get("tab_id", "default")
    state = st.session_state[f"{tab_id}_current_state"]
    env = st.session_state[f"{tab_id}_env"]
    agent = st.session_state[f"{tab_id}_agent"]
    q_dict = st.session_state[f"{tab_id}_q_dict"]
    q_table = st.session_state[f"{tab_id}_q_table"]
    
    alpha = config["alpha"]
    gamma = config["gamma"]
    
    # 1. Choose Action (Epsilon-Greedy) - use agent method
    action = agent._greedy_action_constant(state)
    if action is None:
        return  # Terminal state
    
    # Determine decision type for logging
    import random
    if random.random() < agent.epsilon:
        decision_type = "Exploratory (Random)"
    else:
        decision_type = "Max Value (Greedy)"
    
    # 2. Environment interaction - use env.step()
    next_state, reward, done = env.step(state, action)
    
    # 3. Get old Q-value before update
    old_val = q_dict[(state, action)]
    
    # 4. Bellman update - use agent method
    agent._q_update(state, action, reward, next_state)
    
    # 5. Sync Q-values to session state dict and DataFrame
    new_val = agent.Q[(state, action)]
    q_dict[(state, action)] = new_val
    q_table.at[str(state), action] = new_val
    
    # Also update next_state Q-values in dict (for max_next_q calculation)
    actions_2d = list(env.ACTIONS.keys())
    for a in actions_2d:
        q_dict[(next_state, a)] = agent.Q[(next_state, a)]
        q_table.at[str(next_state), a] = agent.Q[(next_state, a)]
    
    # Calculate max_next_q for logging
    if done:
        max_next_q = 0.0
    else:
        max_next_q = max(agent.Q[(next_state, a)] for a in actions_2d)
    
    # 4. Log step details
    eq_str = (
        f"Q({state}, {action}) = {old_val:.2f} + {alpha} * "
        f"[{reward} + {gamma} * {max_next_q:.2f} - {old_val:.2f}] = **{new_val:.4f}**"
    )
    
    episode_num = st.session_state[f"{tab_id}_total_episodes"] + 1
    episode_step_count = sum(1 for log in st.session_state[f"{tab_id}_step_log"] if log["Episode"] == episode_num) + 1
    
    step_entry = {
        "Episode": episode_num,
        "Step": episode_step_count,
        "State": state,
        "Action": action,
        "Type": decision_type,
        "Equation": eq_str,
        "New Q": new_val
    }
    st.session_state[f"{tab_id}_step_log"].append(step_entry)
    
    # 5. Move agent
    st.session_state[f"{tab_id}_current_state"] = next_state
    st.session_state[f"{tab_id}_current_path"].append(next_state)
    st.session_state[f"{tab_id}_is_terminal"] = done
    st.session_state[f"{tab_id}_ready_for_episode"] = False
    
    if done:
        steps_taken = sum(1 for log in st.session_state[f"{tab_id}_step_log"] if log["Episode"] == episode_num)
        
        # Record steps for completed episode
        st.session_state[f"{tab_id}_steps_per_episode"].append(steps_taken)
        
        st.session_state[f"{tab_id}_total_episodes"] += 1
        st.session_state[f"{tab_id}_ready_for_episode"] = True
        record_q_history_2d(config)
    
    save_checkpoint(config, "step", {"episode": episode_num, "step": episode_step_count})


def run_batch_training_2d(episodes_to_run: int, config: dict) -> None:
    """Fast-forward training for N episodes in 2D grid (Cartesian coords)."""
    tab_id = config.get("tab_id", "default")
    
    if not st.session_state.get(f"{tab_id}_ready_for_episode", True):
        st.session_state[f"{tab_id}_is_terminal"] = True
        st.session_state[f"{tab_id}_ready_for_episode"] = True
    
    progress_bar = st.progress(0)
    
    env = st.session_state[f"{tab_id}_env"]
    agent = st.session_state[f"{tab_id}_agent"]
    goal_pos = env.terminal_state
    q_dict = st.session_state[f"{tab_id}_q_dict"]
    q_table = st.session_state[f"{tab_id}_q_table"]
    actions_2d = list(env.ACTIONS.keys())
    
    for i in range(episodes_to_run):
        # Reset for new episode
        if st.session_state[f"{tab_id}_is_terminal"] or st.session_state[f"{tab_id}_current_state"] == goal_pos:
            start_s = get_start_state_2d(
                config["start_mode"],
                (config["fixed_start_x"], config["fixed_start_y"]),
                config["x_start"], config["x_end"],
                config["y_start"], config["y_end"],
                goal_pos
            )
            st.session_state[f"{tab_id}_current_state"] = start_s
            st.session_state[f"{tab_id}_current_path"] = [start_s]
            st.session_state[f"{tab_id}_is_terminal"] = False
        
        curr_s = st.session_state[f"{tab_id}_current_state"]
        steps = 0
        
        while not env.is_terminal(curr_s) and steps < 100:
            # Use agent's epsilon-greedy action selection
            action = agent._greedy_action_constant(curr_s)
            if action is None:
                break
            
            # Use environment's step method
            next_s, reward, _ = env.step(curr_s, action)
            
            # Use agent's Q-update method
            agent._q_update(curr_s, action, reward, next_s)
            
            # Sync Q-values to session state
            q_dict[(curr_s, action)] = agent.Q[(curr_s, action)]
            q_table.at[str(curr_s), action] = agent.Q[(curr_s, action)]
            # Also sync next_state Q-values
            for a in actions_2d:
                q_dict[(next_s, a)] = agent.Q[(next_s, a)]
                q_table.at[str(next_s), a] = agent.Q[(next_s, a)]
            
            curr_s = next_s
            steps += 1
        
        # End episode
        st.session_state[f"{tab_id}_current_state"] = curr_s
        st.session_state[f"{tab_id}_is_terminal"] = True
        
        # Record steps for completed episode
        st.session_state[f"{tab_id}_steps_per_episode"].append(steps)
        
        st.session_state[f"{tab_id}_total_episodes"] += 1
        progress_bar.progress((i + 1) / episodes_to_run)
    
    # Record Q-history only once after all batch episodes complete (performance optimization)
    record_q_history_2d(config)
    
    st.session_state[f"{tab_id}_is_terminal"] = True
    st.session_state[f"{tab_id}_ready_for_episode"] = True
    
    save_checkpoint(config, "batch", {"episodes": episodes_to_run})

