"""Session state management for Q-learning demo."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from qlearning import LineWorld, QLearningAgent

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
    "save_checkpoint"
]

ACTIONS = ["L", "R"]


def init_session_state(config: dict) -> None:
    """Initialize or reset all session state for new environment (tab-scoped)."""
    tab_id = config.get("tab_id", "default")
    grid_size = config["grid_size"]
    goal_pos = config["goal_pos"]
    reward_val = config["reward_val"]
    
    # Create environment and agent
    env = LineWorld(list(range(grid_size)), goal_pos, reward_val)
    agent = QLearningAgent(env, config["alpha"], config["gamma"], config["epsilon"])
    
    # Store in session state with tab prefix
    st.session_state[f"{tab_id}_env"] = env
    st.session_state[f"{tab_id}_agent"] = agent
    
    # Q-table as DataFrame for display compatibility
    st.session_state[f"{tab_id}_q_table"] = pd.DataFrame(0.0, index=range(grid_size), columns=ACTIONS)
    
    # Episode tracking
    start_s = get_start_state(
        config["start_mode"], config["fixed_start_pos"], grid_size, goal_pos
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
    
    record_q_history(config)
    
    # Save initial checkpoint (Q-matrix all zeros) so user can rewind to beginning
    save_checkpoint(config, "init", {"description": "Initial state"})


def get_start_state(mode: str, fixed_pos: int, grid_size: int, goal_pos: int) -> int:
    """Get starting state based on mode (Fixed/Randomized)."""
    if mode == "Randomized":
        possible_starts = [i for i in range(grid_size) if i != goal_pos]
        if not possible_starts:
            return 0
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
        config["grid_size"],
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
        for a in ACTIONS:
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
        "metadata": metadata or {}
    }
    
    st.session_state[f"{tab_id}_checkpoints"].append(checkpoint)


def step_agent(config: dict) -> None:
    """Perform one Q-learning step with logging - tab-scoped."""
    tab_id = config.get("tab_id", "default")
    state = st.session_state[f"{tab_id}_current_state"]
    q_df = st.session_state[f"{tab_id}_q_table"]
    grid_size = config["grid_size"]
    goal_pos = config["goal_pos"]
    alpha = config["alpha"]
    gamma = config["gamma"]
    epsilon = config["epsilon"]
    reward_val = config["reward_val"]
    
    # 1. Choose Action (Epsilon-Greedy)
    if np.random.rand() < epsilon:
        action = np.random.choice(ACTIONS)
        decision_type = "Exploratory (Random)"
    else:
        current_q = q_df.loc[state]
        max_q = current_q.max()
        best_actions = current_q[current_q == max_q].index.tolist()
        action = np.random.choice(best_actions)
        decision_type = "Max Value (Greedy)"
    
    # 2. Environment interaction
    move = -1 if action == "L" else 1
    next_state = max(0, min(grid_size - 1, state + move))
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
    grid_size = config["grid_size"]
    goal_pos = config["goal_pos"]
    alpha = config["alpha"]
    gamma = config["gamma"]
    epsilon = config["epsilon"]
    reward_val = config["reward_val"]
    
    for i in range(episodes_to_run):
        # Internal reset for batch training (doesn't change ready flag)
        if st.session_state[f"{tab_id}_is_terminal"] or st.session_state[f"{tab_id}_current_state"] == goal_pos:
            start_s = get_start_state(
                config["start_mode"], config["fixed_start_pos"], grid_size, goal_pos
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
                a = np.random.choice(ACTIONS)
            else:
                qs = st.session_state[f"{tab_id}_q_table"].loc[curr_s]
                max_q = qs.max()
                a = np.random.choice(qs[qs == max_q].index.tolist())
            
            # Step
            move = -1 if a == "L" else 1
            next_s = max(0, min(grid_size - 1, curr_s + move))
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
        
        st.session_state[f"{tab_id}_total_episodes"] += 1
        record_q_history(config)
        progress_bar.progress((i + 1) / episodes_to_run)
    
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


def jump_to_latest(config: dict) -> None:
    """Jump back to live (latest) state."""
    tab_id = config.get("tab_id", "default")
    st.session_state[f"{tab_id}_playback_index"] = -1

