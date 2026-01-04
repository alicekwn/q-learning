"""Environment definitions for simple 1-D line-walking tasks.

The core notebook experiments model a discrete 1-dimensional line where an
agent can move left (−1) or right (+1).  This module keeps that logic in a
re-usable class so both notebooks and the Streamlit app can import it.
"""
from __future__ import annotations

from typing import List, Tuple

ACTIONS = {"L": -1, "R": +1}


class LineWorld:
    """Finite or infinite line-world environment.

    Parameters
    ----------
    states:
        Ordered list of *legal* integer states.  For the infinite variant you
        can pass a range like ``range(-20, 21)`` to approximate unbounded
        space.
    terminal_state:
        Reaching this state ends the episode and delivers *reward*.
    reward:
        Scalar returned when the agent first enters *terminal_state*.
    """

    def __init__(self, states: List[int], terminal_state: int, reward: float = 1.0):
        self.states = list(states)
        self.terminal_state = terminal_state
        self.reward_value = reward
        self.low = min(self.states)
        self.high = max(self.states)

    # ---------------------------------------------------------------------
    # Core API ----------------------------------------------------------------

    def is_terminal(self, state: int) -> bool:
        """Return *True* if *state* is terminal."""
        return state == self.terminal_state

    def step(self, state: int, action: str) -> Tuple[int, float, bool]:
        """Transition from *state* using *action* ("L" or "R").

        Returns ``(next_state, reward, done)``.
        """
        if self.is_terminal(state):
            return state, 0.0, True

        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        move = ACTIONS[action]
        next_state = max(self.low, min(self.high, state + move))
        reward = self.reward_value if self.is_terminal(next_state) else 0.0
        done = next_state == self.terminal_state
        return next_state, reward, done

