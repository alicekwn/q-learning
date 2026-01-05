"""Environment definitions for simple 1-D line-walking tasks.

The core notebook experiments model a discrete 1-dimensional line where an
agent can move left (−1) or right (+1).  This module keeps that logic in a
re-usable class so both notebooks and the Streamlit app can import it.
"""
from __future__ import annotations

from typing import List, Tuple



class LineGrid:

    ACTIONS = {"L": -1, "R": +1}
    """Finite or infinite line-grid environment.

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

        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        move = self.ACTIONS[action]
        next_state = max(self.low, min(self.high, state + move))
        reward = self.reward_value if self.is_terminal(next_state) else 0.0
        done = next_state == self.terminal_state
        return next_state, reward, done


class RectangularGrid:
    """2D rectangular grid environment with Cartesian coordinates.
    
    Parameters
    ----------
    x_start, x_end:
        X-axis bounds (inclusive)
    y_start, y_end:
        Y-axis bounds (inclusive)
    terminal_state:
        (x, y) tuple that ends the episode and delivers *reward*
    reward:
        Scalar returned when the agent first enters *terminal_state*
    """
    
    ACTIONS = {"U": (0, 1), "D": (0, -1), "L": (-1, 0), "R": (1, 0)}
    
    def __init__(
        self,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int,
        terminal_state: Tuple[int, int],
        reward: float = 1.0,
    ):
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.terminal_state = terminal_state
        self.reward_value = reward
    
    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """Return *True* if *state* is terminal."""
        return state == self.terminal_state
    
    def step(self, state: Tuple[int, int], action: str) -> Tuple[Tuple[int, int], float, bool]:
        """Transition from *state* using *action* ("U", "D", "L", or "R").
        
        Returns ``(next_state, reward, done)``.
        """
        if self.is_terminal(state):
            return state, 0.0, True
        
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action: {action}")
        
        dx, dy = self.ACTIONS[action]
        x, y = state
        nx = max(self.x_start, min(self.x_end, x + dx))
        ny = max(self.y_start, min(self.y_end, y + dy))
        next_state = (nx, ny)
        
        reward = self.reward_value if self.is_terminal(next_state) else 0.0
        done = next_state == self.terminal_state
        return next_state, reward, done
