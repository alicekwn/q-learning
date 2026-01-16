"""Tabular Q-learning agent reusable across notebooks and Streamlit app."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Protocol, Tuple, TypeVar, Any, Hashable

# Type variables for generic state and action types
State = TypeVar("State", bound=Hashable)
Action = TypeVar("Action", bound=Hashable)


class Environment(Protocol[State, Action]):
    """Protocol defining the interface for Q-learning environments.

    Any environment that implements these methods and attributes can be used
    with QLearningAgent.
    """

    ACTIONS: Dict[Action, Any]  # Action mapping (e.g., {"L": -1} or {"U": (0, 1)})

    def is_terminal(self, state: State) -> bool:
        """Return True if state is terminal."""

    def step(self, state: State, action: Action) -> Tuple[State, float, bool]:
        """Transition from state using action. Returns (next_state, reward, done)."""


class QLearningAgent:
    """Generic tabular Q-learning implementation supporting any environment type.

    The agent works with any environment that implements:
    - ACTIONS: dict mapping action names to their effects
    - is_terminal(state): returns True if state is terminal
    - step(state, action): returns (next_state, reward, done)

    Examples:
        - 1D LineGrid: states are int, actions are str ("L", "R")
        - 2D RectangularGrid: states are tuple[int, int], actions are str ("U", "D", "L", "R")
        - Future: states could be any hashable type, actions could be prices, etc.
    """

    def __init__(
        self,
        env: Any,  # Environment[State, Action] - using Any for flexibility
        alpha: float = 0.7,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table: (state, action) -> Q-value
        # States and actions can be any hashable type
        self.Q: Dict[Tuple[Hashable, Hashable], float] = defaultdict(float)

    def _get_actions(self) -> list[Hashable]:
        """Get available actions from the environment."""
        return list(self.env.ACTIONS.keys())

    # ------------------------------------------------------------------
    # ε-greedy policy ---------------------------------------------------

    def _greedy_action_constant(
        self, state: Hashable, *, eval_mode: bool = False
    ) -> Hashable | None:
        """Return action using ε-greedy (ε=0 in *eval_mode*).

        Works with any hashable state type (int, tuple, custom objects, etc.).
        """
        if self.env.is_terminal(state):
            return None
        eps = 0.0 if eval_mode else self.epsilon

        import random

        actions = self._get_actions()
        if random.random() < eps:
            return random.choice(actions)

        # Greedy tie-break
        q_values = {a: self.Q[(state, a)] for a in actions}
        max_q = max(q_values.values())
        best = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best)

    def choose_action(
        self, state: Hashable, *, eval_mode: bool = False
    ) -> Hashable | None:
        """Public method to choose an action using ε-greedy policy.

        Args:
            state: Current state
            eval_mode: If True, use greedy policy (ε=0), otherwise use ε-greedy

        Returns:
            Selected action or None if state is terminal
        """
        return self._greedy_action_constant(state, eval_mode=eval_mode)

    # def _greedy_action_decay(self, state: int, *, eval_mode: bool = False) -> str | None:
    #     """Return action using ε-greedy while ε decays over each step t until it reaches 0.001(ε=0 in *eval_mode*)."""
    #
    #     # Greedy tie-break
    #     q_values = {a: self.Q[(state, a)] for a in ACTIONS}
    #     max_q = max(q_values.values())
    #     best = [a for a, q in q_values.items() if q == max_q]
    #     return random.choice(best)

    # ------------------------------------------------------------------
    # Learning ----------------------------------------------------------

    def _q_update(self, s: Hashable, a: Hashable, r: float, s_next: Hashable) -> None:
        """One Q-learning update. Works with any hashable state and action types."""
        if self.env.is_terminal(s_next):
            max_next = 0.0
        else:
            actions = self._get_actions()
            max_next = max(self.Q[(s_next, action)] for action in actions)
        target = r + self.gamma * max_next
        self.Q[(s, a)] += self.alpha * (target - self.Q[(s, a)])

    def update_q(
        self, state: Hashable, action: Hashable, reward: float, next_state: Hashable
    ) -> None:
        """Public method to perform one Q-learning update.

        Args:
            state: Current state
            action: Action taken in current state
            reward: Reward received after taking action
            next_state: Next state reached after taking action
        """
        self._q_update(state, action, reward, next_state)

    def train(
        self, start_state: Hashable, episodes: int = 500, max_steps: int = 50
    ) -> None:
        """Learn from *episodes* starting at *start_state*. Works with any state type."""
        for _ in range(episodes):
            s = start_state
            steps = 0
            while not self.env.is_terminal(s) and steps < max_steps:
                a = self._greedy_action_constant(s)
                if a is None:
                    break
                s_next, r, _ = self.env.step(s, a)
                self._q_update(s, a, r, s_next)
                s = s_next
                steps += 1

    # ------------------------------------------------------------------
    # Evaluation --------------------------------------------------------

    def greedy_path(self, start_state: Hashable, max_steps: int = 50) -> list[Hashable]:
        """Roll out the greedy policy (ε=0) returning visited states list.

        Returns list of states (type depends on environment).
        """
        s = start_state
        path: list[Hashable] = [s]
        steps = 0
        while not self.env.is_terminal(s) and steps < max_steps:
            a = self._greedy_action_constant(s, eval_mode=True)
            if a is None:
                break
            s, *_ = self.env.step(s, a)
            path.append(s)
            steps += 1
        return path
