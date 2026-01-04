"""Tabular Q-learning agent reusable across notebooks and Streamlit app."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

from qlearning.environment import ACTIONS, LineWorld


class QLearningAgent:
    """Simple tabular Q-learning implementation."""

    def __init__(
        self,
        env: LineWorld,
        alpha: float = 0.7,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table defaulting to 0.0
        self.Q: Dict[Tuple[int, str], float] = defaultdict(float)

    # ------------------------------------------------------------------
    # ε-greedy policy ---------------------------------------------------

    def _greedy_action(self, state: int, *, eval_mode: bool = False) -> str | None:
        """Return action using ε-greedy (ε=0 in *eval_mode*)."""
        if self.env.is_terminal(state):
            return None
        eps = 0.0 if eval_mode else self.epsilon

        import random

        if random.random() < eps:
            return random.choice(list(ACTIONS))

        # Greedy tie-break
        q_values = {a: self.Q[(state, a)] for a in ACTIONS}
        max_q = max(q_values.values())
        best = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best)

    # ------------------------------------------------------------------
    # Learning ----------------------------------------------------------

    def _q_update(self, s: int, a: str, r: float, s_next: int) -> None:
        """One Q-learning update."""
        max_next = 0.0 if self.env.is_terminal(s_next) else max(
            self.Q[(s_next, "L")], self.Q[(s_next, "R")]
        )
        target = r + self.gamma * max_next
        self.Q[(s, a)] += self.alpha * (target - self.Q[(s, a)])

    def train(self, start_state: int, episodes: int = 500, max_steps: int = 50) -> None:
        """Learn from *episodes* starting at *start_state*."""
        for _ in range(episodes):
            s = start_state
            steps = 0
            while not self.env.is_terminal(s) and steps < max_steps:
                a = self._greedy_action(s)  # type: ignore[arg-type]
                if a is None:
                    break
                s_next, r, _ = self.env.step(s, a)
                self._q_update(s, a, r, s_next)
                s = s_next
                steps += 1

    # ------------------------------------------------------------------
    # Evaluation --------------------------------------------------------

    def greedy_path(self, start_state: int, max_steps: int = 50) -> list[int]:
        """Roll out the greedy policy (ε=0) returning visited states list."""
        s = start_state
        path = [s]
        steps = 0
        while not self.env.is_terminal(s) and steps < max_steps:
            a = self._greedy_action(s, eval_mode=True)  # type: ignore[arg-type]
            if a is None:
                break
            s, *_ = self.env.step(s, a)
            path.append(s)
            steps += 1
        return path

