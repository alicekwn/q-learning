"""Helper utilities"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from qlearning.agent import QLearningAgent


def q_table_dataframe(agent: QLearningAgent, states: List[int]) -> pd.DataFrame:
    """Return Q-table as a DataFrame indexed by *states*."""
    # Get actions from the environment (assumes 1D LineGrid)
    actions = list(agent.env.ACTIONS.keys())
    data = []
    for s in states:
        row = {a: agent.Q.get((s, a), 0.0) for a in actions}
        data.append(row)
    df = pd.DataFrame(data, index=states)
    df.columns = actions
    return df


def plot_q_evolution(records: List[Dict[str, float]]) -> None:
    """Plot Q-value evolution given episode records list."""
    if not records:
        return
    df = pd.DataFrame(records)
    df = df.set_index("Episode")
    plt.figure()
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.xlabel("Episode")
    plt.ylabel("Q-value")
    plt.title("Q-value evolution")
    plt.legend()
    plt.tight_layout()
    plt.show()
