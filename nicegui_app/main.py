from __future__ import annotations

from nicegui import ui

from .streamlit_compat import install_streamlit_stub
from . import dog_bone, econ_pricing


# Ensure the Streamlit compatibility layer is installed before importing
# any of the original `streamlit_app` modules that expect `streamlit`.
install_streamlit_stub()


def navbar() -> None:
    """Top navigation bar with links to all pages."""
    with ui.header().classes("items-center justify-between px-4"):
        ui.label("Q-Learning Demo").classes("text-lg font-semibold")
        with ui.row().classes("items-center gap-4"):
            ui.link("Home", "/")
            ui.link("Introduction", "/introduction")
            ui.link("Dog & Bone", "/dog-bone")
            ui.link("Economics Pricing", "/economics-pricing")
            ui.link("Future Environment", "/future-environment")


@ui.page("/")
def page_home() -> None:
    navbar()
    with ui.column().classes("w-full max-w-4xl mx-auto gap-4 py-8"):
        ui.label("Q-Learning Demo").classes("text-2xl font-bold")
        ui.markdown(
            """
Welcome!

In this web app, we explore Q-learning, a type of reinforcement learning algorithm.

Use the navigation links above to:

1. **Introduction** – Overview of Q-learning
2. **Dog & Bone Example** – Interactive 1D and 2D grid demos
3. **Economics Pricing Example** – Application of Q-learning in pricing strategies
4. **Future Environment** – Placeholder for a more complex setup

For the best experience, use a desktop browser.
"""
        )


@ui.page("/introduction")
def page_introduction() -> None:
    navbar()
    with ui.column().classes("w-full max-w-4xl mx-auto gap-4 py-8"):
        ui.label("Introduction to Q-Learning").classes("text-2xl font-bold")
        # The content here mirrors `pages/1_Introduction.py`
        ui.markdown(
            r"""
Q-learning is a **model-free** reinforcement learning algorithm, meaning it doesn't require prior knowledge of the environment's dynamics. Instead, it learns by interacting with the environment and updating its estimates of action values.

### The Q-Value

At the heart of Q-learning is the Q-values $Q(s, a)$, which represents the expected cumulative reward of taking action $a$ in state $s$ and then following the optimal policy thereafter. The Q-values answer the question: "How good is it to take this action in this state?"

### The Learning Process

Q-learning maintains a **Q-matrix** that stores Q-values for each state-action pair. The Q-values get updated (learns) through the following steps:

0. **Initialisation**: Set the Q-values to 0 for all state-action pairs. This is the initial state of the Q-matrix.

1. **Starting state**: Choose a starting state from the state space - either randomly, or from a fixed starting position. 

   In the Dog and Bone Example, you can experiment with different starting positions to see how the result would be the same no matter whether a starting position is fixed or random.

2. **Exploration vs Exploitation**: The agent uses an **ε-greedy policy** to balance exploration (trying new actions) and exploitation (choosing the best-known action). For fixed exploration, with probability ε, it explores randomly; otherwise, with probability 1-ε, it chooses the action with the highest Q-value. Note that if ε is 0, the agent will only exploit the best-known action, and if ε is 1, the agent will only explore randomly.

   Some other types of exploration rate would include exponential decay, where the exploration rate ε decreases over each step taken.

3. **Q-Value Update**: After taking an action and observing the reward and next state, the agent updates the Q-value using the **Bellman equation**:

   $$Q(s, a) \\leftarrow Q(s, a) + \\alpha [r + \\gamma \\max_{a'} Q(s', a') - Q(s, a)]$$
           
   Where:
   - **$\\alpha$ (alpha)** is the learning rate - controls how much new information overrides old estimates, range 0 to 1,
   - **$\\gamma$ (gamma)** is the discount factor - determines the importance of future rewards, range 0 to 1
   - **$r$** is the immediate reward received when goal is reached, otherwise the reward is 0,
   - **$s'$** is the next state, which is the state the agent ends up in after taking action $a$ at state $s$,
   - **$\\max_{a'} Q(s', a')$** is the maximum Q-value for the next state $s'$, by comparing the Q-values of all possible actions at state $s'$. The action $a'$ is the action that maximises the Q-value for the next state $s'$.
   - Note that the Q matrix gets updated after each step taken, not after each episode.

4. **Episode completion**: Repeat steps 1–3 until the goal is reached, which is when one episode is completed. 

5. **New episode**: To train a new episode, repeat steps 1–4 again.

6. **Convergence**: Over many episodes, the Q-values converge to the optimal values, and the agent learns the optimal policy.

In the following pages, you'll see Q-learning in action through interactive examples!
"""
        )


@ui.page("/dog-bone")
def page_dog_bone() -> None:
    navbar()
    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 py-8"):
        dog_bone.render_page()


@ui.page("/economics-pricing")
def page_economics_pricing() -> None:
    navbar()
    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 py-8"):
        econ_pricing.render_page()


@ui.page("/future-environment")
def page_future_environment() -> None:
    navbar()
    with ui.row().classes("w-full max-w-5xl mx-auto gap-4 py-8"):
        with ui.column().classes("w-1/4 gap-2"):
            ui.label("Future Environment").classes("text-xl font-semibold")
            ui.label("Controls (Coming Soon)").classes("text-base font-medium")
            ui.label("Independent config will go here").classes("text-sm text-gray-600")
        with ui.column().classes("w-px h-full"):
            ui.html("<div style='border-left: 2px solid #e0e0e0; height: 100%;'></div>")
        with ui.column().classes("flex-1 gap-2"):
            ui.label("Environment visualization will go here").classes(
                "text-sm text-gray-700"
            )


def run() -> None:
    """Run the NiceGUI app."""
    ui.run()


# Allow NiceGUI's multiprocessing setup (which may import this module
# with __name__ == "__mp_main__") to start the server correctly.
if __name__ in {"__main__", "__mp_main__"}:
    run()
