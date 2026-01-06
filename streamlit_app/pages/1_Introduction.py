import streamlit as st

st.set_page_config(page_title="Introduction to Q-Learning", layout="wide")

st.title("Introduction to Q-Learning")

st.markdown(
    r"""
    Q-learning is a **model-free** reinforcement learning algorithm, meaning it doesn't require prior knowledge of the environment's dynamics. Instead, it learns by interacting with the environment and updating its estimates of action values.
    
    ### The Q-Value
    
    At the heart of Q-learning is the Q-values $Q(s, a)$, which represents the expected cumulative reward of taking action $a$ in state $s$ and then following the optimal policy thereafter. The Q-values answer the question: "How good is it to take this action in this state?"
    
    ### The Learning Process
    
    Q-learning maintains a **Q-matrix** that stores Q-values for each state-action pair. The Q-values get updated (learns) through the following steps:
    
    0. **Initialisation**: Set the Q-values to 0 for all state-action pairs. This is the initial state of the Q-matrix.

    1. **Starting state**: Choose a starting state from the state space - either randomly, or from a fixed starting position. 
    
    In the next page (Dog and Bone Example), you can experiment with different starting positions to see how the result would be the same no matter whether a starting position is fixed or random.
    
    2. **Exploration vs Exploitation**: The agent uses an **ε-greedy policy** to balance exploration (trying new actions) and exploitation (choosing the best-known action). For fixed exploration, with probability ε, it explores randomly; otherwise, with probability 1-ε, it chooses the action with the highest Q-value. Note that if ε is 0, the agent will only exploit the best-known action, and if ε is 1, the agent will only explore randomly.

    Some other types of exploration rate would include exponential decay, where the exploration rate ε decreases over each step taken.
    
    3. **Q-Value Update**: After taking an action and observing the reward and next state, the agent updates the Q-value using the **Bellman equation**:
    
    $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
           
       Where:
       - **$\alpha$ (alpha)** is the learning rate - controls how much new information overrides old estimates, range 0 to 1,
       - **$\gamma$ (gamma)** is the discount factor - determines the importance of future rewards, range 0 to 1
       - **$r$** is the immediate reward received when goal is reached, otherwise the reward is 0,
       - **$s'$** is the next state, which is the state the agent ends up in after taking action $a$ at state $s$,
       - **$\max_{a'} Q(s', a')$** is the maximum Q-value for the next state $s'$, by comparing the Q-values of all possible actions at state $s'$. The action $a'$ is the action that maximises the Q-value for the next state $s'$.
       - Note that the Q matrix gets updated after each step taken, not after each episode.
    
    4. **Episode completion**: Repeat steps 1-3 until the goal is reached, which is when one episode is completed. 
    
    5. **New episode**: To train a new episode, repeat steps 1-4 again.
    
    6. **Convergence**: Over many episodes, the Q-values converge to the optimal values, and the agent learns the optimal policy.
    

    In the following pages, you'll see Q-learning in action through interactive examples!
    """
)
