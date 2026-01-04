"""Home page for Q-Learning demo multipage Streamlit app."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Q-Learning Demo", layout="wide")

st.title("Q-Learning Demo")

st.markdown(
    """
Welcome! 

In this webpage, we will learn about Q-learning, a type of reinforcement learning algorithm. 

We will start with a simple example of a dog learning to find the bone in a 1D world.

Then, we will explore a more complex example of a dog learning to find the bone in a 2D grid.

Finally, we will explore Q-learning's application in economics pricing strategies.

Use the sidebar on the left to navigate between the pages:

1. **Introduction** – Overview of Q-learning
2. **The Dog & Bone Demo** – Interactive 1-D demo
3. **Future Environment** – Placeholder for a more complex setup
4. **Advanced Experiments** – Placeholder for extended experiments
"""
)
