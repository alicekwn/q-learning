"""Home page for Q-Learning demo multipage Streamlit app."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Q-Learning Demo", layout="wide")

st.title("Q-Learning Demo")

st.markdown(
    """
Welcome! 

In this webpage, we will learn about Q-learning, a type of reinforcement learning algorithm. 

We will start with a simple example of a dog learning to find the bone in a 1D grid.

Then, we will explore a more complex example of a dog learning to find the bone in a 2D grid.

Finally, we will explore Q-learning's application in economics pricing strategies.

Use the sidebar on the left to navigate between the pages:

1. **Introduction** – Overview of Q-learning
2. **Dog & Bone Example** – interactive demonstration of Q matrix updating under "1D Grid" and "2D Grid" scenarios
3. **Economics Pricing Example** – application of Q-learning in economics pricing strategies
4. **Future Environment** – Placeholder for a more complex setup

It is recommended to use a desktop computer for the best experience.
"""
)
