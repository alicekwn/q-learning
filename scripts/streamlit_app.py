import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time

# ==========================================
# 1. APP CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="Algorithmic Pricing Lab", layout="wide", page_icon="⚖️")

st.title("⚖️ The Algorithmic Pricing Lab")
st.markdown("""
Welcome to the lab. Here you will learn how AI learns to make decisions, and then you will 
train your own AI Pricing Manager to compete against your classmates.
""")

# ==========================================
# 2. SHARED GAME THEORY CONFIG (The "Rules")
# ==========================================
# We must fix the price grid so all students' agents are compatible in the battle.
PRICE_MIN = 24
PRICE_MAX = 84
PRICE_STEP = 2
PRICES = np.arange(PRICE_MIN, PRICE_MAX + PRICE_STEP/2, PRICE_STEP).tolist()
N_ACTIONS = len(PRICES)
PRICE_TO_IDX = {p: i for i, p in enumerate(PRICES)}

def get_demand(p_own, p_competitor):
    # From your notebook: q = 62 - p1 + 0.5*p2
    q = 62 - p_own + 0.5 * p_competitor
    return max(0.0, q)

def get_profit(p_own, p_competitor, cost=2.0):
    dem = get_demand(p_own, p_competitor)
    return (p_own - cost) * dem

# State encoding: State = (Index_Price1, Index_Price2)
# We flatten this to a single integer for Q-table indexing
def get_state_index(idx1, idx2):
    return idx1 * N_ACTIONS + idx2

def get_indices_from_state(s):
    idx1 = s // N_ACTIONS
    idx2 = s % N_ACTIONS
    return idx1, idx2

# ==========================================
# TAB 1: THE BASICS (1D WALK)
# ==========================================
tab1, tab2, tab3 = st.tabs(["🎓 Part 1: Learning the Mechanics", "🧠 Part 2: Train Your Agent", "⚔️ Part 3: The Battle Arena"])

with tab1:
    st.header("How Q-Learning Works")
    st.info("Goal: Teach a robot to walk from Start to Goal. Watch how the numbers (Q-values) change.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Settings")
        grid_size = 6
        goal_state = 5
        start_pos = 0
        
        lr = st.slider("Learning Rate (Alpha)", 0.1, 1.0, 0.5, help="How fast we accept new info")
        gamma = st.slider("Discount (Gamma)", 0.1, 1.0, 0.9, help="How much we care about the future")
        
        if st.button("Step 1: Initialize Brain"):
            # Reset session state
            st.session_state['q_1d'] = np.zeros((grid_size, 2)) # Actions: 0=Left, 1=Right
            st.session_state['agent_pos'] = start_pos
            st.session_state['step_count'] = 0
            st.success("Brain reset to zeros!")

    with col2:
        if 'q_1d' not in st.session_state:
            st.session_state['q_1d'] = np.zeros((grid_size, 2))
            st.session_state['agent_pos'] = start_pos
            st.session_state['step_count'] = 0

        # Visualization of the 1D World
        world_map = ["⬜"] * grid_size
        world_map[goal_state] = "🚩"
        world_map[st.session_state['agent_pos']] = "🤖"
        st.write(f"**World:** {' '.join(world_map)}")
        
        # Action Buttons
        c1, c2, c3 = st.columns(3)
        action = None
        if c1.button("Move Left ⬅️"): action = 0
        if c2.button("Move Right ➡️"): action = 1
        
        if action is not None:
            curr = st.session_state['agent_pos']
            
            # Logic
            move = -1 if action == 0 else 1
            next_s = max(0, min(grid_size-1, curr + move))
            reward = 10 if next_s == goal_state else 0
            
            # Bellman Calculation Display
            old_q = st.session_state['q_1d'][curr, action]
            max_future_q = np.max(st.session_state['q_1d'][next_s])
            
            # The Equation
            new_q = old_q + lr * (reward + gamma * max_future_q - old_q)
            
            st.markdown(f"""
            ### The Bellman Update:
            $$Q(s,a) \\leftarrow Q(s,a) + \\alpha [ R + \\gamma \\max Q(s') - Q(s,a) ]$$
            
            **Your Move:**
            * Old Value: `{old_q:.2f}`
            * Reward: `{reward}`
            * Max Future Value: `{max_future_q:.2f}`
            * **New Value:** `{new_q:.2f}`
            """)
            
            # Update State
            st.session_state['q_1d'][curr, action] = new_q
            st.session_state['agent_pos'] = next_s
            
            if next_s == goal_state:
                st.balloons()
                st.session_state['agent_pos'] = start_pos # Reset pos
        
        # Show Matrix
        st.write("### The Robot's Memory (Q-Table)")
        df_q = pd.DataFrame(st.session_state['q_1d'], columns=["Left", "Right"])
        st.dataframe(df_q.style.background_gradient(cmap="Blues"))

# ==========================================
# TAB 2: GAME THEORY TRAINING
# ==========================================
with tab2:
    st.header("Train Your Pricing Algorithm")
    st.markdown("""
    You are training an AI to set prices in a market with one competitor.
    * **Collusion:** Both set High prices (High Profit).
    * **Nash Equilibrium:** Both set Low prices (Medium Profit).
    * **Betrayal:** One High, One Low (Winner takes all).
    
    Adjust the parameters to change your agent's personality!
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hyperparameters")
        alpha_gt = st.number_input("Learning Rate", 0.01, 0.5, 0.125)
        gamma_gt = st.number_input("Discount Factor", 0.5, 0.99, 0.95)
        episodes_gt = st.slider("Training Episodes", 1000, 200000, 50000, step=1000)
        
        student_name = st.text_input("Name your Agent", "Agent_001")
        
    with col2:
        st.subheader("Training Progress")
        train_btn = st.button("Start Training Simulation")
        status_text = st.empty()
        progress_bar = st.progress(0)
    
    if train_btn:
        # Initialize Q-Tables (We assume Agent is Player 1, Opponent is Player 2)
        # We train them together to simulate a market environment
        n_states = N_ACTIONS * N_ACTIONS
        Q1 = np.zeros((n_states, N_ACTIONS))
        Q2 = np.zeros((n_states, N_ACTIONS))
        
        # Training Loop (Simplified for speed)
        # Exploration decay
        beta = 2e-5
        
        start_time = time.time()
        
        # Use a random state
        curr_p1_idx = np.random.randint(0, N_ACTIONS)
        curr_p2_idx = np.random.randint(0, N_ACTIONS)
        curr_s = get_state_index(curr_p1_idx, curr_p2_idx)
        
        for t in range(episodes_gt):
            epsilon = np.exp(-beta * t)
            
            # Action Selection
            if np.random.random() < epsilon:
                a1 = np.random.randint(0, N_ACTIONS)
                a2 = np.random.randint(0, N_ACTIONS)
            else:
                a1 = np.argmax(Q1[curr_s])
                a2 = np.argmax(Q2[curr_s])
            
            # Next State
            next_s = get_state_index(a1, a2)
            
            # Rewards
            p1 = PRICES[a1]
            p2 = PRICES[a2]
            r1 = get_profit(p1, p2)
            r2 = get_profit(p2, p1)
            
            # Updates
            Q1[curr_s, a1] = (1-alpha_gt)*Q1[curr_s, a1] + alpha_gt*(r1 + gamma_gt*np.max(Q1[next_s]))
            Q2[curr_s, a2] = (1-alpha_gt)*Q2[curr_s, a2] + alpha_gt*(r2 + gamma_gt*np.max(Q2[next_s]))
            
            curr_s = next_s
            
            if t % (episodes_gt // 10) == 0:
                progress_bar.progress(t / episodes_gt)
        
        progress_bar.progress(100)
        st.success(f"Training Complete in {time.time() - start_time:.2f} seconds!")
        
        # Save Q1 to session state (This is the student's agent)
        # We save it as a list for JSON serialization
        agent_data = {
            "name": student_name,
            "q_matrix": Q1.tolist(),
            "params": {"alpha": alpha_gt, "gamma": gamma_gt, "episodes": episodes_gt}
        }
        
        # Create Download Button
        json_str = json.dumps(agent_data)
        st.download_button(
            label="💾 Download Your Agent Strategy",
            data=json_str,
            file_name=f"{student_name}_strategy.json",
            mime="application/json"
        )
        
        # Preview Heatmap
        st.write("### Your Agent's Price Map")
        st.caption("X-Axis: Competitor Price, Y-Axis: Your Price. Color: Value")
        # Just showing a slice for visualization
        fig, ax = plt.subplots()
        # Reshape a slice for visualization (Price vs Price)
        # We take the max Q value for each state
        value_matrix = np.max(Q1, axis=1).reshape(N_ACTIONS, N_ACTIONS)
        im = ax.imshow(value_matrix, cmap='viridis', origin='lower')
        ax.set_xlabel("Competitor Price Index")
        ax.set_ylabel("My Previous Price Index")
        st.pyplot(fig)

# ==========================================
# TAB 3: BATTLE ARENA
# ==========================================
with tab3:
    st.header("⚔️ The Market Arena")
    st.write("Upload two strategy files to see how they compete in the market.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        file_a = st.file_uploader("Upload Agent A (JSON)", type="json", key="a")
    with col_b:
        file_b = st.file_uploader("Upload Agent B (JSON)", type="json", key="b")
        
    if file_a and file_b:
        data_a = json.load(file_a)
        data_b = json.load(file_b)
        
        st.divider()
        st.subheader(f"Matchup: {data_a['name']} vs {data_b['name']}")
        
        # Convert lists back to numpy arrays
        Q_A = np.array(data_a['q_matrix'])
        Q_B = np.array(data_b['q_matrix'])
        
        if st.button("▶️ Run Simulation"):
            # Start at a neutral price (middle index)
            start_idx = N_ACTIONS // 2
            
            # State: (Last Price A, Last Price B)
            # In Q-Learning formulation used: State S at time t was defined by prices at t-1
            # Action at time t sets Prices at time t
            
            # Simulation container
            history = []
            
            # Initial state
            curr_idx_a = start_idx
            curr_idx_b = start_idx
            
            # Run for 50 steps
            for t in range(50):
                # 1. Get current state index
                s_idx = get_state_index(curr_idx_a, curr_idx_b)
                
                # 2. Both agents choose action simultaneously based on state
                # Agent A looks at Q_A[s]
                # Agent B looks at Q_B[s] NOTE: For B, the state is (Price B, Price A) or (Price A, Price B)?
                # To simplify, we assumed symmetric training in Tab 2. 
                # Q-tables map State(P_me, P_them) -> Value of my next price.
                
                state_for_A = get_state_index(curr_idx_a, curr_idx_b)
                state_for_B = get_state_index(curr_idx_b, curr_idx_a) # Flip perspective for B
                
                action_a = np.argmax(Q_A[state_for_A])
                action_b = np.argmax(Q_B[state_for_B])
                
                price_a = PRICES[action_a]
                price_b = PRICES[action_b]
                
                profit_a = get_profit(price_a, price_b)
                profit_b = get_profit(price_b, price_a)
                
                history.append({
                    "Step": t,
                    "Price A": price_a,
                    "Price B": price_b,
                    "Profit A": profit_a,
                    "Profit B": profit_b
                })
                
                # Update state for next turn
                curr_idx_a = action_a
                curr_idx_b = action_b
            
            # RESULTS
            df_hist = pd.DataFrame(history)
            
            # 1. Plot Prices
            st.write("### Price War Trajectory")
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(df_hist["Step"], df_hist["Price A"], label=data_a['name'], marker='o')
            ax1.plot(df_hist["Step"], df_hist["Price B"], label=data_b['name'], marker='x')
            ax1.set_ylabel("Price")
            ax1.set_xlabel("Time Step")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            
            # 2. Stats
            avg_a = df_hist["Profit A"].mean()
            avg_b = df_hist["Profit B"].mean()
            
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{data_a['name']} Avg Profit", f"${avg_a:.2f}")
            c2.metric(f"{data_b['name']} Avg Profit", f"${avg_b:.2f}")
            
            winner = "Draw"
            if avg_a > avg_b: winner = data_a['name']
            elif avg_b > avg_a: winner = data_b['name']
            
            c3.metric("Winner", winner)
            
            # 3. Market Outcome Analysis
            st.write("### Market Analysis")
            last_price_a = df_hist.iloc[-1]["Price A"]
            
            if last_price_a > 60:
                st.success("🤝 **Outcome: Collusion.** The agents learned to keep prices high together!")
            elif last_price_a < 45:
                st.error("📉 **Outcome: Price War.** The agents competed prices down to the Nash Equilibrium.")
            else:
                st.warning("🔄 **Outcome: Cycle.** The agents are stuck in a price loop.")