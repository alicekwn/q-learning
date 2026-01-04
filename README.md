# Q-learning 
In this repo, Q learning examples are demonstrated, and economics concepts are explored.

## Setup

```
git clone https://github.com/zcahkwn/q-learning
cd q-learning
```

### Give execute permission to your script and then run `setup_repo.sh`

```
chmod +x setup_repo.sh
./setup_repo.sh
. venv/bin/activate
```

or follow the step-by-step instructions below between the two horizontal rules:

---

#### Create a python virtual environment

- MacOS / Linux

```bash
python3 -m venv venv
```

- Windows

```bash
python -m venv venv
```

#### Activate the virtual environment

- MacOS / Linux

```bash
. venv/bin/activate
```

- Windows (in Command Prompt, NOT Powershell)

```bash
venv\Scripts\activate.bat
```

#### Install toml

```
pip install toml
```

#### Install the project in editable mode

```bash
pip install -e ".[dev]"
```

---

## Usage

### Running the Streamlit App

After installation, run the interactive Q-learning demo:

```bash
streamlit run streamlit_app/app.py
```

The app will open in your browser at `http://localhost:8501`.

### Using the `qlearning` Package

The core Q-learning logic is available as a reusable package. Import it in notebooks or scripts:

```python
from qlearning import LineWorld, QLearningAgent

# Create environment
env = LineWorld(states=list(range(8)), terminal_state=5, reward=1.0)

# Create agent
agent = QLearningAgent(env, alpha=0.7, gamma=0.9, epsilon=0.4)

# Train
agent.train(start_state=2, episodes=500)

# Evaluate greedy policy
path = agent.greedy_path(start_state=2)
print(f"Path: {path}")
```

### Notebooks

Research notebooks are in `notebooks/`:
- `line_example.ipynb` - 1D line-world Q-learning examples
- `grid_example.ipynb` - 2D grid-world experiments
- `econ_game.ipynb` - Economic game theory applications

