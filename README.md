# Q-learning 
In this repo, Q learning examples are demonstrated, and economics concepts are explored.

## Setup

```
git clone https://github.com/alicekwn/q-learning
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

The app is deployed at https://qlearning-demo.streamlit.app/. 

To edit the app, run the interactive Q-learning demo locally:

```bash
streamlit run streamlit_app/Welcome.py
```

The app will open in your browser at `http://localhost:8501`.



### Notebooks

Research notebooks are in `notebooks/`:
- `1d_grid_example.ipynb` - a dog moving in a 1D grid example
- `2d_grid_example.ipynb` - a dog moving in a 2D grid example
- `econ_game.ipynb` - Economic game theory applications