# LEADER: An LLM-Guided Evolutionary Feature Selection and Reinforcement-Learning-Driven Regime-Adaptive Decision Engine for Cryptocurrency Trading Strategies

LEADER is a novel **two-stage framework** designed to leverage the semantic knowledge of Large Language Models (LLMs) for time-series feature selection and subsequent Reinforcement Learning (RL) policy generation.

> ⚠️ **Research only. Not financial advice.**
> This code is intended for methodological research and reproducibility in the domains of FinTech and LLM/RL applications, not for live trading or financial consultation.

---

## The LEADER Framework: Two Stages

The framework is divided into two sequential stages:

1.  **Stage 1 – LLM-Guided GA Feature Selection**
    An LLM steers a genetic algorithm (GA) to select regime-salient features, assisted by a lightweight RL-based Q-evaluator. The LLM provides semantic guidance for initial population generation and adaptive operator control.

2.  **Stage 2 – RL Trading Engine (DQN)**
    A DQN-style RL agent learns a **directional–profit** trading policy on top of the features selected in Stage 1, using a rigorous walk-forward evaluation protocol.


## Repository Structure 📁

The core implementation files and output directories are organized as follows:

```text
.
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── LEADER.pdf                    # Manuscript (to be added soon)
├── Data/
│   └── crypto_signals.csv                  # Input bitcoin market data (time-ordered crypto signals)
└── Codes/
    ├── llm_ga.py                 # Stage 1: LLM-guided GA + lightweight Q-evaluator (LQE)
    ├── rl_dqn.py                 # Stage 2: Walk-forward DQN trading engine
    └── reports/                  # Output directory (created at runtime)
        ├── ga_with_llm/          # GA logs, HoF feature sets, system info
        ├── RL_results/           # Walk-forward segment-level RL outputs
        └── academic_results/     # Backtest metrics, tables for the paper

````

-----

## 🚀 Installation

### 1\. Python Environment

**Requirements:** Python 3.11+. Linux is recommended.

### 1. Python Environment (Conda)

Create and activate a conda environment:

```bash
conda create --name leader_env python=3.11 # Create environment
conda activate leader_env # Activate the environment
```

### 2\. Dependencies

Install the required packages from the minimal `requirements.txt`:

```text
pandas
pandas_ta
yfinance
backtesting
matplotlib
sqlalchemy
kaggle
python-dotenv
xgboost
langchain-openai
scikit-learn
tqdm
ollama
psutil
seaborn
wandb
deap
GPUtil
```

Install them with:

```bash
pip install -r requirements.txt
```

### 3\. Local LLM (Ollama)

Stage 1 requires a locally running LLM, managed via **Ollama** and accessed through `langchain_ollama`.

1.  **Install and run [Ollama](https://ollama.com/download/linux).**

2.  **Pull a model** (e.g., a large model like Llama 3.3 70B is used for the best results in the paper):

    ```bash
    ollama pull llama3.3:70b
    ```

3.  Ensure the `llm_config` in `llm_ga.py` points to your chosen model. No external LLM APIs (OpenAI, etc.) are required; all model calls are local.

-----

## Data Format

Both stages expect a time-ordered CSV of financial signals (e.g., BTC daily data) with the following minimum structure:

  * A date/time index column (e.g., `Date`).
  * OHLCV columns (`Open`, `High`, `Low`, `Close`, `Volume`).
  * **Derived technical features** (e.g., `entropy`, `kurtosis`, moving averages, etc.).

A target variable is computed internally.

| Date | Close | Open | High | Low | Volume | entropy | kurtosis | ... |
| :--- | :---- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2017-01-01 | 1000 | ... | ... | ... | ... | ... | ... | ... |
| 2017-01-02 | 1010 | ... | ... | ... | ... | ... | ... | ... |

**Action required:**
The scripts currently define a constant path: `FILE_PATH = "/path/to/your/crypto_signals.csv"`. You must change this path in **both** `llm_ga.py` and `rl_dqn.py` to point to your dataset (e.g., `FILE_PATH = "Data/crypto_signals.csv"`).

-----

## ⚡ Quickstart

### 1\. Run Stage 1 (Feature Selection)

This phase generates a Hall-of-Fame (HoF) of feature subsets.

1.  Edit `FILE_PATH`, the `feature_columns`, `market_context`, and  list in `llm_ga.py` to match your data and feature space.

2.  Run the script:

    ```bash
    python llm_ga.py
    ```

3.  After completion, identify the best feature subsets in:

      * `reports/ga_with_llm/hof_feature_sets.csv`

### 2\. Run Stage 2 (RL Trading Engine)

This phase trains the DQN agent and performs the backtest.

1.  Edit `FILE_PATH` in `rl_dqn.py` to point to the same CSV.

2.  Set `REQUESTED_FEATURES` in `rl_dqn.py` to one of the chosen HoF subsets from Stage 1.

3.  Run the script:

    ```bash
    python rl_dqn.py
    ```

4.  Inspect the final results:

      * `reports/RL_results/` for segment-level outputs.
      * `reports/academic_results/` for aggregated backtest-style performance metrics (ROI, Sharpe ratio, MDD, etc.).

-----

## Stage 1 – LLM-Guided GA Feature Selection

### File: `llm_ga.py`

This stage implements the search for optimal feature subsets using a **Genetic Algorithm (GA)**, where fitness is determined by LQE evaluator. The LLM's role is to:

  * Generate a semantically structured initial feature population.
  * Adapt crossover/mutation rates via **Trend-Aware Adaptive Operator Control (TA-AOC)** based on GA performance metrics.

### Key Configuration

The main configuration parameters are located in `main()` / `async_main()`:

```python
# Data & Temporal Configuration
LOOKBACK    = 15      # Number of past steps used as input for the RL state
TARGET_H    = 1       # Prediction horizon (h-step ahead)

# Feature Space Definition
feature_columns = [
    "Close", "Open", "High", "Low", "Volume",
    "entropy", "kurtosis", "mad",
    # ... rest of the indicators as defined in the script
]

# LLM Configuration (Local Ollama Model)
llm_config = {
    "model": "llama3.3:70b", # Must match the model pulled in Ollama
    "temperature": 0.7,
    # ... other Ollama/LangChain parameters
}
```

### Internal Flow Highlights

1.  **LLM-Guided Initial Population:** An `LLMGuidedInitializer` builds a feature glossary and calls the local LLM with a structured JSON prompt to generate binary masks for the initial population.
2.  **Lightweight Q-Evaluator (`LQE`):** Implements a small MLP Q-network using Expected SARSA to train a classifier on the feature subset and return classification metrics (accuracy/F1/Recall/Precision) as the GA fitness score.
3.  **GA Evolution + TA-AOC:** The `TrendAwareAdaptiveControlLLM` tracks performance and diversity, querying the LLM to adjust crossover ($p_c$) and mutation ($p_m$) every generation.

### Outputs (Stage 1)

All outputs are saved under `reports/ga_with_llm/`:

  * `hof_feature_sets.csv`: The Hall-of-Fame individuals (best feature subsets).
  * `ga_training_log.csv`: Per-generation statistics.
  * `fitness_over_generations.png`: Plot of min/avg/best fitness trend.
  * `gpu_info.csv`, `system_info.csv`, etc., for reproducibility.

-----

## Stage 2 – RL Trading Engine (DQN)

### File: `rl_dqn.py`

This stage trains a **DQN-style RL agent** to learn a trading policy. It uses a **directional–profit reward** and a rigorous **walk-forward evaluation** protocol to simulate real-world trading conditions.

### Key Configuration

The main configuration parameters are located in `main()`:

```python
# Walk-forward Settings (Matching the Manuscript Protocol)
INITIAL_TRAIN_SIZE = 1200 # Initial training window size
STEP_SIZE          = 250  # Number of steps to advance the window each time
TEST_SIZE          = 250  # Out-of-sample test window size

# Feature Subset (Chosen from Stage 1 HOF)
REQUESTED_FEATURES = [
    "High",
    "ppo_signal",
    # ... replace with your chosen HOF subset
]

# Reward Configuration
reward_configs = [
    {"type": "directional", "name": "directional"}, # Uses the combined reward
]

transaction_cost = 0.001    # 0.1% per trade
```

### Internal Flow Highlights

1.  **Walk-forward Loop:** The `walk_forward_dqn` function splits the data into rolling train/test segments. For each segment, a fresh `DQNAgent` is trained and tested.
2.  **Directional–Profit Reward:** The reward function combines a directional signal (for correct classification) scaled by the profit magnitude (T+1 executed returns), minus transaction costs.
3.  **Backtesting:** The `Backtester` converts predicted positions into executed trades (shifted T+1 execution) and applies transaction costs.

### Outputs (Stage 2)

All outputs are saved under `reports/`:

  * `reports/RL_results/`: Detailed per-segment training and testing logs.
  * `reports/academic_results/`: Overall performance summaries.
-----
