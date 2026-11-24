# Advantage Actor-Critic (A2C) — Code Submission

## Project Overview
This submission implements **Advantage Actor-Critic (A2C)** (based on Mnih et al., 2016) and a **Monte Carlo REINFORCE** baseline in PyTorch using Gymnasium vectorized environments (`SyncVectorEnv`), evaluated across **CartPole-v1** and **LunarLander-v2**.

## Repository Structure
- `train_a2c.py`: Custom PyTorch `ActorCritic` network, $n$-step bootstrapped return computation, advantage estimation, combined actor-critic-entropy loss, gradient clipping, and REINFORCE baseline.
- `run_experiments.py`: Multi-seed experiment runner for main reproduction and sensitivity studies across parallel workers ($N$), rollout horizons ($n$), and learning rates ($\alpha$).
- `configs/cartpole.json`: Experiment configuration for `CartPole-v1` (100k steps, hidden width 64).
- `configs/lunarlander.json`: Experiment configuration for `LunarLander-v2` (300k steps, hidden width 128).
- `results/`: Output folder generated when running experiments (learning curve PNGs and summary JSON files).

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run CartPole-v1 Experiments
```bash
python run_experiments.py --config configs/cartpole.json
```

### 3. Run LunarLander-v2 Experiments
```bash
python run_experiments.py --config configs/lunarlander.json
```

## Generated Main Plots
- `results/cartpole_exp1_reproduction.png`, `results/cartpole_exp2_workers.png`, `results/cartpole_exp3_nsteps.png`, `results/cartpole_exp4_lr.png`
- `results/lunarlander_exp1_reproduction.png`, `results/lunarlander_exp2_workers.png`, `results/lunarlander_exp3_nsteps.png`, `results/lunarlander_exp4_lr.png`
- `results/cartpole_summary_metrics.json` and `results/lunarlander_summary_metrics.json`
