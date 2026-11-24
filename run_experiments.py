import argparse
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from train_a2c import interpolate_curves, train_a2c, train_reinforce


def evaluate_config(name, runner_fn, seeds, total_steps, threshold_score):
    print(f"  [Running] {name} across {len(seeds)} seeds...")
    runs, wall_times = [], []
    for s in seeds:
        t0 = time.time()
        steps, rets = runner_fn(s)
        wall_times.append(time.time() - t0)
        runs.append((steps, rets))

    grid, mean_curve, std_curve = interpolate_curves(runs, max_steps=total_steps)

    steps_to_thresh = []
    for steps, rets in runs:
        win = min(15, max(1, len(rets)))
        smooth = np.convolve(rets, np.ones(win) / win, mode="same") if len(rets) > 0 else np.array([])
        idx = np.where(smooth >= threshold_score)[0]
        steps_to_thresh.append(steps[idx[0]] if len(idx) > 0 else total_steps)

    summary = {
        "name": name,
        "final_mean": float(mean_curve[-1]),
        "final_std": float(std_curve[-1]),
        "steps_to_thresh_mean": float(np.mean(steps_to_thresh)),
        "steps_to_thresh_std": float(np.std(steps_to_thresh)),
        "wall_time_mean": float(np.mean(wall_times)),
        "grid": grid,
        "mean_curve": mean_curve,
        "std_curve": std_curve,
    }
    print(
        f"    -> Final Return: {summary['final_mean']:.1f} ± {summary['final_std']:.1f} | "
        f"Steps to {threshold_score}: {summary['steps_to_thresh_mean']:.0f} | "
        f"Time: {summary['wall_time_mean']:.1f}s"
    )
    return summary


def plot_group(results_list, title, filename, solved_line):
    plt.figure(figsize=(8.5, 5))
    for res in results_list:
        label = f"{res['name']} ({res['final_mean']:.1f} ± {res['final_std']:.1f})"
        plt.plot(res["grid"], res["mean_curve"], label=label, linewidth=2)
        plt.fill_between(
            res["grid"],
            res["mean_curve"] - res["std_curve"],
            res["mean_curve"] + res["std_curve"],
            alpha=0.18,
        )
    plt.axhline(solved_line, color="gray", linestyle="--", linewidth=1, label=f"Solved Threshold ({solved_line})")
    plt.xlabel("Environment Steps")
    plt.ylabel("Episodic Return (Mean ± SD)")
    plt.title(title)
    plt.grid(True, linestyle=":")
    plt.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"  Saved plot: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Run A2C Benchmark Suite")
    parser.add_argument("--config", type=str, default="configs/cartpole.json", help="Path to JSON config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    os.makedirs("results", exist_ok=True)
    env_id = cfg["env_id"]
    tag = cfg.get("tag", env_id.split("-")[0].lower())
    total_steps = cfg["total_timesteps"]
    solved_score = cfg["solved_score"]
    hidden_dim = cfg["hidden_dim"]
    seeds = cfg["seeds"]
    gamma = cfg.get("gamma", 0.99)
    vf_coef = cfg.get("vf_coef", 0.5)
    ent_coef = cfg.get("ent_coef", 0.01)
    max_grad_norm = cfg.get("max_grad_norm", 0.5)
    default_lr = cfg.get("lr", 7e-4)

    all_summaries = []

    exp1 = [
        ("REINFORCE (Baseline, N=1)", lambda s: train_reinforce(env_id, total_steps, default_lr, gamma, hidden_dim, s)),
        ("A2C (Single Worker, N=1, n=5)", lambda s: train_a2c(env_id, 1, 5, total_steps, default_lr, gamma, vf_coef, ent_coef, max_grad_norm, hidden_dim, s)),
        ("A2C (Parallel, N=8, n=5)", lambda s: train_a2c(env_id, 8, 5, total_steps, default_lr, gamma, vf_coef, ent_coef, max_grad_norm, hidden_dim, s)),
    ]
    res1 = [evaluate_config(n, fn, seeds, total_steps, solved_score) for n, fn in exp1]
    all_summaries.extend(res1)
    plot_group(res1, f"{env_id} — Reproduction: Parallel n-Step A2C vs. REINFORCE", f"results/{tag}_exp1_reproduction.png", solved_score)

    exp2 = [
        (f"N = {nw} Workers (n=5)", lambda s, nw=nw: train_a2c(env_id, nw, 5, total_steps, default_lr, gamma, vf_coef, ent_coef, max_grad_norm, hidden_dim, s))
        for nw in cfg["sweep_num_envs"]
    ]
    res2 = [evaluate_config(n, fn, seeds, total_steps, solved_score) for n, fn in exp2]
    all_summaries.extend(res2)
    plot_group(res2, f"{env_id} — Sensitivity 1: Parallel Workers N", f"results/{tag}_exp2_workers.png", solved_score)

    exp3 = [
        (f"n = {ns} Steps (N=8)", lambda s, ns=ns: train_a2c(env_id, 8, ns, total_steps, default_lr, gamma, vf_coef, ent_coef, max_grad_norm, hidden_dim, s))
        for ns in cfg["sweep_n_steps"]
    ]
    res3 = [evaluate_config(n, fn, seeds, total_steps, solved_score) for n, fn in exp3]
    all_summaries.extend(res3)
    plot_group(res3, f"{env_id} — Sensitivity 2: n-Step Rollout Horizon n", f"results/{tag}_exp3_nsteps.png", solved_score)

    exp4 = [
        (f"lr = {lv} (N=8, n=5)", lambda s, lv=lv: train_a2c(env_id, 8, 5, total_steps, lv, gamma, vf_coef, ent_coef, max_grad_norm, hidden_dim, s))
        for lv in cfg["sweep_lrs"]
    ]
    res4 = [evaluate_config(n, fn, seeds, total_steps, solved_score) for n, fn in exp4]
    all_summaries.extend(res4)
    plot_group(res4, f"{env_id} — Sensitivity 3: Learning Rate", f"results/{tag}_exp4_lr.png", solved_score)

    serializable = [{k: v for k, v in item.items() if k not in ("grid", "mean_curve", "std_curve")} for item in all_summaries]
    with open(f"results/{tag}_summary_metrics.json", "w") as f:
        json.dump(serializable, f, indent=2)


if __name__ == "__main__":
    main()
