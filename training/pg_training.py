"""
pg_training.py - Sista Health RL
====================================
Trains PPO and REINFORCE using Stable Baselines 3.
Runs 10 hyperparameter experiments each automatically.
Saves best models, reward curves, entropy curves, and results tables.

Usage:
    python training/pg_training.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import SistaHealthEnv

os.makedirs("models/pg/ppo", exist_ok=True)
os.makedirs("models/pg/reinforce", exist_ok=True)
os.makedirs("results", exist_ok=True)

TIMESTEPS = 100_000


class PGCallback(BaseCallback):
    def __init__(self):
        super().__init__(0)
        self.episode_rewards = []
        self.current_rewards = []
        self.entropy_log     = []

    def _on_step(self):
        self.current_rewards.append(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episode_rewards.append(sum(self.current_rewards))
            self.current_rewards = []
        try:
            dist = self.model.policy.action_dist
            self.entropy_log.append(dist.entropy().mean().item())
        except Exception:
            pass
        return True


def evaluate_model(model, n=30):
    env = SistaHealthEnv()
    rewards = []
    for _ in range(n):
        obs, _ = env.reset()
        ep_r, done = 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action))
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    env.close()
    return np.mean(rewards), np.std(rewards)


# ============================================================
# PPO EXPERIMENTS
# ============================================================

PPO_EXPS = [
    # Run 1 - Baseline
    {"learning_rate": 3e-4,  "gamma": 0.99,  "n_steps": 2048, "ent_coef": 0.01,  "clip_range": 0.2,  "gae_lambda": 0.95},
    # Run 2 - Low LR + short rollout + low entropy (conservative)
    {"learning_rate": 1e-4,  "gamma": 0.99,  "n_steps": 512,  "ent_coef": 0.005, "clip_range": 0.15, "gae_lambda": 0.92},
    # Run 3 - High LR + long rollout + high entropy (aggressive)
    {"learning_rate": 1e-3,  "gamma": 0.99,  "n_steps": 4096, "ent_coef": 0.05,  "clip_range": 0.2,  "gae_lambda": 0.98},
    # Run 4 - Low gamma + short rollout (myopic)
    {"learning_rate": 3e-4,  "gamma": 0.85,  "n_steps": 512,  "ent_coef": 0.01,  "clip_range": 0.2,  "gae_lambda": 0.90},
    # Run 5 - High gamma + wide clip (far-sighted)
    {"learning_rate": 3e-4,  "gamma": 0.995, "n_steps": 2048, "ent_coef": 0.01,  "clip_range": 0.3,  "gae_lambda": 0.98},
    # Run 6 - Zero entropy + tight clip (pure exploitation)
    {"learning_rate": 5e-4,  "gamma": 0.99,  "n_steps": 1024, "ent_coef": 0.0,   "clip_range": 0.1,  "gae_lambda": 0.95},
    # Run 7 - High entropy + low gamma + long rollout
    {"learning_rate": 2e-4,  "gamma": 0.90,  "n_steps": 4096, "ent_coef": 0.05,  "clip_range": 0.25, "gae_lambda": 0.92},
    # Run 8 - Medium LR + medium rollout + tight clip (stable balanced)
    {"learning_rate": 3e-4,  "gamma": 0.97,  "n_steps": 1024, "ent_coef": 0.02,  "clip_range": 0.15, "gae_lambda": 0.95},
    # Run 9 - Very low LR + very long rollout + high gae_lambda
    {"learning_rate": 5e-5,  "gamma": 0.99,  "n_steps": 4096, "ent_coef": 0.01,  "clip_range": 0.2,  "gae_lambda": 0.99},
    # Run 10 - Tuned: moderate LR + balanced rollout + small entropy
    {"learning_rate": 2e-4,  "gamma": 0.995, "n_steps": 2048, "ent_coef": 0.02,  "clip_range": 0.25, "gae_lambda": 0.98},
]


def run_ppo_experiments():
    results     = []
    callbacks   = []
    best_reward = -float("inf")

    print("=" * 65)
    print("   PPO Hyperparameter Experiments - Sista Health RL")
    print("=" * 65)

    for i, p in enumerate(PPO_EXPS):
        print(f"\n[PPO Run {i+1}/10]  LR={p['learning_rate']}  "
              f"gamma={p['gamma']}  n_steps={p['n_steps']}  "
              f"ent={p['ent_coef']}  clip={p['clip_range']}")

        env      = Monitor(SistaHealthEnv())
        callback = PGCallback()

        model = PPO(
            "MlpPolicy", env,
            learning_rate = p["learning_rate"],
            gamma         = p["gamma"],
            n_steps       = p["n_steps"],
            ent_coef      = p["ent_coef"],
            clip_range    = p["clip_range"],
            gae_lambda    = p["gae_lambda"],
            verbose       = 0,
        )

        model.learn(total_timesteps=TIMESTEPS, callback=callback)
        mean_r, std_r = evaluate_model(model)
        print(f"   -> Mean Reward: {mean_r:.2f} +/- {std_r:.2f}")

        if mean_r > best_reward:
            best_reward = mean_r
            model.save("models/pg/ppo/best_ppo_model")
            print("   -> Saved as best PPO model!")

        results.append({
            "Run":           i + 1,
            "Learning Rate": p["learning_rate"],
            "Gamma":         p["gamma"],
            "N Steps":       p["n_steps"],
            "Entropy Coef":  p["ent_coef"],
            "Clip Range":    p["clip_range"],
            "GAE Lambda":    p["gae_lambda"],
            "Mean Reward":   round(mean_r, 2),
            "Std Reward":    round(std_r, 2),
        })
        callbacks.append(callback)
        env.close()

    return results, callbacks


# ============================================================
# REINFORCE EXPERIMENTS
# A2C with vf_coef=0 approximates REINFORCE
# ============================================================

RF_EXPS = [
    # Run 1 - Baseline
    {"learning_rate": 7e-4,  "gamma": 0.99,  "n_steps": 20, "ent_coef": 0.01,  "vf_coef": 0.0,  "max_grad_norm": 0.5},
    # Run 2 - Low LR + long rollout + no entropy
    {"learning_rate": 1e-4,  "gamma": 0.99,  "n_steps": 50, "ent_coef": 0.0,   "vf_coef": 0.0,  "max_grad_norm": 0.5},
    # Run 3 - High LR + short rollout + high entropy (fast + noisy)
    {"learning_rate": 1e-3,  "gamma": 0.99,  "n_steps": 10, "ent_coef": 0.05,  "vf_coef": 0.0,  "max_grad_norm": 1.0},
    # Run 4 - Low gamma + long rollout (myopic)
    {"learning_rate": 7e-4,  "gamma": 0.85,  "n_steps": 50, "ent_coef": 0.01,  "vf_coef": 0.0,  "max_grad_norm": 0.5},
    # Run 5 - High gamma + short rollout + with baseline
    {"learning_rate": 5e-4,  "gamma": 0.995, "n_steps": 10, "ent_coef": 0.01,  "vf_coef": 0.25, "max_grad_norm": 0.5},
    # Run 6 - Medium LR + medium rollout + strong baseline
    {"learning_rate": 3e-4,  "gamma": 0.99,  "n_steps": 30, "ent_coef": 0.02,  "vf_coef": 0.5,  "max_grad_norm": 0.5},
    # Run 7 - High LR + long rollout + strong grad clip
    {"learning_rate": 1e-3,  "gamma": 0.97,  "n_steps": 50, "ent_coef": 0.01,  "vf_coef": 0.1,  "max_grad_norm": 0.3},
    # Run 8 - Low gamma + high entropy + no baseline (unstable)
    {"learning_rate": 7e-4,  "gamma": 0.90,  "n_steps": 20, "ent_coef": 0.05,  "vf_coef": 0.0,  "max_grad_norm": 1.0},
    # Run 9 - Very low LR + full episode + small baseline (careful)
    {"learning_rate": 5e-5,  "gamma": 0.99,  "n_steps": 50, "ent_coef": 0.005, "vf_coef": 0.1,  "max_grad_norm": 0.5},
    # Run 10 - Best combo: moderate LR + full episode + small entropy + baseline
    {"learning_rate": 5e-4,  "gamma": 0.995, "n_steps": 50, "ent_coef": 0.02,  "vf_coef": 0.1,  "max_grad_norm": 0.5},
]


def run_reinforce_experiments():
    results     = []
    callbacks   = []
    best_reward = -float("inf")

    print("\n" + "=" * 65)
    print("   REINFORCE Hyperparameter Experiments - Sista Health RL")
    print("=" * 65)

    for i, p in enumerate(RF_EXPS):
        print(f"\n[REINFORCE Run {i+1}/10]  LR={p['learning_rate']}  "
              f"gamma={p['gamma']}  n_steps={p['n_steps']}  "
              f"ent={p['ent_coef']}  vf={p['vf_coef']}")

        env      = Monitor(SistaHealthEnv())
        callback = PGCallback()

        model = A2C(
            "MlpPolicy", env,
            learning_rate  = p["learning_rate"],
            gamma          = p["gamma"],
            n_steps        = p["n_steps"],
            ent_coef       = p["ent_coef"],
            vf_coef        = p["vf_coef"],
            max_grad_norm  = p["max_grad_norm"],
            verbose        = 0,
        )

        model.learn(total_timesteps=TIMESTEPS, callback=callback)
        mean_r, std_r = evaluate_model(model)
        print(f"   -> Mean Reward: {mean_r:.2f} +/- {std_r:.2f}")

        if mean_r > best_reward:
            best_reward = mean_r
            model.save("models/pg/reinforce/best_reinforce_model")
            print("   -> Saved as best REINFORCE model!")

        results.append({
            "Run":           i + 1,
            "Learning Rate": p["learning_rate"],
            "Gamma":         p["gamma"],
            "N Steps":       p["n_steps"],
            "Entropy Coef":  p["ent_coef"],
            "VF Coef":       p["vf_coef"],
            "Max Grad Norm": p["max_grad_norm"],
            "Mean Reward":   round(mean_r, 2),
            "Std Reward":    round(std_r, 2),
        })
        callbacks.append(callback)
        env.close()

    return results, callbacks


# ============================================================
# PLOTTING
# ============================================================

def plot_ppo(results, callbacks):
    runs   = [r["Run"] for r in results]
    means  = [r["Mean Reward"] for r in results]
    stds   = [r["Std Reward"] for r in results]
    best_i = int(np.argmax(means))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("PPO Hyperparameter Comparison - Sista Health",
                 fontsize=14, fontweight="bold")

    bars = axes[0, 0].bar(runs, means, color="#25d366", alpha=0.85,
                           yerr=stds, capsize=4)
    bars[best_i].set_color("#58a6ff")
    axes[0, 0].set_title("Mean Reward per Run")
    axes[0, 0].set_xlabel("Run #")
    axes[0, 0].set_ylabel("Mean Episode Reward")

    lrs = [r["Learning Rate"] for r in results]
    axes[0, 1].scatter(lrs, means, color="#25d366", s=80)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Learning Rate vs Mean Reward")
    axes[0, 1].set_xlabel("Learning Rate (log)")

    ents = [r["Entropy Coef"] for r in results]
    axes[0, 2].scatter(ents, means, color="#e3b341", s=80)
    axes[0, 2].set_title("Entropy Coef vs Mean Reward")
    axes[0, 2].set_xlabel("Entropy Coefficient")

    n_steps = [r["N Steps"] for r in results]
    axes[1, 0].scatter(n_steps, means, color="#da3633", s=80)
    axes[1, 0].set_title("N Steps vs Mean Reward")
    axes[1, 0].set_xlabel("N Steps")

    # Entropy curves top 3
    sorted_idx = sorted(range(len(results)),
                        key=lambda i: results[i]["Mean Reward"], reverse=True)
    for idx in sorted_idx[:3]:
        cb = callbacks[idx]
        if cb.entropy_log:
            axes[1, 1].plot(cb.entropy_log,
                            label=f"Run {results[idx]['Run']}", linewidth=1.5)
    axes[1, 1].set_title("Entropy Curves (Top 3 Runs)")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Policy Entropy")
    axes[1, 1].legend(fontsize=8)

    best_cb = callbacks[best_i]
    rewards  = best_cb.episode_rewards
    if len(rewards) > 10:
        smoothed = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        axes[1, 2].plot(smoothed, color="#25d366", linewidth=2)
        axes[1, 2].fill_between(range(len(smoothed)), smoothed,
                                 alpha=0.2, color="#25d366")
    axes[1, 2].set_title(f"Best Run Training Curve (Run {best_i+1})")
    axes[1, 2].set_xlabel("Episode")
    axes[1, 2].set_ylabel("Episode Reward")
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/ppo_experiments.png", dpi=150, bbox_inches="tight")
    print("Saved: results/ppo_experiments.png")


def plot_reinforce(results, callbacks):
    runs   = [r["Run"] for r in results]
    means  = [r["Mean Reward"] for r in results]
    stds   = [r["Std Reward"] for r in results]
    best_i = int(np.argmax(means))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("REINFORCE Hyperparameter Comparison - Sista Health",
                 fontsize=14, fontweight="bold")

    bars = axes[0, 0].bar(runs, means, color="#e3b341", alpha=0.85,
                           yerr=stds, capsize=4)
    bars[best_i].set_color("#58a6ff")
    axes[0, 0].set_title("Mean Reward per Run")
    axes[0, 0].set_xlabel("Run #")
    axes[0, 0].set_ylabel("Mean Episode Reward")

    lrs = [r["Learning Rate"] for r in results]
    axes[0, 1].scatter(lrs, means, color="#e3b341", s=80)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Learning Rate vs Mean Reward")
    axes[0, 1].set_xlabel("Learning Rate (log)")

    vf_coefs = [r["VF Coef"] for r in results]
    axes[0, 2].scatter(vf_coefs, means, color="#8957e5", s=80)
    axes[0, 2].set_title("VF Coef vs Mean Reward")
    axes[0, 2].set_xlabel("Value Function Coefficient")

    n_steps = [r["N Steps"] for r in results]
    axes[1, 0].scatter(n_steps, means, color="#da3633", s=80)
    axes[1, 0].set_title("N Steps vs Mean Reward")
    axes[1, 0].set_xlabel("N Steps")

    stds_list = [r["Std Reward"] for r in results]
    axes[1, 1].bar(runs, stds_list, color="salmon", alpha=0.8)
    axes[1, 1].set_title("Reward Std Dev (Stability)")
    axes[1, 1].set_xlabel("Run #")
    axes[1, 1].set_ylabel("Std Deviation")

    best_cb = callbacks[best_i]
    rewards  = best_cb.episode_rewards
    if len(rewards) > 10:
        smoothed = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        axes[1, 2].plot(smoothed, color="#e3b341", linewidth=2)
        axes[1, 2].fill_between(range(len(smoothed)), smoothed,
                                 alpha=0.2, color="#e3b341")
    axes[1, 2].set_title(f"Best Run Training Curve (Run {best_i+1})")
    axes[1, 2].set_xlabel("Episode")
    axes[1, 2].set_ylabel("Episode Reward")
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/reinforce_experiments.png", dpi=150, bbox_inches="tight")
    print("Saved: results/reinforce_experiments.png")


def save_tables(ppo_results, rf_results):
    ppo_df = pd.DataFrame(ppo_results)
    rf_df  = pd.DataFrame(rf_results)
    ppo_df.to_csv("results/ppo_results.csv", index=False)
    rf_df.to_csv("results/reinforce_results.csv", index=False)
    print("\n-- PPO Results --")
    print(ppo_df.to_string(index=False))
    print("\n-- REINFORCE Results --")
    print(rf_df.to_string(index=False))
    return ppo_df, rf_df


if __name__ == "__main__":
    ppo_results, ppo_cbs = run_ppo_experiments()
    rf_results,  rf_cbs  = run_reinforce_experiments()
    plot_ppo(ppo_results, ppo_cbs)
    plot_reinforce(rf_results, rf_cbs)
    save_tables(ppo_results, rf_results)

    best_ppo = max(ppo_results, key=lambda r: r["Mean Reward"])
    best_rf  = max(rf_results,  key=lambda r: r["Mean Reward"])
    print(f"\nBest PPO:       Run #{best_ppo['Run']} - {best_ppo['Mean Reward']}")
    print(f"Best REINFORCE: Run #{best_rf['Run']}  - {best_rf['Mean Reward']}")
