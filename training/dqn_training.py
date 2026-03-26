"""
dqn_training.py - Sista Health RL
====================================
Trains DQN using Stable Baselines 3.
Runs 10 hyperparameter experiments automatically.
Saves best model, reward curves, and results table.

Usage:
    python training/dqn_training.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import SistaHealthEnv

os.makedirs("models/dqn", exist_ok=True)
os.makedirs("results", exist_ok=True)

TIMESTEPS = 100_000


class DQNCallback(BaseCallback):
    def __init__(self):
        super().__init__(0)
        self.episode_rewards = []
        self.current_rewards = []
        self.loss_log        = []
        self.step_log        = []

    def _on_step(self):
        self.current_rewards.append(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episode_rewards.append(sum(self.current_rewards))
            self.current_rewards = []
        try:
            loss = self.model.logger.name_to_value.get("train/loss", None)
            if loss is not None:
                self.loss_log.append(loss)
                self.step_log.append(self.num_timesteps)
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


DQN_EXPS = [
    # Run 1 - Baseline
    {"learning_rate": 1e-3,  "gamma": 0.99,  "batch_size": 64,  "buffer_size": 50000,  "exploration_fraction": 0.3,  "exploration_final_eps": 0.05},
    # Run 2 - Low LR + large buffer + tight eps
    {"learning_rate": 1e-4,  "gamma": 0.99,  "batch_size": 64,  "buffer_size": 100000, "exploration_fraction": 0.3,  "exploration_final_eps": 0.01},
    # Run 3 - High LR + large batch + less exploration
    {"learning_rate": 5e-3,  "gamma": 0.99,  "batch_size": 256, "buffer_size": 50000,  "exploration_fraction": 0.15, "exploration_final_eps": 0.02},
    # Run 4 - Low gamma + small batch + high exploration
    {"learning_rate": 1e-3,  "gamma": 0.85,  "batch_size": 32,  "buffer_size": 50000,  "exploration_fraction": 0.5,  "exploration_final_eps": 0.1},
    # Run 5 - High gamma + large batch + large buffer
    {"learning_rate": 5e-4,  "gamma": 0.995, "batch_size": 256, "buffer_size": 100000, "exploration_fraction": 0.2,  "exploration_final_eps": 0.02},
    # Run 6 - Medium LR + large buffer + low final eps
    {"learning_rate": 1e-3,  "gamma": 0.99,  "batch_size": 64,  "buffer_size": 100000, "exploration_fraction": 0.3,  "exploration_final_eps": 0.01},
    # Run 7 - High LR + low gamma + small batch
    {"learning_rate": 2e-3,  "gamma": 0.97,  "batch_size": 32,  "buffer_size": 50000,  "exploration_fraction": 0.1,  "exploration_final_eps": 0.01},
    # Run 8 - Max exploration + low gamma + medium batch
    {"learning_rate": 5e-4,  "gamma": 0.90,  "batch_size": 128, "buffer_size": 50000,  "exploration_fraction": 0.6,  "exploration_final_eps": 0.1},
    # Run 9 - Balanced LR + large buffer + medium batch
    {"learning_rate": 5e-4,  "gamma": 0.99,  "batch_size": 128, "buffer_size": 100000, "exploration_fraction": 0.25, "exploration_final_eps": 0.02},
    # Run 10 - Conservative: very low LR + large batch + high gamma
    {"learning_rate": 2e-4,  "gamma": 0.995, "batch_size": 256, "buffer_size": 100000, "exploration_fraction": 0.2,  "exploration_final_eps": 0.02},
]


def run_experiments():
    results     = []
    callbacks   = []
    best_reward = -float("inf")

    print("=" * 65)
    print("   DQN Hyperparameter Experiments - Sista Health RL")
    print("=" * 65)

    for i, p in enumerate(DQN_EXPS):
        print(f"\n[DQN Run {i+1}/10]  LR={p['learning_rate']}  "
              f"gamma={p['gamma']}  batch={p['batch_size']}  "
              f"buffer={p['buffer_size']}  explore={p['exploration_fraction']}")

        env      = Monitor(SistaHealthEnv())
        callback = DQNCallback()

        model = DQN(
            "MlpPolicy", env,
            learning_rate          = p["learning_rate"],
            gamma                  = p["gamma"],
            batch_size             = p["batch_size"],
            buffer_size            = p["buffer_size"],
            exploration_fraction   = p["exploration_fraction"],
            exploration_final_eps  = p["exploration_final_eps"],
            learning_starts        = 1000,
            target_update_interval = 500,
            verbose                = 0,
        )

        model.learn(total_timesteps=TIMESTEPS, callback=callback)
        mean_r, std_r = evaluate_model(model)
        print(f"   -> Mean Reward: {mean_r:.2f} +/- {std_r:.2f}")

        if mean_r > best_reward:
            best_reward = mean_r
            model.save("models/dqn/best_dqn_model")
            print("   -> Saved as best DQN model!")

        results.append({
            "Run":               i + 1,
            "Learning Rate":     p["learning_rate"],
            "Gamma":             p["gamma"],
            "Batch Size":        p["batch_size"],
            "Buffer Size":       p["buffer_size"],
            "Explore Fraction":  p["exploration_fraction"],
            "Final Eps":         p["exploration_final_eps"],
            "Mean Reward":       round(mean_r, 2),
            "Std Reward":        round(std_r, 2),
        })
        callbacks.append(callback)
        env.close()

    return results, callbacks


def plot_results(results, callbacks):
    runs   = [r["Run"] for r in results]
    means  = [r["Mean Reward"] for r in results]
    stds   = [r["Std Reward"] for r in results]
    best_i = int(np.argmax(means))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("DQN Hyperparameter Comparison - Sista Health",
                 fontsize=14, fontweight="bold")

    bars = axes[0, 0].bar(runs, means, color="#58a6ff", alpha=0.85,
                           yerr=stds, capsize=4)
    bars[best_i].set_color("#25d366")
    axes[0, 0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 0].set_title("Mean Reward per Run")
    axes[0, 0].set_xlabel("Run #")
    axes[0, 0].set_ylabel("Mean Episode Reward")

    lrs = [r["Learning Rate"] for r in results]
    axes[0, 1].scatter(lrs, means, color="#58a6ff", s=80, zorder=5)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Learning Rate vs Mean Reward")
    axes[0, 1].set_xlabel("Learning Rate (log)")
    axes[0, 1].set_ylabel("Mean Reward")

    gammas = [r["Gamma"] for r in results]
    axes[0, 2].scatter(gammas, means, color="#e3b341", s=80)
    axes[0, 2].set_title("Gamma vs Mean Reward")
    axes[0, 2].set_xlabel("Gamma")
    axes[0, 2].set_ylabel("Mean Reward")

    buffers = [r["Buffer Size"] for r in results]
    axes[1, 0].scatter(buffers, means, color="#da3633", s=80)
    axes[1, 0].set_title("Buffer Size vs Mean Reward")
    axes[1, 0].set_xlabel("Buffer Size")
    axes[1, 0].set_ylabel("Mean Reward")

    explores = [r["Explore Fraction"] for r in results]
    axes[1, 1].scatter(explores, means, color="#8957e5", s=80)
    axes[1, 1].set_title("Exploration Fraction vs Mean Reward")
    axes[1, 1].set_xlabel("Exploration Fraction")
    axes[1, 1].set_ylabel("Mean Reward")

    best_cb = callbacks[best_i]
    rewards  = best_cb.episode_rewards
    if len(rewards) > 10:
        smoothed = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        axes[1, 2].plot(smoothed, color="#58a6ff", linewidth=2)
        axes[1, 2].fill_between(range(len(smoothed)), smoothed,
                                 alpha=0.2, color="#58a6ff")
    axes[1, 2].set_title(f"Best Run Training Curve (Run {best_i+1})")
    axes[1, 2].set_xlabel("Episode")
    axes[1, 2].set_ylabel("Episode Reward")
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/dqn_experiments.png", dpi=150, bbox_inches="tight")
    print("Saved: results/dqn_experiments.png")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("DQN Objective Curves - Best Run", fontweight="bold")

    if best_cb.loss_log:
        axes2[0].plot(best_cb.step_log, best_cb.loss_log,
                      color="#58a6ff", linewidth=1.5, alpha=0.8)
        axes2[0].set_title("DQN Training Loss")
        axes2[0].set_xlabel("Timestep")
        axes2[0].set_ylabel("Loss")
        axes2[0].grid(alpha=0.3)

    if len(rewards) > 10:
        smoothed = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        axes2[1].plot(smoothed, color="#25d366", linewidth=2)
        axes2[1].fill_between(range(len(smoothed)), smoothed,
                               alpha=0.2, color="#25d366")
    axes2[1].set_title("DQN Smoothed Reward Curve")
    axes2[1].set_xlabel("Episode")
    axes2[1].set_ylabel("Reward")
    axes2[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/dqn_reward_curve.png", dpi=150, bbox_inches="tight")
    print("Saved: results/dqn_reward_curve.png")


def save_table(results):
    df = pd.DataFrame(results)
    df.to_csv("results/dqn_results.csv", index=False)
    print("\n" + "=" * 80)
    print("DQN RESULTS TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    best = df.loc[df["Mean Reward"].idxmax()]
    print(f"\nBest Run: #{int(best['Run'])} - Mean Reward: {best['Mean Reward']}")
    return df


if __name__ == "__main__":
    results, callbacks = run_experiments()
    plot_results(results, callbacks)
    save_table(results)
