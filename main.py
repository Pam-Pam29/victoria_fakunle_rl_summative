"""
main.py
-------
Entry point for the Sista Health RL project.

Loads the best saved model (DQN, PPO, or REINFORCE — whichever scored
highest during training) and runs a demonstration evaluation.

Usage:
    python main.py                        # auto-selects best algorithm
    python main.py --algo dqn             # force DQN
    python main.py --algo ppo             # force PPO
    python main.py --algo reinforce       # force REINFORCE
    python main.py --episodes 50          # number of eval episodes (default 50)
    python main.py --render               # print step-by-step actions
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from environment.custom_env import SistaHealthEnv


# ── Model paths ───────────────────────────────────────────────────────────────

MODEL_PATHS = {
    "dqn":       os.path.join(ROOT, "models", "dqn",            "best_dqn_model"),
    "ppo":       os.path.join(ROOT, "models", "pg", "ppo",      "best_ppo_model"),
    "reinforce": os.path.join(ROOT, "models", "pg", "reinforce","best_reinforce_model"),
}

BEST_RUN_FILES = {
    "dqn":       os.path.join(ROOT, "models", "dqn",            "best_run.txt"),
    "ppo":       os.path.join(ROOT, "models", "pg", "ppo",      "best_run.txt"),
    "reinforce": os.path.join(ROOT, "models", "pg", "reinforce","best_run.txt"),
}


def _load_model(algo: str):
    """Load a saved SB3 model for the given algorithm name."""
    path = MODEL_PATHS[algo]
    if not os.path.exists(path + ".zip"):
        raise FileNotFoundError(
            f"No saved model found at {path}.zip\n"
            f"Run training first:  python training/dqn_training.py  OR  "
            f"python training/pg_training.py"
        )

    if algo == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(path)
    elif algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(path)
    elif algo == "reinforce":
        from stable_baselines3 import A2C
        return A2C.load(path)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def _auto_select_algo() -> str:
    """
    Pick the algorithm whose best_run.txt exists and whose model file
    exists. If multiple are found, fall back to a simple priority order.
    """
    priority = ["ppo", "dqn", "reinforce"]
    for algo in priority:
        if os.path.exists(MODEL_PATHS[algo] + ".zip"):
            return algo
    raise FileNotFoundError(
        "No trained models found. Please run the training scripts first:\n"
        "  python training/dqn_training.py\n"
        "  python training/pg_training.py"
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_demo(model, algo: str, n_episodes: int = 50, render: bool = False):
    env = SistaHealthEnv(render_mode="human" if render else None)
    all_rewards       = []
    scenario_rewards  = defaultdict(list)
    action_counts     = defaultdict(lambda: defaultdict(int))

    print("=" * 60)
    print(f"  Sista Health RL — Demo Evaluation")
    print(f"  Algorithm : {algo.upper()}")
    print(f"  Episodes  : {n_episodes}")
    print("=" * 60)

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0
        done      = False
        step      = 0

        lang   = info["language"]
        domain = info["domain"]
        lit    = info["literacy"]

        if ep < 5:
            print(f"\n{'─'*55}")
            print(f"  EPISODE {ep+1} | {lang} | {domain} | Literacy: {lit}")
            print(f"{'─'*55}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            ep_reward += reward
            done       = term or trunc

            if ep < 5:
                sign = "✅" if reward > 0 else ("⚠️" if reward == 0 else "❌")
                print(
                    f"  Step {step+1:2d} | {env.ACTIONS[int(action)]:20s} | "
                    f"{reward:+.0f} {sign}"
                )
            action_counts[f"{lang}|{lit}"][env.ACTIONS[int(action)]] += 1
            step += 1

        if ep < 5:
            print(f"  Episode Reward: {ep_reward:.1f}")

        all_rewards.append(ep_reward)
        scenario_rewards[f"{lang} | {domain} | {lit}"].append(ep_reward)

    env.close()

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY — {algo.upper()}")
    print(f"  Episodes evaluated : {n_episodes}")
    print(f"  Mean Reward        : {np.mean(all_rewards):.2f}")
    print(f"  Std  Reward        : {np.std(all_rewards):.2f}")
    print(f"  Min / Max          : {np.min(all_rewards):.1f} / {np.max(all_rewards):.1f}")
    print(f"{'='*60}")

    print(f"\n{'='*60}")
    print("  BREAKDOWN BY SCENARIO")
    print(f"{'─'*60}")
    for scenario, rewards in sorted(scenario_rewards.items()):
        print(f"  {scenario:<42}  mean={np.mean(rewards):6.1f}  n={len(rewards)}")

    print(f"\n{'='*60}")
    print("  DOMINANT ACTION BY USER PROFILE")
    print(f"{'─'*60}")
    for profile, actions in sorted(action_counts.items()):
        total    = sum(actions.values())
        dominant = max(actions, key=actions.get)
        print(f"  {profile:<25}  → {dominant} ({actions[dominant]/total*100:.0f}%)")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run best Sista Health RL model demo."
    )
    parser.add_argument(
        "--algo",
        choices=["dqn", "ppo", "reinforce", "auto"],
        default="auto",
        help="Algorithm to load (default: auto — picks whichever model file exists).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes (default: 50).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Print step-by-step actions for every episode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    algo = args.algo
    if algo == "auto":
        algo = _auto_select_algo()
        print(f"Auto-selected algorithm: {algo.upper()}")

    print(f"Loading model: {MODEL_PATHS[algo]}.zip …")
    model = _load_model(algo)
    print("Model loaded.\n")

    run_demo(model, algo, n_episodes=args.episodes, render=args.render)


if __name__ == "__main__":
    main()
