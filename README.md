## Sista Health — Mission-Based Reinforcement Learning

**Victoria Fakunle**

---

## Project Overview

This project trains Reinforcement Learning agents to optimise response strategies for **Sista Health**, which is a voice-enabled multilingual sexual and reproductive health assistant for Nigerian women delivered via WhatsApp.

The RL agent simulates the Sista Health response system. Given a user profile (language, literacy level, health topic, urgency), the agent learns the optimal response modality to maximise health comprehension and care-seeking behaviour.

---

## The Problem

Nigeria has one of the highest maternal mortality rates in the world(1,047 deaths per 100,000 live births). A core cause is the information gap: health information is delivered in English to a population where only 30% speak it fluently. Sista Health addresses this by delivering maternal and sexual health information in English, Yoruba, and Nigerian Pidgin via WhatsApp voice notes and text.

This RL project simulates the response optimisation problem: given a user's profile, what is the best way to respond?

---

## Environment Design

### Observation Space (6 features)

| Feature | Values | Description |
|---|---|---|
| language | 0=English, 1=Yoruba, 2=Pidgin | User's language |
| domain | 0=Sexual Health, 1=Maternal Health | Health domain |
| topic | 0-8 | One of 9 health topics |
| urgency | 0=Normal, 1=Emergency | Query urgency level |
| literacy | 0=Low, 1=Medium, 2=High | User's literacy level |
| session_step | 0-9 | Current step in session |

### Health Topics (9)
FGM Complications, VVF Causes, Cultural Barriers, Early Marriage, TBA Dangers, Contraception, STIs and HIV, Antenatal Care, Postpartum Care

### Action Space (4 discrete actions)

| Action | Description |
|---|---|
| 0 — Text Response | Send written text response |
| 1 — Voice Note | Send audio voice note response |
| 2 — Emergency Referral | Immediately escalate to crisis services |
| 3 — Clarify | Ask a clarifying question |

### Reward Structure

| Scenario | Reward |
|---|---|
| Voice note for low literacy user | +10 |
| Voice note for medium literacy user | +5 |
| Pidgin user voice note bonus | +1 |
| Text for high literacy user | +5 |
| Text for medium literacy user | +2 |
| Emergency correctly escalated | +10 |
| Emergency missed (wrong action taken) | -10 |
| Text for low literacy user | -2 |
| Unnecessary emergency referral | -3 |
| Clarify during emergency | -2 |
| Clarify during normal session | +1 |

### Terminal Conditions
- Session reaches 10 steps
- Emergency referral triggered (immediate termination)

### Baseline
A random agent with no training scored **0.00 mean reward**, confirming the environment requires learning to achieve positive outcomes.

---

## Algorithms

Three RL algorithms were implemented and compared using Stable Baselines 3:

| Algorithm | Type | Description |
|---|---|---|
| DQN | Value-Based | Learns Q-values using experience replay and target networks |
| PPO | Policy Gradient | Uses clipped surrogate objective for stable policy updates |
| REINFORCE | Policy Gradient | Monte Carlo policy gradient with no value function (A2C with vf_coef=0) |

All models were trained for **50,000 timesteps** across **10 hyperparameter experiments each** (30 total runs).

---

## Results

### Final Performance (100 evaluation episodes)

| Algorithm | Mean Reward | Std Deviation | Best Run |
|---|---|---|---|
| Random (baseline) | 0.00 | — | — |
| REINFORCE | 46.10 | 37.40 | 59.00 (Run 4) |
| DQN | 55.90 | 32.50 | 66.95 (Run 1) |
| PPO | 56.40 | 31.40 | 67.00 (Run 3) |

**PPO achieved the highest mean reward (56.4) with the lowest standard deviation (31.4), making it the most stable and best-performing algorithm.**

---

## DQN Hyperparameter Experiments

| Run | Learning Rate | Gamma | Batch Size | Buffer Size | Exploration | Mean Reward | Std |
|---|---|---|---|---|---|---|---|
| 1 | 0.0010 | 0.99 | 32 | 10000 | 0.3 | **66.95** | 38.15 |
| 2 | 0.0005 | 0.99 | 32 | 10000 | 0.3 | 59.05 | 30.36 |
| 3 | 0.0001 | 0.99 | 32 | 10000 | 0.3 | 54.50 | 24.99 |
| 4 | 0.0010 | 0.95 | 32 | 10000 | 0.3 | 59.25 | 26.17 |
| 5 | 0.0010 | 0.90 | 32 | 10000 | 0.3 | 49.55 | 32.26 |
| 6 | 0.0010 | 0.99 | 64 | 10000 | 0.3 | 58.10 | 31.06 |
| 7 | 0.0010 | 0.99 | 128 | 10000 | 0.3 | 63.45 | 32.82 |
| 8 | 0.0010 | 0.99 | 32 | 50000 | 0.3 | 55.40 | 35.24 |
| 9 | 0.0010 | 0.99 | 32 | 10000 | 0.1 | 58.10 | 29.27 |
| 10 | 0.0010 | 0.99 | 32 | 10000 | 0.5 | 58.90 | 32.97 |

**Key findings:** Run 1 (lr=0.001, gamma=0.99, batch=32) performed best. Lower gamma (0.90) hurts performance and is consistent with short 10-step episodes where the agent needs to value future rewards highly. Larger batch size (Run 7, batch=128) performed second best.

---

## PPO Hyperparameter Experiments

| Run | Learning Rate | Gamma | N Steps | Entropy Coef | Clip Range | Mean Reward | Std |
|---|---|---|---|---|---|---|---|
| 1 | 0.0003 | 0.99 | 512 | 0.01 | 0.2 | 64.35 | 38.59 |
| 2 | 0.0001 | 0.99 | 512 | 0.01 | 0.2 | 46.00 | 36.52 |
| 3 | 0.0010 | 0.99 | 512 | 0.01 | 0.2 | **67.00** | 33.63 |
| 4 | 0.0003 | 0.95 | 512 | 0.01 | 0.2 | 59.50 | 36.81 |
| 5 | 0.0003 | 0.90 | 512 | 0.01 | 0.2 | 57.35 | 28.66 |
| 6 | 0.0003 | 0.99 | 1024 | 0.01 | 0.2 | 61.50 | 34.39 |
| 7 | 0.0003 | 0.99 | 256 | 0.01 | 0.2 | 59.00 | 33.30 |
| 8 | 0.0003 | 0.99 | 512 | 0.05 | 0.2 | 61.45 | 37.21 |
| 9 | 0.0003 | 0.99 | 512 | 0.00 | 0.2 | 50.90 | 30.35 |
| 10 | 0.0003 | 0.99 | 512 | 0.01 | 0.3 | 43.45 | 42.49 |

**Key findings:** Run 3 (lr=0.001) was best; a higher learning rate worked well for PPO here. Larger clip range (Run 10, clip=0.3) scored lowest and had the highest variance, confirming that tighter clipping improves stability. Removing entropy (Run 9) reduced performance, indicating exploration is still needed.

---

## REINFORCE Hyperparameter Experiments

| Run | Learning Rate | Gamma | N Steps | Entropy Coef | VF Coef | Mean Reward | Std |
|---|---|---|---|---|---|---|---|
| 1 | 0.0007 | 0.99 | 10 | 0.01 | 0.00 | 42.15 | 36.15 |
| 2 | 0.0010 | 0.99 | 10 | 0.01 | 0.00 | 46.70 | 44.25 |
| 3 | 0.0005 | 0.99 | 10 | 0.01 | 0.00 | 49.50 | 36.26 |
| 4 | 0.0007 | 0.95 | 10 | 0.01 | 0.00 | **59.00** | 34.91 |
| 5 | 0.0007 | 0.90 | 10 | 0.01 | 0.00 | 30.10 | 34.29 |
| 6 | 0.0007 | 0.99 | 20 | 0.01 | 0.00 | 39.05 | 25.02 |
| 7 | 0.0007 | 0.99 | 5 | 0.01 | 0.00 | 53.10 | 39.91 |
| 8 | 0.0007 | 0.99 | 10 | 0.05 | 0.00 | 41.05 | 36.14 |
| 9 | 0.0007 | 0.99 | 10 | 0.00 | 0.00 | 48.10 | 35.05 |
| 10 | 0.0007 | 0.99 | 10 | 0.01 | 0.25 | 56.15 | 31.79 |

**Key findings:** Run 4 (gamma=0.95) was best, unexpectedly, a slightly lower gamma outperformed 0.99 for REINFORCE. Run 10 (vf_coef=0.25) showed that adding a small value function component improves stability, confirming the theoretical advantage of actor-critic over pure policy gradient. Run 5 (gamma=0.90) scored lowest across all algorithms.

---

## Agent Behaviour

The best-performing agent (PPO) demonstrated three distinct learned policies:

- **Low literacy users** — agent selects Voice Note consistently, scoring up to +10 per step
- **High literacy users** — agent selects Text Response, scoring +5 per step
- **Emergency queries** — agent immediately triggers Emergency Referral in Step 1, terminating the episode with +10

This behaviour directly mirrors the real-world design of Sista Health: voice notes for users who cannot read, text for literate users, and immediate escalation for medical emergencies.

---

## Generalisation Test Results

Agents were tested on 6 fixed user profiles:

| Profile | DQN | PPO | REINFORCE |
|---|---|---|---|
| Low Literacy Pidgin Emergency | -3 | +10 | +10 |
| High Literacy English Normal | +50 | +50 | +10 |
| Low Literacy Yoruba Normal | +100 | +100 | +100 |
| Medium Literacy Pidgin Normal | +60 | +48 | +60 |
| High Literacy English Emergency | +10 | +10 | +10 |
| Low Literacy Yoruba Emergency | +10 | +10 | +10 |

All three algorithms correctly handle emergencies. DQN showed one failure (Low Literacy Pidgin Emergency scored -3), indicating occasional incorrect action selection for edge cases.

---

## Project Structure

```
fakunle_victoria_rl_summative/
├── environment/
│   ├── custom_env.py          # Custom Gymnasium environment
│   └── rendering.py           # Pygame visualization
├── training/
│   ├── dqn_training.py        # DQN training + 10 experiments
│   └── pg_training.py         # PPO + REINFORCE training + experiments
├── models/
│   ├── dqn/                   # Saved DQN model
│   └── pg/                    # Saved PPO + REINFORCE models
├── notebooks/
│   ├── fakunle_victoria_rl_dqn.ipynb
│   ├── fakunle_victoria_rl_ppo.ipynb
│   ├── fakunle_victoria_rl_reinforce.ipynb
│   └── fakunle_victoria_rl_comparison.ipynb
├── results/                   # Graphs and CSV tables
├── main.py                    # Run best model with visualisation
├── requirements.txt
└── README.md
```

---

## Setup and Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/fakunle_victoria_rl_summative
cd fakunle_victoria_rl_summative

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run best agent (PPO)
python main.py --algo ppo --episodes 5

# 4. Run without GUI (terminal only)
python main.py --algo ppo --no-gui

# 5. Run DQN or REINFORCE
python main.py --algo dqn --episodes 5
python main.py --algo reinforce --episodes 5
```

---

## Requirements

```
gymnasium==0.29.1
stable-baselines3==2.3.2
pygame==2.5.2
numpy==1.26.4
pandas==2.2.1
matplotlib==3.8.3
torch>=2.0.0
seaborn>=0.13.0
```

---

## Key Findings Summary

1. **PPO is the best overall algorithm** — highest mean reward (56.4) and lowest variance (31.4)
2. **DQN is close second** — mean reward 55.9, occasionally unstable on edge cases
3. **REINFORCE is least stable** — mean reward 46.1, highest variance (37.4), consistent with its Monte Carlo nature and lack of replay buffer
4. **Gamma=0.90 consistently hurts all algorithms** — short 10-step episodes need high gamma
5. **All algorithms significantly outperform the random baseline** — random scored 0, trained agents scored 46-56 mean reward
6. **Voice note is the dominant optimal action** — correct for the majority of Nigerian users who have low to medium literacy

---

## Mission Alignment

This project directly supports the Sista Health mission of improving maternal health outcomes for Nigerian women. The RL agent learns that:
- Voice notes maximise comprehension for low-literacy users (aligned with research showing 3.2x higher comprehension for Pidgin voice delivery)
- Emergency escalation must be immediate (aligned with the real-world need for rapid referral to prevent maternal deaths)
- Language-specific bonuses reflect the documented effectiveness of mother-tongue health communication

---

*Victoria Fakunle | African Leadership University | 2026*
