# Reinforcement Learning Game AI (highlight) — tihassfjord

## Goal
Train a Deep Q-Network (DQN) agent to solve the CartPole-v1 environment using deep reinforcement learning.

## Dataset
- OpenAI Gym CartPole-v1 environment (simulated)
- Experience replay buffer for training

## Requirements
- Python 3.8+
- gym
- torch
- numpy
- matplotlib

## How to Run
```bash
# Train DQN agent
python rl_cartpole_tihassfjord.py

# Test trained agent
python rl_cartpole_tihassfjord.py --test

# Train with visualization
python rl_cartpole_tihassfjord.py --render
```

## Example Output
```
Training DQN agent on CartPole (tihassfjord).
Episode 10, Reward: 19, Epsilon: 0.95
Episode 50, Reward: 45, Epsilon: 0.75
Episode 100, Reward: 134, Epsilon: 0.55
Episode 200, Reward: 200, Epsilon: 0.15
Environment solved! Average reward: 195.0
```

## Project Structure
```
rl-game-ai-tihassfjord/
│
├── rl_cartpole_tihassfjord.py        # Main DQN implementation
├── models/                           # Saved models
├── logs/                            # Training logs
├── requirements.txt                 # Dependencies
└── README.md                       # This file
```

## Key Features
- Deep Q-Network (DQN) implementation
- Experience replay buffer
- Epsilon-greedy exploration
- Target network for stability
- Training visualization
- Model persistence
- Performance evaluation

## Learning Outcomes
- Reinforcement learning fundamentals
- Q-learning and value functions
- Deep Q-Networks (DQN)
- Experience replay mechanisms
- Exploration vs exploitation
- RL training strategies

---
*Project by tihassfjord - Advanced ML Portfolio*
