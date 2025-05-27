"""
DQN agent for CartPole by tihassfjord
Complete Deep Q-Network implementation for reinforcement learning
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
from collections import deque, namedtuple
from pathlib import Path
import json

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent for CartPole"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, 
                 epsilon_decay=0.995, buffer_size=10000, batch_size=64,
                 target_update=100):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.losses = []
        self.rewards = []
        self.epsilon_history = []
        
        # Create directories
        Path("models").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
    
    def act(self, state, epsilon=None):
        """Choose action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.epsilon
            
        if random.random() > epsilon:
            # Exploit: choose best action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
        else:
            # Explore: choose random action
            return random.choice(range(self.action_size))
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        self.losses.append(loss.item())
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filename="dqn_cartpole_tihassfjord.pth"):
        """Save trained model and training history"""
        model_path = f"models/{filename}"
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses,
            'rewards': self.rewards,
        }, model_path)
        
        # Save training metadata
        metadata = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'target_update': self.target_update
        }
        
        with open("models/model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved: {model_path}")
    
    def load_model(self, filename="dqn_cartpole_tihassfjord.pth"):
        """Load pre-trained model"""
        model_path = f"models/{filename}"
        
        try:
            checkpoint = torch.load(model_path)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
            self.losses = checkpoint.get('losses', [])
            self.rewards = checkpoint.get('rewards', [])
            
            print(f"Model loaded: {model_path}")
            return True
        except FileNotFoundError:
            print(f"Model file not found: {model_path}")
            return False

class CartPoleTrainer:
    """Training environment for CartPole DQN"""
    
    def __init__(self, render=False):
        self.env = gym.make('CartPole-v1')
        self.render = render
        
        # Environment info
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        print(f"State size: {self.state_size}")
        print(f"Action size: {self.action_size}")
        
        # Create agent
        self.agent = DQNAgent(self.state_size, self.action_size)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.solved = False
    
    def train(self, episodes=500, target_score=195, target_episodes=100):
        """Train the DQN agent"""
        print("Training DQN agent on CartPole (tihassfjord).")
        
        recent_rewards = deque(maxlen=target_episodes)
        
        for episode in range(episodes):
            state = self.env.reset()
            if isinstance(state, tuple):  # Handle new gym API
                state = state[0]
            
            total_reward = 0
            steps = 0
            
            while True:
                if self.render:
                    self.env.render()
                
                # Choose action
                action = self.agent.act(state)
                
                # Take action
                next_state, reward, done, *_ = self.env.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                self.agent.replay()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Update target network
            if episode % self.agent.target_update == 0:
                self.agent.update_target_network()
            
            # Store metrics
            self.agent.rewards.append(total_reward)
            self.agent.epsilon_history.append(self.agent.epsilon)
            recent_rewards.append(total_reward)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(recent_rewards)
                print(f"Episode {episode}, Reward: {total_reward}, "
                      f"Avg: {avg_reward:.1f}, Epsilon: {self.agent.epsilon:.2f}")
            
            # Check if solved
            if len(recent_rewards) >= target_episodes:
                avg_reward = np.mean(recent_rewards)
                if avg_reward >= target_score and not self.solved:
                    print(f"Environment solved! Average reward: {avg_reward:.1f}")
                    self.solved = True
                    self.agent.save_model()
                    break
        
        self.env.close()
        print("Training complete!")
        
        # Plot results
        self.plot_training_results()
        
        return self.agent
    
    def test(self, episodes=10, model_file=None):
        """Test the trained agent"""
        print("Testing DQN agent...")
        
        if model_file:
            self.agent.load_model(model_file)
        
        test_rewards = []
        
        for episode in range(episodes):
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            total_reward = 0
            steps = 0
            
            while True:
                if self.render:
                    self.env.render()
                
                # Choose best action (no exploration)
                action = self.agent.act(state, epsilon=0.0)
                state, reward, done, *_ = self.env.step(action)
                
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            test_rewards.append(total_reward)
            print(f"Test Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")
        
        self.env.close()
        
        avg_reward = np.mean(test_rewards)
        print(f"\nTest Results:")
        print(f"Average reward: {avg_reward:.1f}")
        print(f"Best reward: {max(test_rewards)}")
        print(f"Worst reward: {min(test_rewards)}")
        
        return test_rewards
    
    def plot_training_results(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Training Results (tihassfjord)', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(self.agent.rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Moving average of rewards
        if len(self.agent.rewards) > 10:
            window = min(50, len(self.agent.rewards) // 10)
            moving_avg = np.convolve(self.agent.rewards, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].grid(True)
        
        # Epsilon decay
        axes[1, 0].plot(self.agent.epsilon_history)
        axes[1, 0].set_title('Epsilon Decay')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True)
        
        # Training loss
        if self.agent.losses:
            axes[1, 1].plot(self.agent.losses)
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('logs/training_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='DQN for CartPole by tihassfjord')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--test', action='store_true', help='Test trained agent')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--model', type=str, help='Model file to load for testing')
    parser.add_argument('--test-episodes', type=int, default=10, help='Number of test episodes')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CartPoleTrainer(render=args.render)
    
    if args.test:
        # Test mode
        trainer.test(episodes=args.test_episodes, model_file=args.model)
    else:
        # Training mode
        agent = trainer.train(episodes=args.episodes)
        
        # Test the trained agent
        print("\nTesting trained agent...")
        trainer.test(episodes=5)
    
    print("🎉 Reinforcement learning complete! (tihassfjord)")

if __name__ == "__main__":
    main()
