"""Reinforcement Learning agent implementation using Stable-Baselines3."""

import os
from typing import Optional, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
import torch


class RLAgent:
    """Wrapper around Stable-Baselines3 PPO for Clash Royale bot.
    
    This class provides a clean interface for training, prediction, and model management.
    """
    
    def __init__(self, env, config: Dict[str, Any], tensorboard_log: Optional[str] = None):
        """Initialize RL agent with PPO.
        
        Args:
            env: Gym environment (GymEnvironment instance)
            config: Dictionary with RL hyperparameters from agent_config.py
            tensorboard_log: Directory for TensorBoard logs
        """
        self.env = env
        self.config = config
        self.tensorboard_log = tensorboard_log or "./logs/tensorboard/"
        
        # Create model with hyperparameters from config
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=config.get('learning_rate', 3e-4),
            gamma=config.get('gamma', 0.99),
            gae_lambda=config.get('gae_lambda', 0.95),
            clip_range=config.get('clip_range', 0.2),
            ent_coef=config.get('ent_coef', 0.01),
            vf_coef=config.get('vf_coef', 0.5),
            max_grad_norm=config.get('max_grad_norm', 0.5),
            n_steps=config.get('n_steps', 512),
            batch_size=config.get('batch_size', 64),
            n_epochs=config.get('n_epochs', 10),
            verbose=1,
            tensorboard_log=self.tensorboard_log,
            device='auto'  # Uses GPU if available
        )
        
        print(f"Initialized PPO agent with device: {self.model.device}")
        print(f"Policy architecture: {self.model.policy}")
    
    def train(self, total_timesteps: int, callbacks: Optional[list] = None, 
              reset_num_timesteps: bool = True):
        """Train the agent.
        
        Args:
            total_timesteps: Total number of environment steps to train for
            callbacks: List of callback instances for monitoring/checkpointing
            reset_num_timesteps: Whether to reset timestep counter
        """
        print(f"Starting training for {total_timesteps} timesteps...")
        
        callback = CallbackList(callbacks) if callbacks else None
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=True
        )
        
        print("Training completed!")
    
    def predict(self, observation, deterministic: bool = False, action_mask=None):
        """Get action from policy for a given observation.
        
        Args:
            observation: Current state observation
            deterministic: If True, use mean action (no exploration)
            action_mask: Optional boolean mask of valid actions
            
        Returns:
            Tuple of (action, policy_state)
        """
        # Log observation received by agent for decision making
        print(f"[AGENT] Observation received: {observation}")
        print(f"[AGENT] Deterministic mode: {deterministic}")
        
        # If action masking is provided, set invalid actions to very low probability
        if action_mask is not None:
            # This would require custom policy - simplified version:
            action, _states = self.model.predict(observation, deterministic=deterministic)
            print(f"[AGENT] Action mask provided: {np.sum(action_mask)} valid actions")
            # In production, implement proper action masking in policy
        else:
            action, _states = self.model.predict(observation, deterministic=deterministic)
        
        print(f"[AGENT] Selected action: {action}")
        
        return action, _states
    
    def evaluate(self, n_episodes: int = 5) -> Dict[str, float]:
        """Evaluate agent performance.
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating agent for {n_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(f"Episode {episode + 1}/{n_episodes}: Reward={episode_reward:.2f}, Length={episode_length}")
        
        metrics = {
            'mean_reward': sum(episode_rewards) / len(episode_rewards),
            'std_reward': torch.tensor(episode_rewards).std().item(),
            'mean_length': sum(episode_lengths) / len(episode_lengths),
        }
        
        print(f"Evaluation results: {metrics}")
        return metrics
    
    def save(self, path: str):
        """Save model to disk.
        
        Args:
            path: File path to save model (without extension)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk.
        
        Args:
            path: File path to load model from
        """
        if not os.path.exists(f"{path}.zip"):
            raise FileNotFoundError(f"Model file not found: {path}.zip")
        
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")
    
    def get_model(self):
        """Get the underlying Stable-Baselines3 model.
        
        Returns:
            PPO model instance
        """
        return self.model
    
    def set_learning_rate(self, learning_rate: float):
        """Update learning rate during training.
        
        Args:
            learning_rate: New learning rate value
        """
        self.model.learning_rate = learning_rate
        print(f"Learning rate updated to {learning_rate}")
