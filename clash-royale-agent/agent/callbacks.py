"""Callbacks for training monitoring and model management."""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class CheckpointCallback(BaseCallback):
    """Callback for saving model checkpoints during training.
    
    Saves the model at regular intervals and keeps track of the best model.
    """
    
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "model",
                 verbose: int = 1):
        """Initialize checkpoint callback.
        
        Args:
            save_freq: Save model every N steps
            save_path: Directory to save models
            name_prefix: Prefix for saved model files
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        """Called at each training step.
        
        Returns:
            True to continue training
        """
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps"
            )
            self.model.save(model_path)
            
            if self.verbose > 0:
                print(f"Checkpoint saved: {model_path}")
        
        return True


class TensorBoardCallback(BaseCallback):
    """Callback for logging custom metrics to TensorBoard.
    
    Logs game-specific metrics beyond the default RL metrics.
    """
    
    def __init__(self, verbose: int = 0):
        """Initialize TensorBoard callback.
        
        Args:
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        """Called at each training step.
        
        Returns:
            True to continue training
        """
        # Log info from the environment
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]
            
            # Log game-specific metrics
            self.logger.record("game/elixir", info.get("elixir", 0))
            self.logger.record("game/hand_size", info.get("hand_size", 0))
            self.logger.record("game/troops_detected", info.get("troops_detected", 0))
            self.logger.record("game/step", info.get("step", 0))
            
            # Check if episode ended
            if self.locals.get("dones", [False])[0]:
                self.episode_count += 1
                battle_result = info.get("battle_result")
                
                if battle_result:
                    self.logger.record("game/battle_ended", 1)
                    if self.verbose > 0:
                        print(f"Episode {self.episode_count} ended: {battle_result}")
        
        return True


class EvaluationCallback(BaseCallback):
    """Callback for periodic evaluation during training.
    
    Evaluates the agent at regular intervals to track learning progress.
    """
    
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int = 5,
                 save_best: bool = True, save_path: str = "./models/best/",
                 verbose: int = 1):
        """Initialize evaluation callback.
        
        Args:
            eval_env: Separate environment for evaluation
            eval_freq: Evaluate every N steps
            n_eval_episodes: Number of episodes to evaluate
            save_best: Save model if it's the best so far
            save_path: Directory to save best model
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_best = save_best
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        
        if save_best:
            os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        """Called at each training step.
        
        Returns:
            True to continue training
        """
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation episodes
            episode_rewards = []
            episode_lengths = []
            
            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    episode_length += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            # Calculate statistics
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            
            # Log to TensorBoard
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_episode_length", mean_length)
            
            if self.verbose > 0:
                print(f"Eval (step {self.num_timesteps}): "
                      f"mean_reward={mean_reward:.2f} ± {std_reward:.2f}, "
                      f"mean_length={mean_length:.1f}")
            
            # Save best model
            if self.save_best and mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                model_path = os.path.join(self.save_path, "best_model")
                self.model.save(model_path)
                
                if self.verbose > 0:
                    print(f"New best model saved with reward: {mean_reward:.2f}")
        
        return True
