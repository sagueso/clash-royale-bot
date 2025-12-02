"""OpenAI Gym wrapper for Clash Royale environment."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from typing import Tuple, Dict, Any

from utils import screen_utils
from actions import ActionSpace, ActionExecutor
from environment.core import Environment


class GymEnvironment(gym.Env):
    """Gym-compatible wrapper for the Clash Royale environment.
    
    This wrapper adapts the custom Environment class to the OpenAI Gym interface,
    making it compatible with standard RL libraries like Stable-Baselines3.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, environment, render_mode=None):
        """Initialize Gym environment.
        
        Args:
            environment: Instance of Environment class
            render_mode: Rendering mode (not implemented yet)
        """
        super().__init__()
        
        self.env = environment
        self.render_mode = render_mode
        
        # Initialize action system
        self.action_space_manager = ActionSpace()
        self.action_executor = ActionExecutor(self.action_space_manager)
        
        # Define action space: 25 discrete actions
        self.action_space = spaces.Discrete(self.action_space_manager.get_action_count())
        
        # Define observation space: 5 continuous values in [0, 1]
        # [elixir, ally_troops, enemy_troops, ally_towers, enemy_towers]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps_per_episode = 300  # ~5 minutes at 1s/step
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and start a new battle.
        
        Args:
            seed: Random seed (not used currently)
            options: Additional options (not used currently)
            
        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)
        
        # Start new battle
        print("Starting new battle...")
        success = self.env.start_new_battle()
        
        if not success:
            print("Failed to start battle, retrying...")
            time.sleep(2)
            success = self.env.start_new_battle()
        
        # Wait for battle to fully load
        time.sleep(3)
        
        # Get initial state
        screen = screen_utils.screenshot()
        self.env.update_environment(screen)
        
        observation = self.env.get_observation()
        self.current_step = 0
        
        info = {
            'elixir': self.env.elixir,
            'hand_size': len(self.env.hand),
            'troops_detected': len(self.env.troops)
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one action in the environment.
        
        Args:
            action: Action ID to execute (0-24)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        # Take screenshot and update environment state
        screen = screen_utils.screenshot()
        self.env.update_environment(screen)
        
        # Log pre-action state for decision making analysis
        print(f"\n{'='*80}")
        print(f"[DECISION] Step {self.current_step} - Before action execution")
        print(f"[DECISION] Elixir: {self.env.elixir}/10, Hand size: {len(self.env.hand)}, Troops detected: {len(self.env.troops)}")
        print(f"[DECISION] Selected action ID: {action}")
        
        # Get valid action mask for context
        valid_mask = self.get_valid_action_mask()
        valid_actions = np.where(valid_mask)[0]
        print(f"[DECISION] Valid actions: {valid_actions.tolist()} (total: {len(valid_actions)})")
        print(f"[DECISION] Action valid: {valid_mask[action]}")
        print(f"{'='*80}\n")
        
        # Execute action
        action_success = self.action_executor.execute(action, self.env)
        
        # Wait for action to take effect (~1s for troop detection)
        time.sleep(1.0)
        
        # Get new state after action
        screen = screen_utils.screenshot()
        self.env.update_environment(screen)
        
        # Get observation
        observation = self.env.get_observation()
        print(f"[POST-ACTION] New elixir: {self.env.elixir}/10, Troops: {len(self.env.troops)}, Action success: {action_success}\n")
        
        # Check if battle ended (needs full screen for banners)
        battle_result = self.env.is_battle_ended(screen)
        terminated = battle_result is not None
        
        # Check if max steps reached
        truncated = self.current_step >= self.max_steps_per_episode
        
        # Calculate reward
        reward = self.env.calculate_reward(action, battle_result)
        
        # Additional info for logging
        info = {
            'elixir': self.env.elixir,
            'hand_size': len(self.env.hand),
            'troops_detected': len(self.env.troops),
            'action_success': action_success,
            'battle_result': battle_result,
            'step': self.current_step
        }
        
        # If battle ended, click play again for next episode
        if terminated:
            print(f"Battle ended: {battle_result}")
            time.sleep(2)
            # Note: play again will be clicked in next reset()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (not implemented)."""
        if self.render_mode == 'human':
            # Could display the game screen here
            pass
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_valid_action_mask(self) -> np.ndarray:
        """Get boolean mask of valid actions for action masking.
        
        Returns:
            Boolean array of shape (25,) where True = valid action
        """
        return np.array(
            self.action_space_manager.get_action_mask(self.env.hand, self.env.elixir),
            dtype=np.bool_
        )
