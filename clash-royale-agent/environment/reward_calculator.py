"""Reward calculation for reinforcement learning."""

from typing import Dict


class RewardCalculator:
    """Calculates rewards based on game state changes."""
    
    def __init__(self, weights: Dict[str, float] = None):
        """Initialize reward calculator with weight parameters.
        
        Args:
            weights: Dictionary of reward component weights
        """
        # Default reward weights
        self.weights = weights or {
            'tower_destroyed': 100.0,
            'tower_lost': -100.0,
            'elixir_advantage': 1.0,
            'win': 1000.0,
            'loss': -1000.0,
            'draw': 0.0,
            'time_penalty': -0.01,  # Small penalty per step to encourage faster wins
        }
    
    def calculate(self, prev_state: Dict, curr_state: Dict, 
                  action: int, battle_result: str = None) -> float:
        """Calculate reward based on state transition.
        
        Args:
            prev_state: Previous state features
            curr_state: Current state features
            action: Action taken
            battle_result: 'victory', 'defeat', 'draw', or None if battle ongoing
            
        Returns:
            Calculated reward as float
        """
        reward = 0.0
        
        # Terminal rewards
        if battle_result == 'victory':
            return self.weights['win']
        elif battle_result == 'defeat':
            return self.weights['loss']
        elif battle_result == 'draw':
            return self.weights['draw']
        
        # Tower rewards/penalties
        reward += self._tower_reward(
            curr_state['ally_towers_alive'],
            curr_state['enemy_towers_alive'],
            prev_state['ally_towers_alive'],
            prev_state['enemy_towers_alive']
        )
        
        # Elixir management reward
        reward += self._elixir_advantage_reward(
            curr_state['elixir'],
            action
        )
        
        # Time penalty to encourage faster wins
        reward += self.weights['time_penalty']
        
        return reward
    
    def _tower_reward(self, ally_towers: int, enemy_towers: int, 
                      prev_ally_towers: int, prev_enemy_towers: int) -> float:
        """Calculate reward based on tower status changes.
        
        Args:
            ally_towers: Current ally towers alive
            enemy_towers: Current enemy towers alive
            prev_ally_towers: Previous ally towers alive
            prev_enemy_towers: Previous enemy towers alive
            
        Returns:
            Tower-based reward
        """
        reward = 0.0
        
        # Reward for destroying enemy towers
        if enemy_towers < prev_enemy_towers:
            reward += self.weights['tower_destroyed'] * (prev_enemy_towers - enemy_towers)
        
        # Penalty for losing our towers
        if ally_towers < prev_ally_towers:
            reward += self.weights['tower_lost'] * (prev_ally_towers - ally_towers)
        
        return reward
    
    def _elixir_advantage_reward(self, elixir: float, action: int) -> float:
        """Calculate reward based on elixir management.
        
        Args:
            elixir: Current elixir amount
            action: Action taken
            
        Returns:
            Elixir management reward
        """
        # Small positive reward for maintaining good elixir (not capping at 10)
        if elixir < 10:  # Not at cap
            return self.weights['elixir_advantage'] * 0.1
        else:
            # Small penalty for capping elixir (wasted generation)
            return self.weights['elixir_advantage'] * -0.5
    
