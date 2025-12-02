"""State management and observation preprocessing."""

import numpy as np
from typing import Dict, List


class StateManager:
    """Manages game state representation and feature extraction."""
    
    # Class IDs from Roboflow model
    CLASS_ALLY_KING_TOWER = 0
    CLASS_ALLY_PRINCESS_TOWER = 1
    CLASS_ALLY_TROOP = 2 # TODO Check
    CLASS_ENEMY_KING_TOWER = 3
    CLASS_ENEMY_PRINCESS_TOWER = 4
    CLASS_ENNEMY_TROOP = 5
        
    
    def __init__(self, screen_width: int = 640, screen_height: int = 640):
        """Initialize state manager.
        
        Args:
            screen_width: Width of normalized game screen (Roboflow default: 640)
            screen_height: Height of normalized game screen (Roboflow default: 640)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
    def normalize_troop_positions(self, troops: List[dict]) -> np.ndarray:
        """Normalize troop positions to [0, 1] range.
        
        Args:
            troops: List of troop detections with x_center, y_center, width, height
            
        Returns:
            Normalized positions array
        """
        if not troops:
            return np.array([])
        
        normalized = []
        for troop in troops:
            normalized.append([
                troop["x_center"] / self.screen_width,
                troop["y_center"] / self.screen_height,
                troop["width"] / self.screen_width,
                troop["height"] / self.screen_height,
                troop["class"]
            ])
        
        return np.array(normalized)
    
    def extract_features(self, env) -> Dict:
        """Extract high-level features from environment state.
        
        Args:
            env: Environment instance
            
        Returns:
            Dictionary of extracted features
        """
        print(f"[STATE] Raw elixir value from env: {env.elixir}")
        
        features = {
            'elixir': env.elixir / 10,  # Normalize to [0, 1]
            'ally_troops_count': 0,
            'enemy_troops_count': 0,
            'ally_towers_alive': 0,
            'enemy_towers_alive': 0,
        }
        
        if env.troops is None or len(env.troops) == 0:
            return features
        
        # Count troops and calculate pressure zones
        for troop in env.troops:
            troop_class = troop["class"]
            x_center = troop["x_center"]
            y_center = troop["y_center"]
            
            # Count towers
            if troop_class in [self.CLASS_ALLY_KING_TOWER, self.CLASS_ALLY_PRINCESS_TOWER]:
                features['ally_towers_alive'] += 1
            elif troop_class in [self.CLASS_ENEMY_KING_TOWER, self.CLASS_ENEMY_PRINCESS_TOWER]:
                features['enemy_towers_alive'] += 1
            
            # Skip towers for troop counting and pressure
            if troop_class in [0, 1, 3, 4]:
                continue
            
            if troop_class == self.CLASS_ALLY_TROOP:
                features['ally_troops_count'] += 1
            elif troop_class == self.CLASS_ENNEMY_TROOP:
                features['enemy_troops_count'] += 1
        
        return features
    
    def encode_observation(self, features: Dict) -> np.ndarray:
        """Encode features into a flat observation vector for neural network.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Flattened numpy array of observations
        """
        obs = np.array([
            features['elixir'],
            min(features['ally_troops_count'], 20) / 20.0,
            min(features['enemy_troops_count'], 20) / 20.0,
            features['ally_towers_alive'] / 3.0,
            features['enemy_towers_alive'] / 3.0,
        ], dtype=np.float32)
        
        print(f"[OBSERVATION] Encoded observation vector: elixir={obs[0]:.3f}, ally_troops={obs[1]:.3f}, enemy_troops={obs[2]:.3f}, ally_towers={obs[3]:.3f}, enemy_towers={obs[4]:.3f}")
        print(f"[FEATURES] Raw features: {features}")
        
        return obs
    
    def get_observation_space_size(self) -> int:
        """Get the size of the observation vector.
        
        Returns:
            Size of observation space
        """
        return 5  # Number of features in encode_observation
