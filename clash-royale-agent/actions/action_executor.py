"""Action execution for card placement."""

import time
from typing import Optional
from utils import screen_utils
import ui_constants


class ActionExecutor:
    """Executes actions by placing cards on the battlefield."""
    
    def __init__(self, action_space):
        """Initialize action executor.
        
        Args:
            action_space: ActionSpace instance for decoding actions
        """
        self.action_space = action_space
        # Define 6 strategic placement positions
        self.positions = [
            ui_constants.behind_king_left,      # 0
            ui_constants.behind_king_right,     # 1
            ui_constants.center_left,           # 2
            ui_constants.center_right,          # 3
            ui_constants.princess_left,         # 4
            ui_constants.princess_right,        # 5
        ]
    
    def execute(self, action_id: int, env) -> bool:
        """Execute an action in the environment.
        
        Args:
            action_id: Action ID to execute (0-24)
            env: Environment instance with current game state
            
        Returns:
            True if action executed successfully, False otherwise
        """
        # Decode action
        card_idx, position_idx = self.action_space.decode_action(action_id)
        
        # Wait action - do nothing
        if card_idx is None:
            time.sleep(0.1)
            return True
        
        # Validate card index exists in hand
        if env.hand is None or card_idx >= len(env.hand):
            print(f"Card index {card_idx} not in hand (hand size: {len(env.hand) if env.hand else 0})")
            return False
        
        card_info = env.hand[card_idx]
        
        # Validate elixir
        card_cost = card_info.get("elixir_cost", 0)
        if card_cost > env.elixir:
            print(f"Not enough elixir for {card_info.get('card')} (cost: {card_cost}, have: {env.elixir})")
            return False
        
        # Place card
        return self.place_card(card_info, position_idx)
    
    def place_card(self, card_info: dict, position_idx: int) -> bool:
        """Place a card at a specific position.
        
        Args:
            card_info: Dictionary with 'card', 'click', 'elixir_cost'
            position_idx: Index of deployment position (0-5)
            
        Returns:
            True if placement successful
        """
        # Get card click position from hand (already in absolute screen coordinates)
        card_x, card_y = card_info["click"]
        
        # Get deployment position (game-relative from ui_constants)
        deploy_rel_x, deploy_rel_y = self.positions[position_idx]
        
        # Convert deployment position to absolute screen coordinates
        deploy_abs_x = deploy_rel_x + ui_constants.game_top_left[0]
        deploy_abs_y = deploy_rel_y + ui_constants.game_top_left[1]
        
        # Click card in hand, then click deployment position
        screen_utils.click_n_click(
            (int(card_x), int(card_y)), 
            (int(deploy_abs_x), int(deploy_abs_y))
        )
        
        time.sleep(0.1)  # Wait for card to deploy
        
        position_name = self.action_space.get_position_name(position_idx)
        print(f"Placed {card_info['card']} at {position_name} (game-rel: {deploy_rel_x},{deploy_rel_y} → screen-abs: {deploy_abs_x},{deploy_abs_y})")
        return True
    
    def get_positions(self) -> list:
        """Get list of all deployment positions.
        
        Returns:
            List of (x, y) tuples
        """
        return self.positions.copy()
