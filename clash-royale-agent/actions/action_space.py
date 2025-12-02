"""Action space definition: 4 cards × 6 positions = 24 + 1 wait = 25 actions"""

from typing import List, Tuple, Optional


class ActionSpace:
    """Defines the discrete action space for the RL agent.
    
    Action encoding:
    - Actions 0-23: card_idx * 6 + position_idx
    - Action 24: wait/do nothing
    
    Where:
    - card_idx: 0-3 (index in current hand)
    - position_idx: 0-5 (behind_king_left, behind_king_right, center_left, 
                          center_right, princess_left, princess_right)
    """
    
    def __init__(self):
        """Initialize action space."""
        self.num_positions = 6
        self.max_hand_size = 4
        # 4 cards × 6 positions + 1 wait = 25 actions
        self.num_actions = self.max_hand_size * self.num_positions + 1
        self.wait_action = 24
    
    def get_action_count(self) -> int:
        """Get total number of possible actions.
        
        Returns:
            Total action count (25)
        """
        return self.num_actions
    
    def decode_action(self, action_id: int) -> Tuple[Optional[int], Optional[int]]:
        """Decode action ID to (card_idx, position_idx).
        
        Args:
            action_id: Integer action ID (0-24)
            
        Returns:
            Tuple of (card_idx, position_idx), or (None, None) if wait action
        """
        if action_id < 0 or action_id >= self.num_actions:
            raise ValueError(f"Invalid action_id: {action_id}. Must be 0-{self.num_actions-1}")
        
        if action_id == self.wait_action:
            return None, None
        
        card_idx = action_id // self.num_positions
        position_idx = action_id % self.num_positions
        return card_idx, position_idx
    
    def encode_action(self, card_idx: Optional[int], position_idx: Optional[int]) -> int:
        """Encode (card_idx, position_idx) to action ID.
        
        Args:
            card_idx: Index of card in hand (0-3), or None for wait
            position_idx: Index of position (0-5), or None for wait
            
        Returns:
            Integer action ID
        """
        if card_idx is None or position_idx is None:
            return self.wait_action
        
        if card_idx < 0 or card_idx >= self.max_hand_size:
            raise ValueError(f"Invalid card_idx: {card_idx}. Must be 0-{self.max_hand_size-1}")
        
        if position_idx < 0 or position_idx >= self.num_positions:
            raise ValueError(f"Invalid position_idx: {position_idx}. Must be 0-{self.num_positions-1}")
        
        return card_idx * self.num_positions + position_idx
    
    def get_valid_actions(self, hand: List[dict], elixir: int) -> List[int]:
        """Get list of valid action IDs based on current hand and elixir.
        
        This is for action masking - prevents the agent from selecting
        invalid actions (cards too expensive or hand slots that don't exist).
        
        Args:
            hand: List of cards currently in hand with 'elixir_cost'
            elixir: Current elixir amount
            
        Returns:
            List of valid action IDs
        """
        valid_actions = [self.wait_action]  # Can always wait
        
        if hand is None or len(hand) == 0:
            return valid_actions
        
        for card_idx, card_info in enumerate(hand):
            card_cost = card_info.get("elixir_cost", 0)
            
            # Check if we have enough elixir for this card
            if card_cost <= elixir:
                # Add all 6 positions for this card
                for position_idx in range(self.num_positions):
                    action_id = self.encode_action(card_idx, position_idx)
                    valid_actions.append(action_id)
        
        return valid_actions
    
    def is_action_valid(self, action_id: int, hand: List[dict], elixir: int) -> bool:
        """Check if a specific action is valid.
        
        Args:
            action_id: Action to check
            hand: Current hand
            elixir: Current elixir
            
        Returns:
            True if action is valid
        """
        return action_id in self.get_valid_actions(hand, elixir)
    
    def get_action_mask(self, hand: List[dict], elixir: int) -> List[bool]:
        """Get boolean mask of valid actions for neural network.
        
        Args:
            hand: Current hand
            elixir: Current elixir
            
        Returns:
            List of 25 booleans, True if action is valid
        """
        valid_actions = self.get_valid_actions(hand, elixir)
        return [i in valid_actions for i in range(self.num_actions)]
    
    def get_position_name(self, position_idx: int) -> str:
        """Get human-readable name for position index.
        
        Args:
            position_idx: Position index (0-5)
            
        Returns:
            Position name string
        """
        position_names = [
            "behind_king_left",
            "behind_king_right",
            "center_left",
            "center_right",
            "princess_left",
            "princess_right"
        ]
        return position_names[position_idx]
