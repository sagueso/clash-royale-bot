import numpy as np
import ui_constants
from utils import screen_utils, roboflow_utils
from config.deck_config import DECK, get_card_by_name
from environment import BattleDetector, StateManager, RewardCalculator


def start_battle():
    while True:
        screen = screen_utils.screenshot()
        is_found = screen_utils.find_n_click('./ref_images/battle_button.png', screen)
        if is_found:
            break


def get_elixir():
    elixir = 0
    pixel = ui_constants.elixir_left
    detected_colors = []
    
    for i in range(10):
        # Get actual pixel color for logging
        actual_color = screen_utils.get_pixel_color(pixel)
        detected_colors.append(actual_color)
        
        is_match = screen_utils.check_pixel_color(pixel, ui_constants.elixir_rgb)
        
        if not is_match:
            print(f"[ELIXIR] Stopped at position {i}: Color {actual_color} doesn't match target {ui_constants.elixir_rgb}")
            break
        
        elixir += 1
        pixel = (pixel[0] + ui_constants.elixir_distance, pixel[1])
    
    # Log elixir detection details
    print(f"[ELIXIR] Detected elixir: {elixir}/10")
    print(f"[ELIXIR] Expected RGB: {ui_constants.elixir_rgb}, Tolerance: ±80")
    print(f"[ELIXIR] Pixel positions checked: {len(detected_colors)}")
    print(f"[ELIXIR] Detected colors: {detected_colors}")
    
    return elixir


class Environment:
    def __init__(self, client_param):
        self.client = client_param
        self.game_screen = None
        self.in_battle = False
        self.elixir = 0
        self.timer = None
        self.troops = []
        self.score = 0
        self.hand = []
        self.deck = DECK
        self.prev_state = None
        
        # Initialize environment modules
        self.battle_detector = BattleDetector()
        self.state_manager = StateManager()
        self.reward_calculator = RewardCalculator()

    def _get_timer(self):
        """Extract timer from game screen using OCR.
        
        Robust timer extraction with fallback to previous value if OCR fails.
        """
        try:
            timer_crop = screen_utils.crop_area(self.game_screen, ui_constants.time_top_left, ui_constants.time_bottom_right)
            timer_text = screen_utils.read_text(timer_crop).strip()
            
            # Handle newlines and extra whitespace
            timer_text = timer_text.replace('\n', '').replace(' ', '')
            
            # Check if timer text contains ':'
            if ':' not in timer_text:
                # Keep previous timer value if OCR fails
                return
            
            # Split and convert to seconds
            parts = timer_text.split(':')
            if len(parts) != 2:
                # Invalid format, keep previous value
                return
            
            minutes = int(parts[0])
            seconds = int(parts[1])
            
            # Sanity check: timer should be between 0 and 360 seconds (6 minutes max)
            total_seconds = minutes * 60 + seconds
            if 0 <= total_seconds <= 360:
                self.timer = total_seconds
            # else: keep previous timer value
            
        except (ValueError, AttributeError, IndexError) as e:
            # OCR failed or invalid data, keep previous timer value
            print(f"Timer extraction failed: {e}, keeping previous value: {self.timer}")
            pass

    def _get_troops(self):
        """Detect troops on the game screen using Roboflow API."""
        self.troops = []  # Clear previous troops
        try:
            predictions = roboflow_utils.detect_troop(self.client, self.game_screen)["predictions"]
            for det in predictions:
                self.troops.append({
                    "x_center": det["x"],
                    "y_center": det["y"],
                    "width": det["width"],
                    "height": det["height"],
                    "class": det["class_id"],
                    "confidence": det.get("confidence", 1.0)
                })
        except Exception as e:
            print(f"Error detecting troops: {e}")
            self.troops = []

    def _count_score(self):
        for troop in self.troops:
            if troop["class"] == 3:  # enemy kt
                self.score += -1
            elif troop["class"] == 4:  # enemy pt
                self.score += -1
            elif troop["class"] == 0:  # ally kt
                self.score += 1
            elif troop["class"] == 1:  # ally pt
                self.score += 1

    def _find_hand(self):
        """Find which cards are currently in hand using template matching."""
        hand = []
        for card in self.deck:
            is_found, pt1, pt2 = screen_utils.find(card["url"], self.game_screen)
            if is_found:
                # Calculate center position relative to game screen
                center_x = (pt1[0] + pt2[0]) / 2
                center_y = (pt1[1] + pt2[1]) / 2
                
                # Convert to absolute screen coordinates
                abs_x = center_x + ui_constants.game_top_left[0]
                abs_y = center_y + ui_constants.game_top_left[1]
                
                hand.append({
                    "card": card["name"],
                    "click": (abs_x, abs_y),
                    "elixir_cost": card["elixir_cost"]
                })
        self.hand = hand

    def update_environment(self, screen):
        """Update all environment state from a new screenshot.
        
        Args:
            screen: Full screenshot of the game
        """
        # Extract game area first
        print(f"\n[ENV UPDATE] Starting environment update...")
        prev_elixir = self.elixir
        self.elixir = get_elixir()
        if self.elixir != prev_elixir:
            print(f"[ENV UPDATE] Elixir changed: {prev_elixir} -> {self.elixir}")
        self.game_screen = screen_utils.crop_area(screen, ui_constants.game_top_left, ui_constants.game_bottom_right)
        
        # Check if battle has ended (uses game_screen for banners, full screen for play again)
        battle_result = self.battle_detector.detect_battle_end(self.game_screen)
        if battle_result:
            self.in_battle = False
        
        # Update all game state
        self._get_timer()
        self._get_troops()
        self._count_score()
        self._find_hand()
        
        # Check if battle has ended
        battle_result = self.battle_detector.detect_battle_end(screen)
        if battle_result:
            self.in_battle = False
    
    def reset_state(self):
        """Reset environment state for a new battle."""
        self.game_screen = None
        self.in_battle = True
        self.elixir = 0
        self.timer = None
        self.troops = []
        self.score = 0
        self.hand = []
        self.prev_state = None
    
    def get_observation(self) -> np.ndarray:
        """Get normalized observation vector for RL agent.
        
        Returns:
            Numpy array of 5 normalized features
        """
        features = self.state_manager.extract_features(self)
        return self.state_manager.encode_observation(features)
    
    def calculate_reward(self, action: int, battle_result: str = None) -> float:
        """Calculate reward for the last action taken.
        
        Args:
            action: Action ID that was taken
            battle_result: 'ended' if battle finished, None otherwise
            
        Returns:
            Reward value as float
        """
        curr_features = self.state_manager.extract_features(self)
        
        if self.prev_state is None:
            # First step, no reward yet
            self.prev_state = curr_features
            return 0.0
        
        reward = self.reward_calculator.calculate(
            self.prev_state, 
            curr_features, 
            action, 
            battle_result
        )
        
        self.prev_state = curr_features
        return reward
    
    def is_battle_ended(self, screen) -> str:
        """Check if battle has ended.
        
        Args:
            screen: Full screenshot of the game
            
        Returns:
            'victory', 'defeat', 'draw', or None if still active
        """
        # Crop to game area for banner detection
        game_screen = screen_utils.crop_area(screen, ui_constants.game_top_left, ui_constants.game_bottom_right)
        return self.battle_detector.detect_battle_end(game_screen)
    
    def start_new_battle(self) -> bool:
        """Start a new battle using battle detector.
        
        Returns:
            True if battle started successfully
        """
        success = self.battle_detector.start_battle()
        if success:
            self.reset_state()
        return success
    
    def get_state(self) -> dict:
        """Get current environment state as a dictionary.
        
        Returns:
            Dictionary containing all state information
        """
        return {
            'elixir': self.elixir,
            'troops': self.troops,
            'hand': self.hand,
            'score': self.score,
            'timer': self.timer,
            'game_screen': self.game_screen,
            'in_battle': self.in_battle
        }
