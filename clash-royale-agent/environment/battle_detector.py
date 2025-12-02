"""Battle state detection for game flow management."""

import cv2
import numpy as np
import time
from utils import screen_utils
import ui_constants


class BattleDetector:
    """Detects battle states: start, end, victory, defeat, draw."""
    
    def __init__(self):
        self.victory_template = './ref_images/victory.png'
        self.defeat_template = './ref_images/defeat.png'
        self.draw_template = './ref_images/draw.png'
        self.battle_button_template = './ref_images/battle_button.png'
        self.play_again_template = './ref_images/play_again.png'
        self.cancel_button_template = './ref_images/cancel_button.png'
        self.ok_button_template = './ref_images/ok_button.png'
        
    def is_battle_active(self, screen: np.ndarray) -> bool:
        """Check if a battle is currently active.
        
        Detects if we're in an active battle by checking for the absence of
        end-game screens and presence of game UI elements.
        
        Args:
            screen: Screenshot of the game
            
        Returns:
            True if battle is active, False otherwise
        """
        # Check if any end-game screen is present
        end_result = self.detect_battle_end(screen)
        if end_result is not None:
            return False
        
        # Check for cancel button (indicates not in battle)
        is_found, _, _ = screen_utils.find(self.cancel_button_template, screen)
        if is_found:
            return False
        
        # Battle is active if no end screen detected and cancel button not found
        return True
    
    def detect_battle_result(self, game_screen: np.ndarray) -> str:
        """Detect specific battle result using OCR on banner regions.
        
        Args:
            game_screen: Cropped game area screenshot (game-relative coordinates)
            
        Returns:
            'victory', 'defeat', or None if no specific result detected
        """
        # Crop victory banner region and check for "Winner!" text
        victory_crop = screen_utils.crop_area(
            game_screen,
            ui_constants.victory_banner_top_left,
            ui_constants.victory_banner_bottom_right
        )
        victory_text = screen_utils.read_text(victory_crop).strip()
        if victory_text == "Winner!":
            return 'victory'
        
        # Crop defeat banner region and check for "Winner!" text (opponent's banner)
        defeat_crop = screen_utils.crop_area(
            game_screen,
            ui_constants.defeat_banner_top_left,
            ui_constants.defeat_banner_bottom_right
        )
        defeat_text = screen_utils.read_text(defeat_crop).strip()
        if defeat_text == "Winner!":
            return 'defeat'
        
        return None
    
    def detect_battle_end(self, game_screen: np.ndarray) -> str:
        """Detect if battle has ended and return the result.
        
        Args:
            screen: Screenshot of the game
            
        Returns:
            'victory', 'defeat', 'draw', or None if battle still active
        """
        # First try OCR-based detection for specific results
        result = self.detect_battle_result(game_screen)
        if result:
            return result
        
        # Fallback: check for ok button (indicates draw or generic end)
        is_found, _, _ = screen_utils.find(self.ok_button_template, game_screen)
        if is_found:
            return 'draw'  # No winner detected but battle ended = draw
        
        return None
    
    def wait_for_battle_start(self, timeout: int = 30) -> bool:
        """Wait for battle to start after clicking battle button.
        
        Waits until battle UI elements appear or timeout is reached.
        
        Args:
            timeout: Maximum seconds to wait
            
        Returns:
            True if battle started, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            screen = screen_utils.screenshot()
            
            # Check if battle button is gone (indicates loading/battle started)
            is_found, _, _ = screen_utils.find(self.battle_button_template, screen)
            if not is_found:
                # Wait a bit more for battle to fully load
                time.sleep(3)
                return True
            
            time.sleep(0.5)
        
        return False
    
    def start_battle(self) -> bool:
        """Click battle button or play again button and wait for battle to start.
        
        Handles both first battle (battle_button) and subsequent battles (play_again).
        
        Returns:
            True if battle started successfully, False otherwise
        """
        max_attempts = 5
        
        for attempt in range(max_attempts):
            screen = screen_utils.screenshot()
            
            # First try play again button (for subsequent battles)
            is_found = screen_utils.find_n_click(self.play_again_template, screen)
            if is_found:
                print("Clicked play again button")
                # Wait for battle to start
                if self.wait_for_battle_start(timeout=30):
                    return True

            # If not found, try ok button and then battle button (for first battle, or play again timeout)
            is_found = screen_utils.find_n_click(self.ok_button_template, screen)
            if is_found:
                print("Clicked ok button")
                time.sleep(2)  # Wait a bit before clicking battle button
            is_found = screen_utils.find_n_click(self.battle_button_template, screen)
            if is_found:
                print("Clicked battle button")
                # Wait for battle to start
                if self.wait_for_battle_start(timeout=30):
                    return True
            
            time.sleep(2)
        
        return False
