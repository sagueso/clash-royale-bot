"""Simple test for card placement - detect card and place it."""

import cv2
import time
from utils import screen_utils
import ui_constants


def test_simple_card_placement():
    """Test card detection and placement with detailed logging."""
    print("=" * 60)
    print("SIMPLE CARD PLACEMENT TEST")
    print("=" * 60)
    
    # Configuration - EDIT THESE
    card_image_path = "./ref_images/deck/knight.png"  # Change to your card
    card_name = "Knight"  # Change to match your card
    
    # Target position (index 0-5)
    # 0: behind_king_left, 1: behind_king_right
    # 2: center_left, 3: center_right
    # 4: princess_left, 5: princess_right
    position_index = 5  # center_left
    
    positions = [
        ("behind_king_left", ui_constants.behind_king_left),
        ("behind_king_right", ui_constants.behind_king_right),
        ("center_left", ui_constants.center_left),
        ("center_right", ui_constants.center_right),
        ("princess_left", ui_constants.princess_left),
        ("princess_right", ui_constants.princess_right),
    ]
    
    position_name, position_coords = positions[position_index]
    
    print(f"\nConfiguration:")
    print(f"  Card image: {card_image_path}")
    print(f"  Card name: {card_name}")
    print(f"  Target position: {position_name} at {position_coords}")
    print(f"\nMake sure you are in battle with {card_name} in hand!")
    print("Starting in 3 seconds...\n")
    time.sleep(3)
    
    # Step 1: Take screenshot
    print("Step 1: Taking screenshot...")
    screen = screen_utils.screenshot()
    print(f"  ✓ Screenshot captured: {screen.shape}")
    
    # Save full screenshot for debugging
    cv2.imwrite("images/result_images/test_full_screen.png", screen)
    print("  ✓ Saved: images/result_images/test_full_screen.png")
    
    # Step 2: Crop to game area
    print("\nStep 2: Cropping to game area...")
    game_screen = screen_utils.crop_area(
        screen, 
        ui_constants.game_top_left, 
        ui_constants.game_bottom_right
    )
    print(f"  ✓ Game area: {ui_constants.game_top_left} to {ui_constants.game_bottom_right}")
    print(f"  ✓ Cropped size: {game_screen.shape}")
    
    # Save game screen for debugging
    cv2.imwrite("images/result_images/test_game_screen.png", game_screen)
    print("  ✓ Saved: images/result_images/test_game_screen.png")
    
    # Step 3: Find card in hand
    print(f"\nStep 3: Looking for {card_name} in hand...")
    print(f"  Template: {card_image_path}")
    
    is_found, pt1, pt2 = screen_utils.find(card_image_path, game_screen)
    
    if not is_found:
        print(f"  ✗ FAILED: {card_name} not found in hand!")
        print(f"  Make sure:")
        print(f"    - You are in an active battle")
        print(f"    - {card_name} is visible in your hand")
        print(f"    - Template image matches card appearance")
        return False
    
    print(f"  ✓ Card found!")
    print(f"    Top-left corner: {pt1}")
    print(f"    Bottom-right corner: {pt2}")
    
    # Calculate click position (center of card) - relative to game screen
    card_click_x = (pt1[0] + pt2[0]) // 2
    card_click_y = (pt1[1] + pt2[1]) // 2
    print(f"    Center (relative to game screen): ({card_click_x}, {card_click_y})")
    
    # Convert to absolute screen coordinates
    card_abs_x = card_click_x + ui_constants.game_top_left[0]
    card_abs_y = card_click_y + ui_constants.game_top_left[1]
    print(f"    Center (absolute screen): ({card_abs_x}, {card_abs_y})")
    
    # Draw card location on game screen
    debug_screen = game_screen.copy()
    cv2.rectangle(debug_screen, pt1, pt2, (0, 255, 0), 3)
    cv2.circle(debug_screen, (card_click_x, card_click_y), 10, (0, 0, 255), -1)
    cv2.putText(debug_screen, card_name, (pt1[0], pt1[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite("images/result_images/test_card_detected.png", debug_screen)
    print("  ✓ Saved: images/result_images/test_card_detected.png")
    
    # Step 4: Prepare placement
    print(f"\nStep 4: Preparing to place card at {position_name}...")
    # Note: positions in ui_constants are GAME-RELATIVE, need conversion to absolute screen
    deploy_rel_x, deploy_rel_y = position_coords
    deploy_x = deploy_rel_x + ui_constants.game_top_left[0]
    deploy_y = deploy_rel_y + ui_constants.game_top_left[1]
    print(f"  Deployment position (game-relative): ({deploy_rel_x}, {deploy_rel_y})")
    print(f"  Deployment position (absolute screen): ({deploy_x}, {deploy_y})")
    
    # Draw placement location on game screen (already have game-relative coords)
    cv2.circle(debug_screen, (deploy_rel_x, deploy_rel_y), 15, (255, 0, 0), 3)
    cv2.putText(debug_screen, position_name, (deploy_rel_x - 50, deploy_rel_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.imwrite("images/result_images/test_placement_plan.png", debug_screen)
    print("  ✓ Saved: images/result_images/test_placement_plan.png")
    
    # Step 5: Execute placement
    print("\nStep 5: Executing card placement...")
    print(f"  Action: Click card at ({card_abs_x}, {card_abs_y})")
    print(f"  Then: Click deployment at ({deploy_x}, {deploy_y})")
    print("  Executing in 2 seconds...")
    time.sleep(2)
    
    try:
        # Click card in hand
        print(f"  → Clicking card...")
        screen_utils.click(card_abs_x, card_abs_y)
        time.sleep(0.15)  # Small delay between clicks
        
        # Click deployment position
        print(f"  → Clicking deployment position...")
        screen_utils.click(deploy_x, deploy_y)
        time.sleep(0.15)
        
        print("  ✓ Card placement executed!")
        
    except Exception as e:
        print(f"  ✗ FAILED during execution: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Verify (take new screenshot)
    print("\nStep 6: Taking post-placement screenshot...")
    time.sleep(0.5)
    screen_after = screen_utils.screenshot()
    game_screen_after = screen_utils.crop_area(
        screen_after,
        ui_constants.game_top_left,
        ui_constants.game_bottom_right
    )
    cv2.imwrite("images/result_images/test_after_placement.png", game_screen_after)
    print("  ✓ Saved: images/result_images/test_after_placement.png")
    print("  Check this image to see if card was placed!")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nGenerated debug images:")
    print("  1. test_full_screen.png - Full screenshot")
    print("  2. test_game_screen.png - Cropped game area")
    print("  3. test_card_detected.png - Card location marked")
    print("  4. test_placement_plan.png - Card + deployment positions")
    print("  5. test_after_placement.png - Result after placement")
    print("\nAll images in: images/result_images/")
    
    return True


def test_all_positions():
    """Test placing a card at all 6 positions sequentially."""
    print("=" * 60)
    print("TEST ALL POSITIONS")
    print("=" * 60)
    
    card_image_path = "./ref_images/deck/knight.png"
    card_name = "Knight"
    
    positions = [
        ("behind_king_left", ui_constants.behind_king_left),
        ("behind_king_right", ui_constants.behind_king_right),
        ("center_left", ui_constants.center_left),
        ("center_right", ui_constants.center_right),
        ("princess_left", ui_constants.princess_left),
        ("princess_right", ui_constants.princess_right),
    ]
    
    print(f"\nWill test placing {card_name} at all 6 positions")
    print("Make sure you have enough elixir and the card in hand!")
    print("Starting in 5 seconds...\n")
    time.sleep(5)
    
    for i, (pos_name, pos_coords) in enumerate(positions):
        print(f"\n{'='*60}")
        print(f"Position {i+1}/6: {pos_name}")
        print(f"{'='*60}")
        
        # Take screenshot
        screen = screen_utils.screenshot()
        game_screen = screen_utils.crop_area(
            screen,
            ui_constants.game_top_left,
            ui_constants.game_bottom_right
        )
        
        # Find card
        is_found, pt1, pt2 = screen_utils.find(card_image_path, game_screen)
        
        if not is_found:
            print(f"✗ Card not found, skipping {pos_name}")
            continue
        
        # Calculate card position (convert to absolute screen)
        card_click_x = (pt1[0] + pt2[0]) // 2 + ui_constants.game_top_left[0]
        card_click_y = (pt1[1] + pt2[1]) // 2 + ui_constants.game_top_left[1]
        
        # Convert deployment position to absolute screen
        deploy_abs_x = pos_coords[0] + ui_constants.game_top_left[0]
        deploy_abs_y = pos_coords[1] + ui_constants.game_top_left[1]
        
        # Execute placement
        print(f"Placing at {pos_name} (game-rel: {pos_coords}, screen-abs: ({deploy_abs_x}, {deploy_abs_y}))")
        screen_utils.click(card_click_x, card_click_y)
        time.sleep(0.15)
        screen_utils.click(deploy_abs_x, deploy_abs_y)
        time.sleep(0.15)
        
        print(f"✓ Placed at {pos_name}")
        
        # Wait before next placement
        print("Waiting 3 seconds before next placement...")
        time.sleep(3)
    
    print("\n" + "=" * 60)
    print("ALL POSITIONS TESTED")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        test_all_positions()
    else:
        test_simple_card_placement()
        print("\nTo test all 6 positions: python test_simple_placement.py all")
