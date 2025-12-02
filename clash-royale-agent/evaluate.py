"""Evaluation script for trained Clash Royale RL bot."""

import sys
import argparse
from utils import roboflow_utils
from environment import GymEnvironment
from environment.core import Environment
from agent import RLAgent
from config.agent_config import RL_CONFIG


def evaluate_model(model_path: str, n_episodes: int = 10, deterministic: bool = True):
    """Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to evaluate
        deterministic: Use deterministic policy (no exploration)
    """
    print("=" * 60)
    print("Clash Royale RL Bot - Evaluation")
    print("=" * 60)
    
    # Initialize Roboflow client
    print("\nInitializing Roboflow client...")
    roboflow_client = roboflow_utils.init_roboflow()
    print("✓ Roboflow client initialized")
    
    # Create environment
    print("\nCreating environment...")
    env_core = Environment(roboflow_client)
    gym_env = GymEnvironment(env_core)
    print("✓ Environment created")
    
    # Create agent and load model
    print(f"\nLoading model from {model_path}...")
    agent = RLAgent(
        env=gym_env,
        config=RL_CONFIG
    )
    agent.load(model_path)
    print("✓ Model loaded successfully")
    
    # Run evaluation
    print(f"\n🎮 Running {n_episodes} evaluation episodes...")
    print("Mode:", "Deterministic (no exploration)" if deterministic else "Stochastic")
    print("-" * 60)
    
    episode_rewards = []
    episode_lengths = []
    victories = 0
    defeats = 0
    draws = 0
    
    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        obs, info = gym_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = gym_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            
            print(f"  Step {episode_length}: Reward={reward:.2f}, "
                  f"Elixir={info.get('elixir', 0)}, "
                  f"Troops={info.get('troops_detected', 0)}")
        
        # Record results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        battle_result = info.get('battle_result')
        if battle_result == 'victory':
            victories += 1
            result_emoji = "🏆"
        elif battle_result == 'defeat':
            defeats += 1
            result_emoji = "💀"
        elif battle_result == 'draw':
            draws += 1
            result_emoji = "🤝"
        else:
            result_emoji = "❓"
        
        print(f"\n{result_emoji} Episode {episode + 1} ended: {battle_result}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Episode length: {episode_length} steps")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"\nBattle Results:")
    print(f"  🏆 Victories: {victories} ({victories/n_episodes*100:.1f}%)")
    print(f"  💀 Defeats: {defeats} ({defeats/n_episodes*100:.1f}%)")
    print(f"  🤝 Draws: {draws} ({draws/n_episodes*100:.1f}%)")
    print(f"\nReward Statistics:")
    print(f"  Mean: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"  Min: {min(episode_rewards):.2f}")
    print(f"  Max: {max(episode_rewards):.2f}")
    print(f"\nEpisode Length:")
    print(f"  Mean: {sum(episode_lengths)/len(episode_lengths):.1f} steps")
    print(f"  Min: {min(episode_lengths)} steps")
    print(f"  Max: {max(episode_lengths)} steps")
    print("=" * 60)
    
    # Cleanup
    gym_env.close()
    print("\n✓ Evaluation complete")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained Clash Royale RL bot')
    parser.add_argument('model_path', type=str, help='Path to saved model (without .zip)')
    parser.add_argument('--episodes', '-n', type=int, default=10, 
                        help='Number of episodes to evaluate (default: 10)')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy (with exploration)')
    
    args = parser.parse_args()
    
    deterministic = not args.stochastic
    
    try:
        evaluate_model(
            model_path=args.model_path,
            n_episodes=args.episodes,
            deterministic=deterministic
        )
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
