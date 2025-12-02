"""Training script for Clash Royale RL bot."""

import os
import time
from datetime import datetime

from utils import roboflow_utils
from environment import GymEnvironment
from environment.core import Environment
from agent import RLAgent, CheckpointCallback, TensorBoardCallback, EvaluationCallback
from config.agent_config import RL_CONFIG, REWARD_WEIGHTS


def setup_directories():
    """Create necessary directories for training."""
    directories = [
        "./logs/tensorboard/",
        "./models/checkpoints/",
        "./models/best/",
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def main():
    """Main training function."""
    print("=" * 60)
    print("Clash Royale RL Bot - Training")
    print("=" * 60)
    
    # Setup directories
    setup_directories()
    
    # Initialize Roboflow client
    print("\nInitializing Roboflow client...")
    roboflow_client = roboflow_utils.init_roboflow()
    print("✓ Roboflow client initialized")
    
    # Create environment
    print("\nCreating environment...")
    env_core = Environment(roboflow_client)
    gym_env = GymEnvironment(env_core)
    print("✓ Environment created")
    print(f"  - Observation space: {gym_env.observation_space}")
    print(f"  - Action space: {gym_env.action_space}")
    
    # Create agent
    print("\nInitializing RL agent...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log = f"./logs/tensorboard/cr_bot_{timestamp}"
    
    agent = RLAgent(
        env=gym_env,
        config=RL_CONFIG,
        tensorboard_log=tensorboard_log
    )
    print("✓ Agent initialized")
    print(f"  - TensorBoard logs: {tensorboard_log}")
    
    # Setup callbacks
    print("\nSetting up training callbacks...")
    callbacks = [
        CheckpointCallback(
            save_freq=RL_CONFIG['save_freq'],
            save_path="./models/checkpoints/",
            name_prefix=f"cr_bot_{timestamp}",
            verbose=1
        ),
        TensorBoardCallback(verbose=1),
    ]
    print("✓ Callbacks configured")
    print(f"  - Checkpoint every {RL_CONFIG['save_freq']} steps")
    print(f"  - TensorBoard logging enabled")
    
    # Training configuration
    total_timesteps = RL_CONFIG['total_timesteps']
    print("\n" + "=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: {RL_CONFIG['learning_rate']}")
    print(f"Batch size: {RL_CONFIG['batch_size']}")
    print(f"N steps: {RL_CONFIG['n_steps']}")
    print(f"Estimated time: ~{total_timesteps // 3600} hours (at 1s/step)")
    print("=" * 60)
    
    # Start training
    print("\n🚀 Starting training...")
    print("Press Ctrl+C to stop training and save the model\n")
    
    try:
        start_time = time.time()
        agent.train(
            total_timesteps=total_timesteps,
            callbacks=callbacks
        )
        training_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("✓ Training completed!")
        print(f"Total training time: {training_time / 3600:.2f} hours")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        training_time = time.time() - start_time
        print(f"Training time: {training_time / 3600:.2f} hours")
    
    # Save final model
    final_model_path = f"./models/cr_bot_final_{timestamp}"
    print(f"\n💾 Saving final model to {final_model_path}...")
    agent.save(final_model_path)
    print("✓ Model saved successfully")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    gym_env.close()
    print("✓ Environment closed")
    
    print("\n" + "=" * 60)
    print("Training session ended")
    print("=" * 60)
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir={tensorboard_log}")
    print(f"\nTo evaluate the model:")
    print(f"  python evaluate.py {final_model_path}")


if __name__ == "__main__":
    main()
