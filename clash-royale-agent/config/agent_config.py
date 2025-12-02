"""Agent and training configuration."""

# Reinforcement Learning Hyperparameters
RL_CONFIG = {
    # PPO Hyperparameters
    'learning_rate': 3e-4,
    'gamma': 0.99,  # Discount factor
    'gae_lambda': 0.95,  # GAE lambda
    'clip_range': 0.2,  # PPO clip range
    'ent_coef': 0.01,  # Entropy coefficient for exploration
    'vf_coef': 0.5,  # Value function coefficient
    'max_grad_norm': 0.5,  # Gradient clipping
    'n_steps': 512,  # Steps per update (considering ~1s per step)
    'batch_size': 64,
    'n_epochs': 10,
    
    # Environment
    'action_space_size': 25,  # 4 cards × 6 positions + 1 wait
    'observation_space_size': 5,  # elixir, ally_troops, enemy_troops, ally_towers, enemy_towers
    
    # Training
    'total_timesteps': 100000,  # Total training steps
    'save_freq': 1000,  # Save model every N steps
    'log_interval': 10,  # Log every N updates
    
    # Evaluation
    'eval_freq': 5000,  # Evaluate every N steps
    'eval_episodes': 5,  # Number of episodes for evaluation
}

# Reward Weights (can override RewardCalculator defaults)
REWARD_WEIGHTS = {
    'tower_destroyed': 100.0,
    'tower_lost': -100.0,
    'elixir_advantage': 1.0,
    'win': 1000.0,
    'loss': -1000.0,
    'draw': 0.0,
    'time_penalty': -0.01,
}

# Action Execution Settings
ACTION_CONFIG = {
    'click_delay': 0.05,  # Delay between card click and placement click
    'deploy_delay': 0.1,  # Delay after placing card
    'wait_delay': 0.1,  # Delay for wait action
}
