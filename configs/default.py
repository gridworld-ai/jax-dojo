"""Default hyperparameters for PPO training on Haber-Bosch environment."""

from environments.haber_bosch import P_NOMINAL_PA, P_MIN_PA

# Environment parameters
ENV_CONFIG = {
    "dt": 1.0,
    "max_steps": 1000,
}

# PPO hyperparameters - tuned for continuous control
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,           # Rollout buffer size per env
    "batch_size": 256,         # Larger batch for continuous control
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,     # No value function clipping
    "normalize_advantage": True,
    "ent_coef": 0.01,          # Small entropy bonus for exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": None,         # No early stopping on KL
    "verbose": 1,
}

# Normalization settings
NORMALIZE_CONFIG = {
    "norm_obs": True,          # Normalize observations
    "norm_reward": True,       # Normalize rewards (important!)
    "clip_obs": 10.0,
    "clip_reward": 10.0,
    "gamma": 0.99,             # Must match PPO gamma
}

# Training settings
TRAINING_CONFIG = {
    "total_timesteps": 1_000_000,
    "n_envs": 4,               # Parallel environments
    "eval_freq": 10_000,       # Evaluate every N timesteps
    "n_eval_episodes": 10,
    "checkpoint_freq": 50_000,
    "log_interval": 10,        # Log every N updates
}

# Paths
PATHS = {
    "tensorboard_log": "./logs/tensorboard/",
    "checkpoint_dir": "./logs/checkpoints/",
    "best_model_dir": "./logs/best_model/",
    "eval_log_dir": "./logs/eval/",
    "vecnorm_path": "./logs/vecnormalize.pkl",
}
