#!/usr/bin/env python
"""Train PPO agent on Haber-Bosch environment using SBX."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from environments import HaberBoschEnv, NormalizedActionWrapper
from configs.default import (
    ENV_CONFIG,
    PPO_CONFIG,
    NORMALIZE_CONFIG,
    TRAINING_CONFIG,
    PATHS,
)


class HaberBoschMetricsCallback(BaseCallback):
    """Custom callback to log Haber-Bosch-specific metrics to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_temp_violations = []
        self.episode_pressure_violations = []
        self.T_MAX = 273.15 + 520.0
        self.T_MIN = 273.15 + 350.0
        self.P_NOM = 152e5
        self.P_MIN = 100e5

    def _on_step(self) -> bool:
        # Access info from the environment
        infos = self.locals.get("infos", [])
        for info in infos:
            if "state" in info:
                state = info["state"]
                # Track temperature violations
                T = float(state.T_reactor)
                if T > self.T_MAX - 10.0 or T < self.T_MIN:
                    self.episode_temp_violations.append(1)
                else:
                    self.episode_temp_violations.append(0)

                # Track pressure violations
                p = float(state.p)
                if p > self.P_NOM or p < self.P_MIN:
                    self.episode_pressure_violations.append(1)
                else:
                    self.episode_pressure_violations.append(0)

        # Log aggregated metrics every 1000 steps
        if self.n_calls % 1000 == 0 and len(self.episode_temp_violations) > 0:
            self.logger.record(
                "haber_bosch/temp_violation_rate",
                np.mean(self.episode_temp_violations[-1000:])
            )
            self.logger.record(
                "haber_bosch/pressure_violation_rate",
                np.mean(self.episode_pressure_violations[-1000:])
            )
        return True


def make_env(seed: int = 0, normalize_actions: bool = True, **env_kwargs):
    """Factory function to create environment instances."""
    def _init():
        env = HaberBoschEnv(**env_kwargs)
        if normalize_actions:
            env = NormalizedActionWrapper(env)
        return env
    return _init


def train(args: argparse.Namespace) -> None:
    """Main training function."""
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_haber_bosch_{timestamp}"

    # Setup directories
    tensorboard_log = Path(PATHS["tensorboard_log"]) / run_name
    checkpoint_dir = Path(PATHS["checkpoint_dir"]) / run_name
    best_model_dir = Path(PATHS["best_model_dir"]) / run_name
    eval_log_dir = Path(PATHS["eval_log_dir"]) / run_name

    for dir_path in [tensorboard_log, checkpoint_dir, best_model_dir, eval_log_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Create vectorized training environment
    n_envs = args.n_envs or TRAINING_CONFIG["n_envs"]
    train_env = make_vec_env(
        make_env(seed=args.seed, **ENV_CONFIG),
        n_envs=n_envs,
        seed=args.seed,
    )

    # Wrap with VecNormalize for observation and reward normalization
    train_env = VecNormalize(
        train_env,
        norm_obs=NORMALIZE_CONFIG["norm_obs"],
        norm_reward=NORMALIZE_CONFIG["norm_reward"],
        clip_obs=NORMALIZE_CONFIG["clip_obs"],
        clip_reward=NORMALIZE_CONFIG["clip_reward"],
        gamma=NORMALIZE_CONFIG["gamma"],
    )

    # Create separate evaluation environment (with same normalization)
    eval_env = make_vec_env(
        make_env(seed=args.seed + 1000, **ENV_CONFIG),
        n_envs=1,
        seed=args.seed + 1000,
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=NORMALIZE_CONFIG["norm_obs"],
        norm_reward=False,  # Don't normalize rewards during evaluation
        clip_obs=NORMALIZE_CONFIG["clip_obs"],
        gamma=NORMALIZE_CONFIG["gamma"],
        training=False,  # Don't update statistics during evaluation
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max((args.checkpoint_freq or TRAINING_CONFIG["checkpoint_freq"]) // n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_haber_bosch",
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=max((args.eval_freq or TRAINING_CONFIG["eval_freq"]) // n_envs, 1),
        n_eval_episodes=TRAINING_CONFIG["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    metrics_callback = HaberBoschMetricsCallback(verbose=args.verbose)

    callbacks = CallbackList([checkpoint_callback, eval_callback, metrics_callback])

    # Build PPO config with CLI overrides
    ppo_kwargs = PPO_CONFIG.copy()
    if args.learning_rate:
        ppo_kwargs["learning_rate"] = args.learning_rate
    if args.batch_size:
        ppo_kwargs["batch_size"] = args.batch_size
    ppo_kwargs["verbose"] = args.verbose
    ppo_kwargs["tensorboard_log"] = str(tensorboard_log)

    # Create PPO model using SBX
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        seed=args.seed,
        **ppo_kwargs,
    )

    # Print configuration
    print("=" * 60)
    print("Training PPO on Haber-Bosch Environment")
    print("=" * 60)
    print(f"Run name: {run_name}")
    print(f"Total timesteps: {args.total_timesteps or TRAINING_CONFIG['total_timesteps']:,}")
    print(f"Number of envs: {n_envs}")
    print(f"Learning rate: {ppo_kwargs['learning_rate']}")
    print(f"Batch size: {ppo_kwargs['batch_size']}")
    print(f"TensorBoard: {tensorboard_log}")
    print("=" * 60)

    # Train
    model.learn(
        total_timesteps=args.total_timesteps or TRAINING_CONFIG["total_timesteps"],
        callback=callbacks,
        log_interval=TRAINING_CONFIG["log_interval"],
        progress_bar=True,
    )

    # Save final model and VecNormalize statistics
    final_model_path = best_model_dir / "final_model"
    model.save(str(final_model_path))
    train_env.save(str(best_model_dir / "vecnormalize.pkl"))

    print("=" * 60)
    print(f"Training complete!")
    print(f"Model saved to: {final_model_path}")
    print(f"VecNormalize saved to: {best_model_dir / 'vecnormalize.pkl'}")
    print("=" * 60)

    # Cleanup
    train_env.close()
    eval_env.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO on Haber-Bosch environment"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=None,
        help=f"Total training timesteps (default: {TRAINING_CONFIG['total_timesteps']:,})"
    )
    parser.add_argument(
        "--n-envs", type=int, default=None,
        help=f"Number of parallel environments (default: {TRAINING_CONFIG['n_envs']})"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help=f"Learning rate (default: {PPO_CONFIG['learning_rate']})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=f"Batch size (default: {PPO_CONFIG['batch_size']})"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=None,
        help=f"Evaluation frequency in timesteps (default: {TRAINING_CONFIG['eval_freq']:,})"
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=None,
        help=f"Checkpoint frequency in timesteps (default: {TRAINING_CONFIG['checkpoint_freq']:,})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        help="Verbosity level 0, 1, or 2 (default: 1)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
