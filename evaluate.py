#!/usr/bin/env python
"""Evaluate trained PPO model and visualize trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments import HaberBoschEnv
from environments.haber_bosch import (
    P_NOMINAL_PA, P_MIN_PA,
    T_CATALYST_MAX_K, T_CATALYST_MIN_K,
    M_NH3,
)
from configs.default import ENV_CONFIG


def load_model_and_env(
    model_path: str,
    vecnorm_path: str | None = None,
) -> tuple[PPO, DummyVecEnv | VecNormalize]:
    """Load trained model and VecNormalize statistics."""
    # Create evaluation environment
    env = DummyVecEnv([lambda: HaberBoschEnv(**ENV_CONFIG)])

    # Load VecNormalize if available
    if vecnorm_path and Path(vecnorm_path).exists():
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded VecNormalize from {vecnorm_path}")
    else:
        print("Warning: No VecNormalize found. Using raw environment.")

    # Load model
    model = PPO.load(model_path, env=env)
    print(f"Loaded model from {model_path}")

    return model, env


def run_episode(
    model: PPO,
    env: DummyVecEnv | VecNormalize,
    deterministic: bool = True,
) -> dict:
    """Run a single episode and collect trajectory data."""
    obs = env.reset()

    # Storage for trajectory
    trajectory = {
        "pressure": [],
        "temperature": [],
        "production": [],
        "lambda_load": [],
        "w_nh3_out": [],
        "rewards": [],
        "action_load": [],
        "action_pressure": [],
        "action_valve": [],
    }

    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, dones, info = env.step(action)
        done = dones[0]

        # Extract state from info
        state = info[0].get("state")
        if state is not None:
            trajectory["pressure"].append(float(state.p) / 1e5)  # bar
            trajectory["temperature"].append(float(state.T_reactor) - 273.15)  # Celsius
            trajectory["lambda_load"].append(float(state.lambda_load))
            trajectory["w_nh3_out"].append(float(state.w_NH3_out))

            # Calculate production (kg/s NH3)
            xi_dot = float(state.M_loop) / M_NH3 * (
                float(state.w_NH3_out) - float(state.w_NH3_in)
            )
            xi_dot = max(xi_dot, 0.0)
            production = xi_dot * M_NH3
            trajectory["production"].append(production)

        trajectory["rewards"].append(float(reward[0]))
        trajectory["action_load"].append(float(action[0][0]))
        trajectory["action_pressure"].append(float(action[0][1]) / 1e5)  # bar
        trajectory["action_valve"].append(float(action[0][2]))

        total_reward += float(reward[0])

    trajectory["total_reward"] = total_reward
    trajectory["episode_length"] = len(trajectory["rewards"])

    return trajectory


def plot_trajectory(trajectory: dict, save_path: str | None = None) -> None:
    """Create multi-panel plot of episode trajectory."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(
        f"Haber-Bosch Episode Trajectory | "
        f"Total Reward: {trajectory['total_reward']:.2f} | "
        f"Length: {trajectory['episode_length']}",
        fontsize=12,
    )

    timesteps = np.arange(len(trajectory["pressure"]))

    # Pressure
    ax = axes[0, 0]
    ax.plot(timesteps, trajectory["pressure"], "b-", linewidth=1.5)
    ax.axhline(y=P_NOMINAL_PA / 1e5, color="r", linestyle="--", label="Max (152 bar)")
    ax.axhline(y=P_MIN_PA / 1e5, color="orange", linestyle="--", label="Min (100 bar)")
    ax.set_ylabel("Pressure (bar)")
    ax.set_title("Loop Pressure")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Temperature
    ax = axes[0, 1]
    ax.plot(timesteps, trajectory["temperature"], "r-", linewidth=1.5)
    ax.axhline(y=T_CATALYST_MAX_K - 273.15, color="r", linestyle="--", label="Max (520C)")
    ax.axhline(y=T_CATALYST_MIN_K - 273.15, color="orange", linestyle="--", label="Min (350C)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Reactor Temperature")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Production
    ax = axes[1, 0]
    ax.plot(timesteps, trajectory["production"], "g-", linewidth=1.5)
    ax.set_ylabel("NH3 Production (kg/s)")
    ax.set_title("Ammonia Production Rate")
    ax.grid(True, alpha=0.3)

    # Load
    ax = axes[1, 1]
    ax.plot(timesteps, trajectory["lambda_load"], "m-", linewidth=1.5, label="Actual Load")
    ax.plot(timesteps, trajectory["action_load"], "c--", linewidth=1, alpha=0.7, label="Load Setpoint")
    ax.set_ylabel("Load (fraction)")
    ax.set_title("Load (Lambda)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Rewards
    ax = axes[2, 0]
    ax.plot(timesteps, trajectory["rewards"], "k-", linewidth=1, alpha=0.5)
    cumulative_avg = np.cumsum(trajectory["rewards"]) / (np.arange(len(trajectory["rewards"])) + 1)
    ax.plot(timesteps, cumulative_avg, "b-", linewidth=1.5, label="Running Average")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward")
    ax.set_title("Step Rewards")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Actions (valve and pressure setpoint)
    ax = axes[2, 1]
    ax.plot(timesteps, trajectory["action_valve"], "g-", linewidth=1, label="Valve")
    ax2 = ax.twinx()
    ax2.plot(timesteps, trajectory["action_pressure"], "b-", linewidth=1, alpha=0.7, label="P setpoint")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Valve Position", color="g")
    ax2.set_ylabel("Pressure Setpoint (bar)", color="b")
    ax.set_title("Control Actions")
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def evaluate_multiple_episodes(
    model: PPO,
    env: DummyVecEnv | VecNormalize,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> dict:
    """Evaluate over multiple episodes and compute statistics."""
    all_rewards = []
    all_lengths = []
    all_productions = []

    for i in range(n_episodes):
        trajectory = run_episode(model, env, deterministic=deterministic)
        all_rewards.append(trajectory["total_reward"])
        all_lengths.append(trajectory["episode_length"])
        all_productions.append(np.sum(trajectory["production"]))

        print(f"Episode {i+1}/{n_episodes}: "
              f"Reward={trajectory['total_reward']:.2f}, "
              f"Length={trajectory['episode_length']}, "
              f"Total Production={np.sum(trajectory['production']):.4f} kg NH3")

    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Mean Reward: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
    print(f"Mean Length: {np.mean(all_lengths):.1f} +/- {np.std(all_lengths):.1f}")
    print(f"Mean Total Production: {np.mean(all_productions):.4f} +/- {np.std(all_productions):.4f} kg NH3")

    return {
        "rewards": all_rewards,
        "lengths": all_lengths,
        "productions": all_productions,
    }


def main(args: argparse.Namespace) -> None:
    """Main evaluation entry point."""
    model, env = load_model_and_env(args.model_path, args.vecnorm_path)

    if args.n_episodes > 1:
        # Run multiple episode evaluation
        evaluate_multiple_episodes(
            model, env, n_episodes=args.n_episodes, deterministic=args.deterministic
        )

    # Run single episode for visualization
    print("\nRunning episode for visualization...")
    trajectory = run_episode(model, env, deterministic=args.deterministic)

    # Plot
    save_path = args.output if args.output else None
    plot_trajectory(trajectory, save_path=save_path)

    env.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO model on Haber-Bosch environment"
    )
    parser.add_argument(
        "model_path", type=str,
        help="Path to trained model (e.g., logs/best_model/best_model.zip)"
    )
    parser.add_argument(
        "--vecnorm-path", type=str, default=None,
        help="Path to VecNormalize statistics (e.g., logs/best_model/vecnormalize.pkl)"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=10,
        help="Number of episodes to evaluate (default: 10)"
    )
    parser.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use deterministic actions (default: True)"
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic actions"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path for trajectory plot (PNG)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.stochastic:
        args.deterministic = False
    main(args)
