"""Tests for training infrastructure."""

import pytest
import numpy as np

from environments import HaberBoschEnv, NormalizedActionWrapper
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class TestEnvironmentCompatibility:
    """Verify environment compatibility with SB3/SBX."""

    def test_sb3_env_checker(self):
        """Environment should pass SB3 compatibility check."""
        env = HaberBoschEnv(max_steps=100)
        # This will raise if there are issues
        check_env(env, warn=True)

    def test_vec_env_wrapping(self):
        """Environment should work with DummyVecEnv."""
        env = DummyVecEnv([lambda: HaberBoschEnv(max_steps=100)])
        obs = env.reset()
        assert obs.shape == (1, 8)
        action = env.action_space.sample()
        obs, rewards, dones, infos = env.step([action])
        assert obs.shape == (1, 8)
        assert rewards.shape == (1,)
        env.close()

    def test_vecnormalize_wrapping(self):
        """Environment should work with VecNormalize."""
        env = DummyVecEnv([lambda: HaberBoschEnv(max_steps=100)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        obs = env.reset()
        assert obs.shape == (1, 8)
        # Run a few steps to build statistics
        for _ in range(10):
            action = env.action_space.sample()
            obs, rewards, dones, infos = env.step([action])
        # Check normalization is happening (obs should be roughly normalized)
        assert np.abs(obs).max() < 100, "Observations should be normalized"
        env.close()

    def test_multiple_envs(self):
        """Should support multiple parallel environments."""
        n_envs = 4
        env = DummyVecEnv([lambda: HaberBoschEnv(max_steps=100) for _ in range(n_envs)])
        obs = env.reset()
        assert obs.shape == (n_envs, 8)
        actions = [env.action_space.sample() for _ in range(n_envs)]
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (n_envs, 8)
        assert rewards.shape == (n_envs,)
        env.close()


class TestConfigImports:
    """Test that config files are importable."""

    def test_default_config_imports(self):
        """Default config should be importable."""
        from configs.default import (
            ENV_CONFIG,
            PPO_CONFIG,
            NORMALIZE_CONFIG,
            TRAINING_CONFIG,
            PATHS,
        )
        assert "max_steps" in ENV_CONFIG
        assert "learning_rate" in PPO_CONFIG
        assert "norm_obs" in NORMALIZE_CONFIG
        assert "total_timesteps" in TRAINING_CONFIG
        assert "tensorboard_log" in PATHS

    def test_env_config_valid(self):
        """ENV_CONFIG should create valid environment."""
        from configs.default import ENV_CONFIG
        env = HaberBoschEnv(**ENV_CONFIG)
        obs, info = env.reset()
        assert obs.shape == (8,)

    def test_ppo_config_keys(self):
        """PPO_CONFIG should have required keys."""
        from configs.default import PPO_CONFIG
        required_keys = [
            "learning_rate", "n_steps", "batch_size", "n_epochs",
            "gamma", "gae_lambda", "clip_range"
        ]
        for key in required_keys:
            assert key in PPO_CONFIG, f"Missing key: {key}"


class TestTrainingSmoke:
    """Smoke tests for training components (without full training)."""

    def test_make_vec_env(self):
        """make_vec_env should work with HaberBoschEnv."""
        from stable_baselines3.common.env_util import make_vec_env
        from configs.default import ENV_CONFIG

        def make_env():
            return HaberBoschEnv(**ENV_CONFIG)

        env = make_vec_env(make_env, n_envs=2)
        obs = env.reset()
        assert obs.shape == (2, 8)
        env.close()

    def test_vecnormalize_save_load(self, tmp_path):
        """VecNormalize should be saveable and loadable."""
        env = DummyVecEnv([lambda: HaberBoschEnv(max_steps=100)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        # Run some steps to build statistics
        obs = env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            env.step([action])

        # Save
        save_path = tmp_path / "vecnormalize.pkl"
        env.save(str(save_path))

        # Close and recreate
        env.close()
        new_env = DummyVecEnv([lambda: HaberBoschEnv(max_steps=100)])
        new_env = VecNormalize.load(str(save_path), new_env)

        # Should work
        obs = new_env.reset()
        assert obs.shape == (1, 8)
        new_env.close()


class TestNormalizedActionWrapper:
    """Test the normalized action wrapper."""

    def test_action_space_normalized(self):
        """Wrapped env should have [-1, 1] action space."""
        env = NormalizedActionWrapper(HaberBoschEnv(max_steps=100))
        assert env.action_space.low.tolist() == [-1.0, -1.0, -1.0]
        assert env.action_space.high.tolist() == [1.0, 1.0, 1.0]
        env.close()

    def test_action_mapping_boundaries(self):
        """Actions at boundaries should map correctly."""
        env = NormalizedActionWrapper(HaberBoschEnv(max_steps=100))

        # Test -1 maps to low bounds
        low_action = env.action(np.array([-1.0, -1.0, -1.0]))
        assert abs(low_action[0] - 0.1) < 0.01  # lambda min
        assert abs(low_action[1] - 100e5) < 1e4  # pressure min
        assert abs(low_action[2] - 0.01) < 0.01  # valve min

        # Test +1 maps to high bounds
        high_action = env.action(np.array([1.0, 1.0, 1.0]))
        assert abs(high_action[0] - 1.0) < 0.01  # lambda max
        assert abs(high_action[1] - 152e5) < 1e4  # pressure max
        assert abs(high_action[2] - 1.0) < 0.01  # valve max
        env.close()

    def test_sb3_compatible_with_wrapper(self):
        """Wrapped environment should pass SB3 check without warning."""
        from stable_baselines3.common.env_checker import check_env
        env = NormalizedActionWrapper(HaberBoschEnv(max_steps=100))
        # This should not raise any warnings about action space
        check_env(env, warn=True)
        env.close()

    def test_step_with_normalized_actions(self):
        """Should be able to step with normalized actions."""
        env = NormalizedActionWrapper(HaberBoschEnv(max_steps=100))
        obs, info = env.reset()

        # Step with normalized action [0, 0, 0] (middle of range)
        obs, reward, done, trunc, info = env.step(np.array([0.0, 0.0, 0.0]))
        assert obs.shape == (8,)
        assert isinstance(reward, float)
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
