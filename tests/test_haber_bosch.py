"""Unit tests for Haber-Bosch environment."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from environments import HaberBoschEnv, EnvState, step_jax
from environments.haber_bosch import (
    _equilibrium_constant_Kp,
    _mass_frac_to_mole_frac_NH3,
    _reaction_rate_mol_per_s,
    _dynamics,
    _observation,
    _reward,
    P_NOMINAL_PA,
    P_MIN_PA,
    T_CATALYST_MAX_K,
    T_CATALYST_MIN_K,
    T_FEED_K,
    M_NH3,
    M_H2,
    M_N2,
    R_GAS,
    V_LOOP_M3,
    Z_DEFAULT,
    X_DISS,
    N_DOT_IN_NOMINAL,
    M_H2_FEED_NOMINAL_KGS,
)

# Safe operating values for tests (within soft margins)
P_SAFE_MID = 126e5  # 126 bar (safe zone: 105-147 bar)
T_SAFE_MID = 723.0  # 450°C = 723K (safe zone: 643-773K, i.e., 370-500°C)

# Default previous action for tests (nominal operation)
DEFAULT_PREV_ACTION = jnp.array([1.0, P_NOMINAL_PA, 0.5], dtype=jnp.float32)


class TestEquilibriumConstant:
    """Tests for _equilibrium_constant_Kp."""

    def test_returns_positive(self):
        """Equilibrium constant should always be positive."""
        temps = jnp.array([623.0, 700.0, 793.0])  # K (350-520 C range)
        for T in temps:
            Kp = _equilibrium_constant_Kp(T)
            assert float(Kp) > 0, f"Kp should be positive at T={T}"

    def test_decreases_with_temperature(self):
        """Higher T should give lower Kp (exothermic reaction)."""
        T_low = jnp.array(623.0)  # 350 C
        T_high = jnp.array(793.0)  # 520 C
        Kp_low = _equilibrium_constant_Kp(T_low)
        Kp_high = _equilibrium_constant_Kp(T_high)
        assert float(Kp_low) > float(Kp_high), "Kp should decrease with temperature"

    def test_reasonable_magnitude(self):
        """Kp should be in a physically reasonable range."""
        T = jnp.array(723.0)  # 450 C - middle of operating range
        Kp = _equilibrium_constant_Kp(T)
        # Kp is in Pa^-1, typical values are 1e-8 to 1e-6 range
        assert 1e-12 < float(Kp) < 1e-3, f"Kp={Kp} seems unreasonable"


class TestMassFracToMoleFrac:
    """Tests for _mass_frac_to_mole_frac_NH3."""

    def test_zero_mass_frac(self):
        """Zero mass fraction should give zero mole fraction."""
        y = _mass_frac_to_mole_frac_NH3(jnp.array(0.0))
        assert float(y) == pytest.approx(0.0, abs=1e-10)

    def test_pure_nh3(self):
        """Mass fraction of 1 should give mole fraction of 1."""
        y = _mass_frac_to_mole_frac_NH3(jnp.array(1.0))
        assert float(y) == pytest.approx(1.0, abs=1e-6)

    def test_typical_values(self):
        """Typical operating mass fractions should give reasonable mole fractions."""
        # At typical outlet: w_NH3 ~ 0.12 (12% by mass)
        w = jnp.array(0.12)
        y = _mass_frac_to_mole_frac_NH3(w)
        # NH3 is heavier than H2/N2 mix, so mole frac < mass frac
        assert 0 < float(y) < float(w), "Mole fraction should be less than mass fraction for NH3"

    def test_monotonic(self):
        """Higher mass fraction should give higher mole fraction."""
        w_low = jnp.array(0.05)
        w_high = jnp.array(0.15)
        y_low = _mass_frac_to_mole_frac_NH3(w_low)
        y_high = _mass_frac_to_mole_frac_NH3(w_high)
        assert float(y_low) < float(y_high), "Mole fraction should increase with mass fraction"

    def test_clipping(self):
        """Output should be clipped to [0, 1]."""
        y_neg = _mass_frac_to_mole_frac_NH3(jnp.array(-0.1))
        y_over = _mass_frac_to_mole_frac_NH3(jnp.array(1.5))
        assert 0.0 <= float(y_neg) <= 1.0
        assert 0.0 <= float(y_over) <= 1.0


class TestReactionRate:
    """Tests for _reaction_rate_mol_per_s."""

    def test_returns_non_negative(self):
        """Reaction rate should never be negative."""
        p = jnp.array(150e5)  # 150 bar
        T = jnp.array(723.0)  # 450 C
        y_NH3 = jnp.array(0.1)
        M_loop = jnp.array(2.0)
        r = _reaction_rate_mol_per_s(p, T, y_NH3, M_loop)
        assert float(r) >= 0, "Reaction rate should be non-negative"

    def test_increases_with_pressure(self):
        """Higher pressure should give higher reaction rate."""
        T = jnp.array(723.0)
        y_NH3 = jnp.array(0.1)
        M_loop = jnp.array(2.0)
        r_low = _reaction_rate_mol_per_s(jnp.array(100e5), T, y_NH3, M_loop)
        r_high = _reaction_rate_mol_per_s(jnp.array(150e5), T, y_NH3, M_loop)
        assert float(r_high) > float(r_low), "Rate should increase with pressure"

    def test_near_equilibrium_low_rate(self):
        """At high NH3 concentration (near equilibrium), rate should be lower."""
        p = jnp.array(150e5)
        T = jnp.array(723.0)
        M_loop = jnp.array(2.0)
        r_low_nh3 = _reaction_rate_mol_per_s(p, T, jnp.array(0.05), M_loop)
        r_high_nh3 = _reaction_rate_mol_per_s(p, T, jnp.array(0.25), M_loop)
        assert float(r_low_nh3) > float(r_high_nh3), "Rate should decrease as NH3 increases"


class TestDynamics:
    """Tests for _dynamics."""

    @pytest.fixture
    def nominal_state(self):
        """Create a nominal operating state."""
        return EnvState(
            p=jnp.array(P_NOMINAL_PA),
            N_gas=jnp.array(P_NOMINAL_PA * V_LOOP_M3 / (Z_DEFAULT * R_GAS * T_FEED_K)),
            T_reactor=jnp.array(723.0),
            w_NH3_in=jnp.array(0.05),
            w_NH3_out=jnp.array(0.12),
            M_loop=jnp.array(2.0 * M_H2_FEED_NOMINAL_KGS),
            lambda_load=jnp.array(1.0),
            step=jnp.array(0),
            prev_action=DEFAULT_PREV_ACTION,
        )

    def test_step_increments(self, nominal_state):
        """Step counter should increment by 1."""
        action = jnp.array([1.0, P_NOMINAL_PA, 0.5])
        new_state = _dynamics(
            nominal_state, action, dt=1.0,
            V_loop=V_LOOP_M3, Z=Z_DEFAULT, X_diss=X_DISS,
            N_dot_in_nominal=N_DOT_IN_NOMINAL, k0=1e3, Ea=1e5,
        )
        assert int(new_state.step) == int(nominal_state.step) + 1

    def test_pressure_stays_bounded(self, nominal_state):
        """Pressure should stay within physical bounds."""
        action = jnp.array([1.0, P_NOMINAL_PA, 0.5])
        state = nominal_state
        for _ in range(100):
            state = _dynamics(
                state, action, dt=1.0,
                V_loop=V_LOOP_M3, Z=Z_DEFAULT, X_diss=X_DISS,
                N_dot_in_nominal=N_DOT_IN_NOMINAL, k0=1e3, Ea=1e5,
            )
        p = float(state.p)
        assert 0.5 * P_MIN_PA <= p <= 1.1 * P_NOMINAL_PA, f"Pressure {p} out of bounds"

    def test_temperature_stays_bounded(self, nominal_state):
        """Temperature should stay within catalyst limits."""
        action = jnp.array([1.0, P_NOMINAL_PA, 0.5])
        state = nominal_state
        for _ in range(100):
            state = _dynamics(
                state, action, dt=1.0,
                V_loop=V_LOOP_M3, Z=Z_DEFAULT, X_diss=X_DISS,
                N_dot_in_nominal=N_DOT_IN_NOMINAL, k0=1e3, Ea=1e5,
            )
        T = float(state.T_reactor)
        # Allow small floating point tolerance
        assert T_FEED_K - 0.01 <= T <= T_CATALYST_MAX_K + 0.01, f"Temperature {T} out of bounds"

    def test_mass_fractions_bounded(self, nominal_state):
        """Mass fractions should stay in [0, 1]."""
        action = jnp.array([0.5, 120e5, 0.7])
        state = nominal_state
        for _ in range(50):
            state = _dynamics(
                state, action, dt=1.0,
                V_loop=V_LOOP_M3, Z=Z_DEFAULT, X_diss=X_DISS,
                N_dot_in_nominal=N_DOT_IN_NOMINAL, k0=1e3, Ea=1e5,
            )
        assert 0 <= float(state.w_NH3_in) <= 1
        assert 0 <= float(state.w_NH3_out) <= 1

    def test_load_tracks_setpoint(self, nominal_state):
        """Load should eventually track the setpoint."""
        action = jnp.array([0.5, P_NOMINAL_PA, 0.5])  # 50% load setpoint
        state = nominal_state
        for _ in range(200):
            state = _dynamics(
                state, action, dt=1.0,
                V_loop=V_LOOP_M3, Z=Z_DEFAULT, X_diss=X_DISS,
                N_dot_in_nominal=N_DOT_IN_NOMINAL, k0=1e3, Ea=1e5,
            )
        assert float(state.lambda_load) == pytest.approx(0.5, abs=0.05)


class TestReward:
    """Tests for _reward function."""

    def test_production_is_positive_component(self):
        """With good conditions, production reward should be positive."""
        state = EnvState(
            p=jnp.array(P_SAFE_MID),  # Safe pressure (126 bar)
            N_gas=jnp.array(1e5),
            T_reactor=jnp.array(T_SAFE_MID),  # Safe temperature (450°C)
            w_NH3_in=jnp.array(0.05),
            w_NH3_out=jnp.array(0.15),  # Good conversion
            M_loop=jnp.array(2.0),
            lambda_load=jnp.array(1.0),
            step=jnp.array(0),
            prev_action=DEFAULT_PREV_ACTION,
        )
        action = DEFAULT_PREV_ACTION  # Same action = no smoothness penalty
        reward = _reward(state, action)
        # Reward should be positive when operating well in safe zone
        assert float(reward) > 0, "Reward should be positive under good conditions"

    def test_high_temperature_penalty(self):
        """Temperature above safe zone should incur penalty."""
        state_normal = EnvState(
            p=jnp.array(P_SAFE_MID),
            N_gas=jnp.array(1e5),
            T_reactor=jnp.array(T_SAFE_MID),  # Safe (450°C)
            w_NH3_in=jnp.array(0.05),
            w_NH3_out=jnp.array(0.12),
            M_loop=jnp.array(2.0),
            lambda_load=jnp.array(1.0),
            step=jnp.array(0),
            prev_action=DEFAULT_PREV_ACTION,
        )
        state_hot = state_normal._replace(T_reactor=jnp.array(T_CATALYST_MAX_K + 5.0))  # Above max
        action = DEFAULT_PREV_ACTION
        r_normal = _reward(state_normal, action)
        r_hot = _reward(state_hot, action)
        assert float(r_hot) < float(r_normal), "High temperature should reduce reward"

    def test_low_temperature_penalty(self):
        """Temperature below safe zone should incur penalty."""
        state_normal = EnvState(
            p=jnp.array(P_SAFE_MID),
            N_gas=jnp.array(1e5),
            T_reactor=jnp.array(T_SAFE_MID),  # Safe (450°C)
            w_NH3_in=jnp.array(0.05),
            w_NH3_out=jnp.array(0.12),
            M_loop=jnp.array(2.0),
            lambda_load=jnp.array(1.0),
            step=jnp.array(0),
            prev_action=DEFAULT_PREV_ACTION,
        )
        state_cold = state_normal._replace(T_reactor=jnp.array(T_CATALYST_MIN_K - 10.0))  # Below min
        action = DEFAULT_PREV_ACTION
        r_normal = _reward(state_normal, action)
        r_cold = _reward(state_cold, action)
        assert float(r_cold) < float(r_normal), "Low temperature should reduce reward"

    def test_overpressure_penalty(self):
        """Pressure above nominal should incur penalty."""
        state_normal = EnvState(
            p=jnp.array(P_SAFE_MID),  # Safe (126 bar)
            N_gas=jnp.array(1e5),
            T_reactor=jnp.array(T_SAFE_MID),
            w_NH3_in=jnp.array(0.05),
            w_NH3_out=jnp.array(0.12),
            M_loop=jnp.array(2.0),
            lambda_load=jnp.array(1.0),
            step=jnp.array(0),
            prev_action=DEFAULT_PREV_ACTION,
        )
        state_overpressure = state_normal._replace(p=jnp.array(1.1 * P_NOMINAL_PA))  # 167 bar
        action = DEFAULT_PREV_ACTION
        r_normal = _reward(state_normal, action)
        r_over = _reward(state_overpressure, action)
        assert float(r_over) < float(r_normal), "Overpressure should reduce reward"

    def test_underpressure_penalty(self):
        """Pressure below minimum should incur penalty."""
        state_normal = EnvState(
            p=jnp.array(P_SAFE_MID),  # Safe (126 bar)
            N_gas=jnp.array(1e5),
            T_reactor=jnp.array(T_SAFE_MID),
            w_NH3_in=jnp.array(0.05),
            w_NH3_out=jnp.array(0.12),
            M_loop=jnp.array(2.0),
            lambda_load=jnp.array(1.0),
            step=jnp.array(0),
            prev_action=DEFAULT_PREV_ACTION,
        )
        state_underpressure = state_normal._replace(p=jnp.array(0.9 * P_MIN_PA))  # 90 bar
        action = DEFAULT_PREV_ACTION
        r_normal = _reward(state_normal, action)
        r_under = _reward(state_underpressure, action)
        assert float(r_under) < float(r_normal), "Underpressure should reduce reward"

    def test_action_smoothness_penalty(self):
        """Large action changes should incur penalty."""
        # State with previous action at safe values
        prev_act = jnp.array([0.5, P_SAFE_MID, 0.5])
        state = EnvState(
            p=jnp.array(P_SAFE_MID),
            N_gas=jnp.array(1e5),
            T_reactor=jnp.array(T_SAFE_MID),
            w_NH3_in=jnp.array(0.05),
            w_NH3_out=jnp.array(0.12),
            M_loop=jnp.array(2.0),
            lambda_load=jnp.array(1.0),
            step=jnp.array(0),
            prev_action=prev_act,
        )
        # Same action as previous - no smoothness penalty
        action_same = prev_act
        # Large action change - should incur penalty
        action_different = jnp.array([0.1, P_MIN_PA, 1.0])

        r_same = _reward(state, action_same)
        r_different = _reward(state, action_different)

        # Same action should have higher reward (no smoothness penalty)
        assert float(r_same) > float(r_different), "Large action changes should reduce reward"


class TestObservation:
    """Tests for _observation function."""

    def test_observation_shape(self):
        """Observation should have shape (8,)."""
        state = EnvState(
            p=jnp.array(P_NOMINAL_PA),
            N_gas=jnp.array(1e5),
            T_reactor=jnp.array(723.0),
            w_NH3_in=jnp.array(0.05),
            w_NH3_out=jnp.array(0.12),
            M_loop=jnp.array(2.0),
            lambda_load=jnp.array(1.0),
            step=jnp.array(0),
            prev_action=DEFAULT_PREV_ACTION,
        )
        obs = _observation(state)
        assert obs.shape == (8,), f"Expected shape (8,), got {obs.shape}"

    def test_observation_dtype(self):
        """Observation should be float32."""
        state = EnvState(
            p=jnp.array(P_NOMINAL_PA),
            N_gas=jnp.array(1e5),
            T_reactor=jnp.array(723.0),
            w_NH3_in=jnp.array(0.05),
            w_NH3_out=jnp.array(0.12),
            M_loop=jnp.array(2.0),
            lambda_load=jnp.array(1.0),
            step=jnp.array(0),
            prev_action=DEFAULT_PREV_ACTION,
        )
        obs = _observation(state)
        assert obs.dtype == jnp.float32, f"Expected float32, got {obs.dtype}"


class TestHaberBoschEnv:
    """Tests for the Gymnasium environment."""

    def test_reset_returns_valid_obs(self):
        """Reset should return observation matching observation space."""
        env = HaberBoschEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)

    def test_step_returns_correct_tuple(self):
        """Step should return (obs, reward, terminated, truncated, info)."""
        env = HaberBoschEnv(max_steps=100)
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_terminates_at_max_steps(self):
        """Environment should terminate after max_steps."""
        env = HaberBoschEnv(max_steps=10)
        obs, _ = env.reset(seed=42)
        for i in range(15):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                assert i >= 9, f"Terminated too early at step {i}"
                break
        assert terminated, "Should have terminated by now"

    def test_action_space_valid(self):
        """Action space should have correct bounds."""
        env = HaberBoschEnv()
        low = env.action_space.low
        high = env.action_space.high
        assert low[0] == pytest.approx(0.1)  # lambda min
        assert high[0] == pytest.approx(1.0)  # lambda max
        assert low[1] == pytest.approx(P_MIN_PA)  # pressure min
        assert high[1] == pytest.approx(P_NOMINAL_PA)  # pressure max


class TestStepJax:
    """Tests for pure JAX step function."""

    def test_step_jax_returns_correct_types(self):
        """step_jax should return (state, obs, reward)."""
        state = EnvState(
            p=jnp.array(P_NOMINAL_PA),
            N_gas=jnp.array(1e5),
            T_reactor=jnp.array(723.0),
            w_NH3_in=jnp.array(0.05),
            w_NH3_out=jnp.array(0.12),
            M_loop=jnp.array(2.0),
            lambda_load=jnp.array(1.0),
            step=jnp.array(0),
            prev_action=DEFAULT_PREV_ACTION,
        )
        action = jnp.array([0.8, 130e5, 0.7])
        new_state, obs, reward = step_jax(state, action)
        assert isinstance(new_state, EnvState)
        assert obs.shape == (8,)
        assert reward.shape == ()

    def test_step_jax_is_jittable(self):
        """step_jax should be JIT-compilable."""
        state = EnvState(
            p=jnp.array(P_NOMINAL_PA),
            N_gas=jnp.array(1e5),
            T_reactor=jnp.array(723.0),
            w_NH3_in=jnp.array(0.05),
            w_NH3_out=jnp.array(0.12),
            M_loop=jnp.array(2.0),
            lambda_load=jnp.array(1.0),
            step=jnp.array(0),
            prev_action=DEFAULT_PREV_ACTION,
        )
        action = jnp.array([0.8, 130e5, 0.7])
        jit_step = jax.jit(step_jax)
        new_state, obs, reward = jit_step(state, action)
        assert isinstance(new_state, EnvState)

    def test_step_jax_is_vmappable(self):
        """step_jax should be vmap-able for batch processing."""
        # Create batch of states
        batch_size = 4
        states = EnvState(
            p=jnp.ones(batch_size) * P_NOMINAL_PA,
            N_gas=jnp.ones(batch_size) * 1e5,
            T_reactor=jnp.ones(batch_size) * 723.0,
            w_NH3_in=jnp.ones(batch_size) * 0.05,
            w_NH3_out=jnp.ones(batch_size) * 0.12,
            M_loop=jnp.ones(batch_size) * 2.0,
            lambda_load=jnp.ones(batch_size) * 1.0,
            step=jnp.zeros(batch_size, dtype=jnp.int32),
            prev_action=jnp.tile(DEFAULT_PREV_ACTION, (batch_size, 1)),
        )
        actions = jnp.tile(jnp.array([[0.8, 130e5, 0.7]]), (batch_size, 1))

        vmap_step = jax.vmap(step_jax)
        new_states, obs_batch, rewards = vmap_step(states, actions)

        assert new_states.p.shape == (batch_size,)
        assert obs_batch.shape == (batch_size, 8)
        assert rewards.shape == (batch_size,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
