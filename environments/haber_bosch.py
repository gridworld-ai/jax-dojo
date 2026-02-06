"""
JAX environment for a simplified load-flexible Haber–Bosch process.

Based on: Fahr et al., "Dynamic simulation of a highly load-flexible
Haber–Bosch plant", International Journal of Hydrogen Energy 102 (2025) 1231–1242.

Physics implemented:
- Pressure dynamics (mass balance in synthesis loop): Eq. (2), (3), (7)
- Reaction stoichiometry: 0.5 N2 + 1.5 H2 ⇌ NH3
- Reaction rate and link to mass fractions: Eq. (8)
- Simplified kinetics approaching equilibrium (Temkin-style)
- Operating range: 10 – 100% load, 100 – 152 bar, catalyst temperature limits
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, NamedTuple

# -----------------------------------------------------------------------------
# Physical constants (SI)
# -----------------------------------------------------------------------------
R_GAS = 8.314462618  # J/(mol·K)
# Molar masses (kg/mol)
M_NH3 = 0.01703052
M_H2 = 0.00201588
M_N2 = 0.0280134
# Stoichiometry: 0.5 N2 + 1.5 H2 -> NH3  =>  nu_N2=-0.5, nu_H2=-1.5, nu_NH3=+1
# H2:N2 molar ratio in feed = 3:1
H2_N2_MOLAR_RATIO = 3.0

# Nominal operating point (from paper)
P_NOMINAL_PA = 152.0 * 1e5   # 152 bar
P_MIN_PA = 100.0 * 1e5       # 100 bar at part load
T_FEED_K = 273.15 + 140.0    # 140 °C reactor feed
T_CATALYST_MAX_K = 273.15 + 520.0  # 520 °C max catalyst (paper)
T_CATALYST_MIN_K = 273.15 + 350.0  # 350 °C min to avoid blow-out (paper)
# Nominal H2 feed (kg/s) – reference for load λ = Ṁ_H2_feed / Ṁ_H2_feed_nominal
M_H2_FEED_NOMINAL_KGS = 1.0
# Molar feed rate at nominal load: n_H2 + n_N2 = (1/M_H2 + 1/(3*M_H2)) * M_H2_feed = (4/3) * M_H2_feed / M_H2
N_DOT_IN_NOMINAL = (4.0 / 3.0) * M_H2_FEED_NOMINAL_KGS / M_H2  # mol/s

# Loop volume (m³) – effective gas holdup in synthesis loop (typical order)
V_LOOP_M3 = 50.0
# Compressibility (simplified: Z ≈ 1 at these conditions, slightly < 1 for real gas)
Z_DEFAULT = 0.95
# Dissolved gas loading in liquid NH3 (paper: X_diss small)
X_DISS = 0.01

# Default time step (s)
DT_DEFAULT = 1.0


class EnvState(NamedTuple):
    """State of the Haber–Bosch synthesis loop (JAX-friendly)."""
    # p: loop pressure (Pa)
    p: jnp.ndarray
    # N_gas: total moles of gas in loop (mol)
    N_gas: jnp.ndarray
    # T_reactor: representative reactor temperature (K), e.g. bed outlet
    T_reactor: jnp.ndarray
    # w_NH3_in: ammonia mass fraction at reactor inlet
    w_NH3_in: jnp.ndarray
    # w_NH3_out: ammonia mass fraction at reactor outlet
    w_NH3_out: jnp.ndarray
    # M_loop: mass flow through reactor (kg/s)
    M_loop: jnp.ndarray
    # lambda_load: current load (0.1 to 1.0)
    lambda_load: jnp.ndarray
    # time step (integer)
    step: jnp.ndarray
    # prev_action: previous action for smoothness penalty [lambda_sp, p_sp, valve]
    prev_action: jnp.ndarray


def _equilibrium_constant_Kp(T: jnp.ndarray) -> jnp.ndarray:
    """Equilibrium constant K_p for NH3 synthesis (bar^-1).
    K_p = p_NH3 / (p_N2^0.5 * p_H2^1.5) at equilibrium.
    Simplified correlation for 400–500 °C range (e.g. log10(K_p) form)."""
    # Approximate log10(K_p) from literature (T in K)
    log10_Kp = (
        2.689
        - 2.691 * jnp.log10(T)
        - 5.519e-5 * T
        + 1.848e-7 * (T ** 2)
        + 2.994
    )
    K_p = 10.0 ** log10_Kp
    # K_p in bar^-1; convert to Pa^-1 for use with p in Pa: K_p_Pa = K_p / 1e5
    return jnp.maximum(K_p / 1e5, 1e-20)


def _mass_frac_to_mole_frac_NH3(w_NH3: jnp.ndarray) -> jnp.ndarray:
    """Convert NH3 mass fraction to NH3 mole fraction assuming rest is H2:N2 = 3:1 molar."""
    # w_NH3 = (y_NH3*M_NH3) / (y_NH3*M_NH3 + (1-y_NH3)*y_H2*M_H2 + (1-y_NH3)*y_N2*M_N2)
    # For (1-y_NH3): y_H2 = 3/4, y_N2 = 1/4 => (1-y_NH3)*(3/4*M_H2 + 1/4*M_N2)
    M_inert = 0.75 * M_H2 + 0.25 * M_N2
    # w_NH3 = y_NH3*M_NH3 / (y_NH3*M_NH3 + (1-y_NH3)*M_inert)
    # => w_NH3 * (y_NH3*M_NH3 + (1-y_NH3)*M_inert) = y_NH3*M_NH3
    # => y_NH3 = w_NH3*M_inert / (M_NH3 - w_NH3*(M_NH3 - M_inert))
    denom = M_NH3 - w_NH3 * (M_NH3 - M_inert)
    return jnp.where(
        jnp.abs(denom) < 1e-12,
        jnp.clip(w_NH3, 0.0, 1.0),
        jnp.clip(w_NH3 * M_inert / denom, 0.0, 1.0),
    )


def _reaction_rate_mol_per_s(
    p: jnp.ndarray,
    T: jnp.ndarray,
    y_NH3_avg: jnp.ndarray,
    M_loop: jnp.ndarray,
    k0: float = 1e3,
    Ea: float = 1.0e5,
) -> jnp.ndarray:
    """Net formation rate of NH3 in mol/s (lumped reactor).
    Temkin-style: r = k * (K_eq * p_N2^0.5 * p_H2^1.5 - p_NH3), clipped to r >= 0.
    Uses average composition (in/out) for partial pressures."""
    R = R_GAS
    y_H2 = (1.0 - y_NH3_avg) * 0.75
    y_N2 = (1.0 - y_NH3_avg) * 0.25
    p_NH3 = y_NH3_avg * p
    p_H2 = y_H2 * p
    p_N2 = y_N2 * p
    # Avoid zeros in denominators
    p_H2_safe = jnp.maximum(p_H2, 1.0)
    p_N2_safe = jnp.maximum(p_N2, 1.0)
    p_NH3_safe = jnp.maximum(p_NH3, 1.0)
    K_eq = _equilibrium_constant_Kp(T)
    # Driving force: forward - backward (Temkin-like)
    # r ∝ (p_N2^0.5 * p_H2^1.5 / p_NH3) * (1 - gamma/K_eq), gamma = p_NH3/(p_N2^0.5*p_H2^1.5)
    gamma = p_NH3_safe / (jnp.sqrt(p_N2_safe) * (p_H2_safe ** 1.5))
    driving = 1.0 - gamma / jnp.maximum(K_eq, 1e-20)
    k = k0 * jnp.exp(-Ea / (R * T))
    # Rate proportional to catalyst amount and flow (simplified: scale by M_loop^0.5 for residence time effect)
    r_raw = k * jnp.sqrt(p_N2_safe) * (p_H2_safe ** 1.5) / p_NH3_safe * driving
    r = jnp.maximum(r_raw, 0.0)
    # Scale to plausible mol/s (reactor “size” absorbed in k0)
    return r


def _dynamics(
    state: EnvState,
    action: jnp.ndarray,
    dt: float,
    V_loop: float,
    Z: float,
    X_diss: float,
    N_dot_in_nominal: float,
    k0: float,
    Ea: float,
) -> EnvState:
    """One discrete step of dynamics. Actions: [lambda_setpoint, pressure_setpoint, recycle_valve]."""
    lambda_sp = jnp.clip(action[0], 0.1, 1.0)
    p_sp = jnp.clip(action[1], P_MIN_PA, P_NOMINAL_PA)
    recycle_valve = jnp.clip(action[2], 0.01, 1.0)  # 0 = closed, 1 = open

    p, N_gas, T_reactor, w_NH3_in, w_NH3_out, M_loop, lambda_load, step, _ = state

    # Feed molar flow (paper Eq. (1): load = M_H2_feed / M_H2_feed_nominal)
    N_dot_in = lambda_sp * N_dot_in_nominal

    # Reaction rate: use average composition for kinetics
    y_NH3_in = _mass_frac_to_mole_frac_NH3(w_NH3_in)
    y_NH3_out = _mass_frac_to_mole_frac_NH3(w_NH3_out)
    y_NH3_avg = 0.5 * (y_NH3_in + y_NH3_out)
    xi_dot = _reaction_rate_mol_per_s(p, T_reactor, y_NH3_avg, M_loop, k0=k0, Ea=Ea)

    # Molar flow leaving as condensed NH3 (paper Eq. (6)): N_dot_out ≈ xi_dot * (1 + X_diss)
    N_dot_out = xi_dot * (1.0 + X_diss)

    # Mass balance (paper Eq. (3)): dN_gas/dt = N_dot_in - N_dot_out - xi_dot
    # With N_dot_out = xi_dot*(1+X_diss): dN_gas/dt = N_dot_in - xi_dot*(2 + X_diss)
    dN_gas_dt = N_dot_in - xi_dot * (2.0 + X_diss)
    N_gas_new = N_gas + dt * dN_gas_dt
    N_gas_new = jnp.maximum(N_gas_new, 1e3)

    # Pressure (paper Eq. (2)): p = Z * N * R * T / V
    p_new = Z * R_GAS * T_reactor * N_gas_new / V_loop
    p_new = jnp.clip(p_new, 0.5 * P_MIN_PA, 1.1 * P_NOMINAL_PA)

    # Pressure controller: adjust M_loop via recycle valve (paper: control pressure by recycle flow)
    # Simplified: M_loop setpoint from valve and pressure error. Higher p than setpoint -> open valve -> more flow.
    p_error = p_sp - p_new
    M_loop_base = lambda_sp * M_H2_FEED_NOMINAL_KGS * (4.0 / 3.0) * (M_NH3 / M_H2)  # scale by load
    M_loop_new = M_loop_base * recycle_valve * (1.0 + 0.1 * jnp.tanh(p_error / 1e6))
    M_loop_new = jnp.maximum(M_loop_new, 0.01 * M_H2_FEED_NOMINAL_KGS)

    # w_NH3_out from reaction (paper Eq. (8)): xi_dot = M_loop/M_NH3 * (w_NH3_out - w_NH3_in)
    # => w_NH3_out = w_NH3_in + xi_dot * M_NH3 / M_loop
    delta_w = xi_dot * M_NH3 / jnp.maximum(M_loop_new, 1e-6)
    w_NH3_out_new = jnp.clip(w_NH3_in + delta_w, 0.0, 0.5)

    # Condenser recycles liquid; inlet w_NH3_in is from loop after mixing with feed (simplified: exponential smoothing)
    alpha = 0.1
    w_NH3_in_new = (1.0 - alpha) * w_NH3_in + alpha * (w_NH3_out_new * 0.5)  # some NH3 remains in loop

    # Reactor temperature: balance of reaction heat and cooling
    # Delta_H = -46.2 kJ/mol NH3 (exothermic)
    delta_H = -46.2e3  # J/mol
    cp_mix = 35.0  # J/(mol·K) approx

    # Heat generation from reaction
    Q_rxn = -delta_H * xi_dot  # J/s (positive = heating)

    # Heat removal through cooling system (controllable via valve position)
    # Higher valve = more recycle flow = more cooling via heat exchanger
    # T_setpoint represents the controller target (influenced by load and valve)
    T_setpoint = T_FEED_K + 200.0 + 100.0 * lambda_sp  # 340-440°C depending on load
    k_cool = 0.1 * (0.5 + recycle_valve)  # cooling rate: higher valve = more cooling
    Q_cool = k_cool * N_gas_new * cp_mix * (T_reactor - T_setpoint)

    # Net temperature change
    dT_dt = jnp.where(
        N_gas_new > 1.0,
        (Q_rxn - Q_cool) / (N_gas_new * cp_mix),
        0.0,
    )
    T_reactor_new = T_reactor + dt * dT_dt
    T_reactor_new = jnp.clip(T_reactor_new, T_FEED_K, T_CATALYST_MAX_K + 20.0)  # Allow slight overshoot

    # Load tracks setpoint with first-order lag
    lambda_new = (1.0 - 0.05) * lambda_load + 0.05 * lambda_sp
    lambda_new = jnp.clip(lambda_new, 0.1, 1.0)

    return EnvState(
        p=p_new,
        N_gas=N_gas_new,
        T_reactor=T_reactor_new,
        w_NH3_in=w_NH3_in_new,
        w_NH3_out=w_NH3_out_new,
        M_loop=M_loop_new,
        lambda_load=lambda_new,
        step=step + 1,
        prev_action=action,
    )


def _observation(state: EnvState) -> jnp.ndarray:
    """Observation vector for RL: normalized/scaled state."""
    p, N_gas, T_reactor, w_NH3_in, w_NH3_out, M_loop, lambda_load, step, _ = state
    return jnp.array([
        p / P_NOMINAL_PA,
        N_gas / 1e5,
        (T_reactor - T_FEED_K) / (T_CATALYST_MAX_K - T_FEED_K),
        w_NH3_in,
        w_NH3_out,
        M_loop / (M_H2_FEED_NOMINAL_KGS * 2.0),
        lambda_load,
        jnp.log1p(step) / 10.0,
    ], dtype=jnp.float32)


def _reward(state: EnvState, action: jnp.ndarray) -> jnp.ndarray:
    """Reward: production (NH3 formation) minus penalties for constraint violation and action changes.

    Reward design:
    - Production: scaled to ~1.0 at nominal operation
    - Temperature: soft margin penalty starting 20K before limits
    - Pressure: soft margin penalty starting 5 bar before limits
    - Smoothness: penalize rapid action changes
    - Safe operation bonus: +0.5 when fully within bounds
    """
    p, _, T_reactor, w_NH3_in, w_NH3_out, M_loop, lambda_load, _, prev_action = state
    xi_dot = M_loop / M_NH3 * (state.w_NH3_out - state.w_NH3_in)
    xi_dot = jnp.maximum(xi_dot, 0.0)
    production_raw = xi_dot * M_NH3  # kg/s NH3
    # Scale production so nominal ~0.05 kg/s -> reward ~1.0
    production = production_raw * 20.0

    # --- Temperature penalties (soft margin) ---
    # Safe zone: 370°C - 500°C (20K margins from hard limits 350-520°C)
    T_SAFE_LOW = T_CATALYST_MIN_K + 20.0   # 370°C
    T_SAFE_HIGH = T_CATALYST_MAX_K - 20.0  # 500°C

    # Quadratic penalty in margin zone, steep linear beyond hard limit
    temp_penalty = jnp.where(
        T_reactor > T_CATALYST_MAX_K,
        # Beyond hard limit: -10 base + steep linear
        -10.0 - 5.0 * (T_reactor - T_CATALYST_MAX_K) / 10.0,
        jnp.where(
            T_reactor > T_SAFE_HIGH,
            # In upper margin: quadratic ramp from 0 to -10
            -10.0 * ((T_reactor - T_SAFE_HIGH) / 20.0) ** 2,
            jnp.where(
                T_reactor < T_CATALYST_MIN_K,
                # Beyond low limit
                -5.0 - 2.0 * (T_CATALYST_MIN_K - T_reactor) / 10.0,
                jnp.where(
                    T_reactor < T_SAFE_LOW,
                    # In lower margin
                    -5.0 * ((T_SAFE_LOW - T_reactor) / 20.0) ** 2,
                    0.0,  # Safe zone
                ),
            ),
        ),
    )

    # --- Pressure penalties (soft margin) ---
    # Safe zone: 105-147 bar (5 bar margins from hard limits 100-152 bar)
    P_SAFE_LOW = P_MIN_PA + 5e5     # 105 bar
    P_SAFE_HIGH = P_NOMINAL_PA - 5e5  # 147 bar

    pressure_penalty = jnp.where(
        p > P_NOMINAL_PA,
        # Beyond upper limit: -5 base + steep linear
        -5.0 - 10.0 * (p - P_NOMINAL_PA) / P_NOMINAL_PA,
        jnp.where(
            p > P_SAFE_HIGH,
            # In upper margin: quadratic ramp from 0 to -5
            -5.0 * ((p - P_SAFE_HIGH) / 5e5) ** 2,
            jnp.where(
                p < P_MIN_PA,
                # Beyond low limit
                -5.0 - 10.0 * (P_MIN_PA - p) / P_MIN_PA,
                jnp.where(
                    p < P_SAFE_LOW,
                    # In lower margin
                    -5.0 * ((P_SAFE_LOW - p) / 5e5) ** 2,
                    0.0,  # Safe zone
                ),
            ),
        ),
    )

    # --- Safe operation bonus ---
    # Small positive reward for staying fully within safe bounds
    in_safe_temp = (T_reactor >= T_SAFE_LOW) & (T_reactor <= T_SAFE_HIGH)
    in_safe_pressure = (p >= P_SAFE_LOW) & (p <= P_SAFE_HIGH)
    safe_bonus = jnp.where(in_safe_temp & in_safe_pressure, 0.5, 0.0)

    # --- Action smoothness penalty ---
    # Normalize action changes by their respective ranges
    action_ranges = jnp.array([0.9, 52e5, 0.99])
    normalized_action_change = (action - prev_action) / action_ranges
    smoothness_penalty = -0.5 * jnp.sum(normalized_action_change ** 2)

    return production + temp_penalty + pressure_penalty + safe_bonus + smoothness_penalty


# -----------------------------------------------------------------------------
# Gymnasium environment
# -----------------------------------------------------------------------------


class HaberBoschEnv(gym.Env[EnvState, jnp.ndarray]):
    """JAX-friendly Gymnasium environment for the simplified Haber–Bosch process."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = DT_DEFAULT,
        V_loop_m3: float = V_LOOP_M3,
        Z: float = Z_DEFAULT,
        X_diss: float = X_DISS,
        N_dot_in_nominal: float = N_DOT_IN_NOMINAL,
        k0: float = 1e3,
        Ea: float = 1.0e5,
        max_steps: int = 1000,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._dt = dt
        self._V_loop = V_loop_m3
        self._Z = Z
        self._X_diss = X_diss
        self._N_dot_in_nominal = N_dot_in_nominal
        self._k0 = k0
        self._Ea = Ea
        self._max_steps = max_steps

        # State: p, N_gas, T_reactor, w_NH3_in, w_NH3_out, M_loop, lambda_load, step, prev_action
        # Actions: lambda_setpoint [0.1,1], pressure_setpoint [P_MIN, P_NOM], recycle_valve [0,1]
        self.action_space = spaces.Box(
            low=np.array([0.1, P_MIN_PA, 0.01], dtype=np.float32),
            high=np.array([1.0, P_NOMINAL_PA, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )

    def _initial_state(self, seed: int | None = None) -> EnvState:
        if seed is not None:
            key = jax.random.PRNGKey(seed)
        else:
            key = jax.random.PRNGKey(0)
        key, k1, k2 = jax.random.split(key, 3)
        # Start near nominal
        p0 = P_NOMINAL_PA * (0.95 + 0.1 * jax.random.uniform(k1))
        N_gas0 = p0 * self._V_loop / (self._Z * R_GAS * T_FEED_K)
        T0 = T_FEED_K + 50.0 + 20.0 * jax.random.uniform(k2)
        T0 = jnp.clip(T0, T_FEED_K, T_CATALYST_MAX_K - 20.0)
        # Initial "neutral" action: nominal load, nominal pressure, valve half-open
        initial_action = jnp.array([1.0, P_NOMINAL_PA, 0.5], dtype=jnp.float32)
        return EnvState(
            p=p0,
            N_gas=N_gas0,
            T_reactor=T0,
            w_NH3_in=jnp.array(0.05, dtype=jnp.float32),
            w_NH3_out=jnp.array(0.12, dtype=jnp.float32),
            M_loop=jnp.array(2.0 * M_H2_FEED_NOMINAL_KGS, dtype=jnp.float32),
            lambda_load=jnp.array(1.0, dtype=jnp.float32),
            step=jnp.array(0, dtype=jnp.int32),
            prev_action=initial_action,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        state = self._initial_state(seed=(seed if seed is not None else 0))
        self._state = state
        obs = _observation(state)
        return np.asarray(obs), {"state": state}

    def step(
        self,
        action: jnp.ndarray | np.ndarray,
    ) -> tuple[jnp.ndarray, float, bool, bool, dict[str, Any]]:
        if isinstance(action, np.ndarray):
            action = jnp.asarray(action)
        state = self._get_state()
        state_new = _dynamics(
            state,
            action,
            self._dt,
            self._V_loop,
            self._Z,
            self._X_diss,
            self._N_dot_in_nominal,
            self._k0,
            self._Ea,
        )
        self._state = state_new
        obs = _observation(state_new)
        reward = float(_reward(state_new, action))
        terminated = bool(state_new.step >= self._max_steps)
        truncated = False
        info = {"state": state_new}
        return np.asarray(obs), reward, terminated, truncated, info

    def _get_state(self) -> EnvState:
        if not hasattr(self, "_state"):
            self._state = self._initial_state(0)
        return self._state

    def set_state(self, state: EnvState) -> None:
        """Set internal state (e.g. for JAX jit with explicit state)."""
        self._state = state


# -----------------------------------------------------------------------------
# Pure JAX step (for jit / vmap)
# -----------------------------------------------------------------------------


def step_jax(
    state: EnvState,
    action: jnp.ndarray,
    dt: float = DT_DEFAULT,
    V_loop: float = V_LOOP_M3,
    Z: float = Z_DEFAULT,
    X_diss: float = X_DISS,
    N_dot_in_nominal: float = N_DOT_IN_NOMINAL,
    k0: float = 1e3,
    Ea: float = 1.0e5,
) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    """Pure JAX step: (state, action) -> (state_new, obs, reward). Jit- and vmap-friendly."""
    state_new = _dynamics(
        state, action, dt, V_loop, Z, X_diss, N_dot_in_nominal, k0, Ea
    )
    obs = _observation(state_new)
    reward = _reward(state_new, action)
    return state_new, obs, reward


class NormalizedActionWrapper(gym.ActionWrapper):
    """Wrapper that normalizes action space to [-1, 1].

    Maps:
        [-1, 1] -> [low, high] for each action dimension

    This improves training stability for RL algorithms like PPO/SAC.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._orig_low = env.action_space.low
        self._orig_high = env.action_space.high
        self._orig_range = self._orig_high - self._orig_low

        # New normalized action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized action [-1, 1] to original action space."""
        # Map [-1, 1] -> [0, 1] -> [low, high]
        normalized_01 = (action + 1.0) / 2.0
        return self._orig_low + normalized_01 * self._orig_range

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Convert original action to normalized [-1, 1] (for logging)."""
        normalized_01 = (action - self._orig_low) / self._orig_range
        return normalized_01 * 2.0 - 1.0


if __name__ == "__main__":
    # Quick smoke test
    env = HaberBoschEnv(dt=1.0, max_steps=100)
    obs, info = env.reset(seed=42)
    print("Initial obs shape:", obs.shape)
    action = jnp.array([0.8, 130e5, 0.7], dtype=jnp.float32)  # load 80%, 130 bar, valve 0.7
    obs, reward, term, trunc, info = env.step(action)
    print("After step - reward:", reward, "obs[0] (p_norm):", obs[0])
    # JAX step
    state = info["state"]
    state2, obs2, r2 = step_jax(state, jnp.array([0.5, 100e5, 0.5]))
    print("JAX step - reward:", float(r2))
    print("OK")