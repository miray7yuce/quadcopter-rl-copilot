"""Egitim ve degerlendirme icin ortak F450 ortami/VecEnv kurulum yardimcilari.

Hem F450HoverEnv (hover gorevi) hem F450FlightEnv (irtifa+heading gorevi)
icin ayri fonksiyon setleri barindirir. Biri digerini etkilemez.
"""

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drone_rl.envs.f450_env import F450HoverEnv
from drone_rl.envs.f450_flight_env import F450FlightEnv
from drone_rl.config import EnvConfig, FlightEnvConfig


# ---------------------------------------------------------------------
# Hover gorevi (degismedi)
# ---------------------------------------------------------------------

def make_env(env_config: EnvConfig) -> F450HoverEnv:
    return F450HoverEnv(
        target_altitude_ft=env_config.target_altitude_ft,
        episode_seconds=env_config.episode_seconds,
        physics_hz=env_config.physics_hz,
        control_hz=env_config.control_hz,
        hover_throttle=env_config.hover_throttle,
        throttle_range=env_config.throttle_range,
        reward_alt_weight=env_config.reward_alt_weight,
        reward_tilt_weight=env_config.reward_tilt_weight,
        reward_spin_weight=env_config.reward_spin_weight,
        reward_jerk_weight=env_config.reward_jerk_weight,
        crash_penalty=env_config.crash_penalty,
        crash_min_alt_ft=env_config.crash_min_alt_ft,
        crash_max_alt_offset_ft=env_config.crash_max_alt_offset_ft,
        crash_max_tilt_rad=env_config.crash_max_tilt_rad,
    )


def make_training_vec_env(env_config: EnvConfig, n_envs: int, training: bool,
                           norm_reward: bool, clip_obs: float = 10.0):
    def _make():
        return Monitor(make_env(env_config))

    venv = DummyVecEnv([_make for _ in range(n_envs)])
    venv = VecNormalize(
        venv, norm_obs=True, norm_reward=norm_reward, clip_obs=clip_obs, training=training
    )
    return venv


def make_eval_vec_env(env_config: EnvConfig):
    return DummyVecEnv([lambda: make_env(env_config)])


# ---------------------------------------------------------------------
# Flight gorevi (YENI: hedef irtifa + hedef yon)
# ---------------------------------------------------------------------

def make_flight_env(flight_config: FlightEnvConfig) -> F450FlightEnv:
    return F450FlightEnv(
        target_altitude_min_ft=flight_config.target_altitude_min_ft,
        target_altitude_max_ft=flight_config.target_altitude_max_ft,
        target_speed_fps=flight_config.target_speed_fps,
        episode_seconds=flight_config.episode_seconds,
        physics_hz=flight_config.physics_hz,
        control_hz=flight_config.control_hz,
        hover_throttle=flight_config.hover_throttle,
        throttle_range=flight_config.throttle_range,
        reward_alt_weight=flight_config.reward_alt_weight,
        reward_heading_weight=flight_config.reward_heading_weight,
        reward_tilt_weight=flight_config.reward_tilt_weight,
        reward_spin_weight=flight_config.reward_spin_weight,
        reward_jerk_weight=flight_config.reward_jerk_weight,
        crash_penalty=flight_config.crash_penalty,
        crash_min_alt_ft=flight_config.crash_min_alt_ft,
        crash_max_alt_offset_ft=flight_config.crash_max_alt_offset_ft,
        crash_max_tilt_rad=flight_config.crash_max_tilt_rad,
    )


def make_flight_training_vec_env(flight_config: FlightEnvConfig, n_envs: int,
                                  training: bool, norm_reward: bool, clip_obs: float = 10.0):
    def _make():
        return Monitor(make_flight_env(flight_config))

    venv = DummyVecEnv([_make for _ in range(n_envs)])
    venv = VecNormalize(
        venv, norm_obs=True, norm_reward=norm_reward, clip_obs=clip_obs, training=training
    )
    return venv


def make_flight_eval_vec_env(flight_config: FlightEnvConfig):
    return DummyVecEnv([lambda: make_flight_env(flight_config)])
