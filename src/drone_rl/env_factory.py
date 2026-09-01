"""Egitim ve degerlendirme icin ortak F450HoverEnv/VecEnv kurulum yardimcilari.

train.py ve evaluate.py'de tekrarlanan ortam kurulum mantigini
tekillestiriyoruz.
"""

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drone_rl.envs.f450_env import F450HoverEnv
from drone_rl.config import EnvConfig


def make_env(env_config: EnvConfig) -> F450HoverEnv:
    """Tek bir F450HoverEnv olusturur (Monitor sarmadan)."""
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
    """Egitim/EvalCallback icin Monitor + DummyVecEnv + VecNormalize sarilmis
    ortam. train.py'nin hem egitim hem eval_env'i icin kullanilir.

    training=True  + norm_reward=True  -> egitim ortami
    training=False + norm_reward=False -> EvalCallback icin eval ortami
    """
    def _make():
        return Monitor(make_env(env_config))

    venv = DummyVecEnv([_make for _ in range(n_envs)])
    venv = VecNormalize(
        venv, norm_obs=True, norm_reward=norm_reward, clip_obs=clip_obs, training=training
    )
    return venv


def make_eval_vec_env(env_config: EnvConfig):
    """evaluate.py icin: TEK, Monitor'SUZ F450HoverEnv iceren DummyVecEnv.

    Monitor sarilmamasi bilincli bir tercih: evaluate.py, venv.envs[0]
    uzerinden dogrudan .fdm'e erisip JSBSim property'lerini okuyor.
    Monitor sarsaydi venv.envs[0] bir Monitor nesnesi olur, .fdm
    bulunamazdi (AttributeError).
    """
    return DummyVecEnv([lambda: make_env(env_config)])
