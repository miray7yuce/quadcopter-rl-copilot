"""Dogfight env/VecEnv kurulum yardimcilari."""

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drone_rl.dogfight.dogfight_env import (
    DogfightEnv, ScriptedCircleOpponent, PPOOpponentController, NormalizerStats,
)
from drone_rl.dogfight.checkpoint_pool import CheckpointPool


def load_opponent_controller(model_path: str, vecnorm_path: str):
    from stable_baselines3 import PPO
    dummy_venv = DummyVecEnv([lambda: Monitor(DogfightEnv(_dummy_cfg()))])
    vecnorm = VecNormalize.load(vecnorm_path, dummy_venv)
    stats = NormalizerStats(vecnorm)
    model = PPO.load(model_path, device="cpu")
    return PPOOpponentController(model, stats)


def _dummy_cfg():
    from drone_rl.dogfight.config import DogfightEnvConfig
    return DogfightEnvConfig()


def make_dogfight_env(env_cfg, stage: str = "a", pool_dir: str = None,
                       fixed_opponent=None) -> DogfightEnv:
    """stage='a' -> ScriptedCircleOpponent.
    stage='b'   -> pool_dir'den (varsa) PPOOpponentController, yoksa
                   scripted'e fallback.
    fixed_opponent=(model_path, vecnorm_path) verilirse SABIT (hot-reload
    olmayan) bir opponent kullanilir - degerlendirme/promotion-check icin."""
    if fixed_opponent is not None:
        controller = load_opponent_controller(*fixed_opponent)
        env = DogfightEnv(env_cfg, opponent_controller=controller)
        return env

    if stage == "b" and pool_dir:
        pool = CheckpointPool(pool_dir)
        sampled = pool.sample()
        if sampled is not None:
            controller = load_opponent_controller(*sampled)
            env = DogfightEnv(env_cfg, opponent_controller=controller)
            return env

    env = DogfightEnv(env_cfg, opponent_controller=ScriptedCircleOpponent())
    return env


def make_dogfight_training_vec_env(env_cfg, n_envs: int, stage: str, pool_dir: str,
                                    training: bool, norm_reward: bool, clip_obs: float = 10.0):
    def _make():
        return Monitor(make_dogfight_env(env_cfg, stage=stage, pool_dir=pool_dir))

    venv = DummyVecEnv([_make for _ in range(n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=norm_reward,
                         clip_obs=clip_obs, training=training)
    return venv



