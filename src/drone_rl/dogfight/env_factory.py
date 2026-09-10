"""Dogfight env/VecEnv kurulum yardimcilari.

DUZELTME: load_opponent_controller() eskiden HER cagrildiginda gercek
bir DogfightEnv (2 JSBSim FDM'i) kuruyordu - sadece VecNormalize'i
yuklemek icin. "Her episode'da havuzdan yeniden ornekleme" ozelligiyle
birlikte bu, saniyede onlarca kez GEREKSIZ JSBSim motoru kurulup
atilmasina yol aciyordu - hem logu spam'liyor hem egitimi ciddi
yavaslatiyordu. Simdi:
  1. VecNormalize icin GERCEK JSBSim GEREKTIRMEYEN, sadece observation/
     action_space'i eslesen SAHTE bir env kullaniliyor.
  2. Ayni (model_path, vecnorm_path) tekrar istenirse ONBELLEKTEN
     donduruluyor (ki bu COK sik oluyor, %70 ihtimalle hep en guncel
     checkpoint isteniyor).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drone_rl.dogfight.dogfight_env import (
    DogfightEnv, ScriptedCircleOpponent, PPOOpponentController, NormalizerStats,
)
from drone_rl.dogfight.checkpoint_pool import CheckpointPool


class _DummyObsEnv(gym.Env):
    """VecNormalize.load() icin JSBSim GEREKTIRMEYEN, sadece observation/
    action_space uyumlu sahte bir env. Gercek fizik hic calismiyor -
    tek amaci VecNormalize'in bir Venv'e 'baglanabilmesi' icin bir
    yer tutucu olmak."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.zeros(17, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(17, dtype=np.float32), 0.0, True, False, {}


# YENI: ayni (model_path, vecnorm_path) tekrar istenirse tekrar
# yuklemek yerine onbellekten donduruluyor.
_controller_cache: dict = {}


def load_opponent_controller(model_path: str, vecnorm_path: str):
    cache_key = (model_path, vecnorm_path)
    if cache_key in _controller_cache:
        return _controller_cache[cache_key]

    from stable_baselines3 import PPO
    dummy_venv = DummyVecEnv([lambda: Monitor(_DummyObsEnv())])
    vecnorm = VecNormalize.load(vecnorm_path, dummy_venv)
    stats = NormalizerStats(vecnorm)
    model = PPO.load(model_path, device="cpu")

    controller = PPOOpponentController(model, stats)
    _controller_cache[cache_key] = controller
    return controller


def make_dogfight_env(env_cfg, stage: str = "a", pool_dir: str = None,
                       fixed_opponent=None) -> DogfightEnv:
    if fixed_opponent is not None:
        controller = load_opponent_controller(*fixed_opponent)
        return DogfightEnv(env_cfg, opponent_controller=controller)

    if stage == "b" and pool_dir:
        pool = CheckpointPool(pool_dir)
        return DogfightEnv(
            env_cfg,
            opponent_controller=ScriptedCircleOpponent(),
            opponent_pool=pool,
            opponent_latest_prob=env_cfg.opponent_latest_prob,
        )

    return DogfightEnv(env_cfg, opponent_controller=ScriptedCircleOpponent())


def make_dogfight_training_vec_env(env_cfg, n_envs: int, stage: str, pool_dir: str,
                                    training: bool, norm_reward: bool, clip_obs: float = 10.0):
    def _make():
        return Monitor(make_dogfight_env(env_cfg, stage=stage, pool_dir=pool_dir))

    venv = DummyVecEnv([_make for _ in range(n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=norm_reward,
                         clip_obs=clip_obs, training=training)
    return venv
