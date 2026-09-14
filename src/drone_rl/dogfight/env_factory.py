"""Dogfight env / VecEnv kurulum yardimcilari - v8.

v7'den gelen iyilestirmeler korundu:
  * load_opponent_controller() VecNormalize'i yuklemek icin gercek bir
    DogfightEnv (2 JSBSim FDM'i) kurmuyor; sadece observation/action
    space'i eslesen sahte bir env kullaniyor.
  * Ayni (model, vecnorm) ciftinin denetleyicisi onbellekten donuyor.

v8'de eklenenler:
  * _DummyObsEnv artik OBS_DIM'i dogfight_env'den ALIYOR - gozlem
    boyutu degistiginde burayi guncellemeyi unutma riski kalmadi.
  * load_opponent_controller() gozlem boyutu uyusmazligini ACIK bir
    hata mesajiyla bildiriyor (v7 checkpoint'leri 17 boyutlu, v8 ise
    30 boyutlu - eski havuz kullanilirsa sessizce sacmalamak yerine
    net bir hata verir).
  * Stage A artik kolaydan-zora bir rakip mufredati kullaniyor
    (OpponentCurriculum): hover -> daire -> kappa-PPG saf takip.
  * SubprocVecEnv destegi (T4).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from drone_rl.dogfight.dogfight_env import (
    OBS_DIM, ACT_DIM, DogfightEnv, OpponentCurriculum,
    PPOOpponentController, NormalizerStats,
)
from drone_rl.dogfight.checkpoint_pool import CheckpointPool


class _DummyObsEnv(gym.Env):
    """VecNormalize.load() icin JSBSim GEREKTIRMEYEN yer tutucu env."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, True, False, {}


def make_dummy_vecnorm_env():
    return DummyVecEnv([lambda: Monitor(_DummyObsEnv())])


_controller_cache: dict = {}


def load_opponent_controller(model_path: str, vecnorm_path: str,
                             deterministic: bool = False):
    cache_key = (model_path, vecnorm_path, bool(deterministic))
    if cache_key in _controller_cache:
        return _controller_cache[cache_key]

    from stable_baselines3 import PPO
    vecnorm = VecNormalize.load(vecnorm_path, make_dummy_vecnorm_env())
    stats = NormalizerStats(vecnorm)

    if stats.obs_dim != OBS_DIM:
        raise ValueError(
            f"Gozlem boyutu uyusmuyor: checkpoint {stats.obs_dim} boyutlu, "
            f"bu surum {OBS_DIM} boyutlu bekliyor.\n"
            f"  -> {vecnorm_path}\n"
            f"v8'de gozlem uzayi genisletildi (isaretli govde-eksen LOS, "
            f"govde hizlari, rakip hizi, HP vb.). ESKI havuz ve "
            f"checkpoint'ler GECERSIZDIR: havuz klasorunu silip Stage A'yi "
            f"sifirdan calistirin."
        )

    model = PPO.load(model_path, device="cpu")
    controller = PPOOpponentController(model, stats, deterministic=deterministic)
    _controller_cache[cache_key] = controller
    return controller


def make_dogfight_env(env_cfg, stage: str = "a", pool_dir: str = None,
                      fixed_opponent=None, deterministic_opponent=None) -> DogfightEnv:
    det = env_cfg.opponent_deterministic if deterministic_opponent is None else deterministic_opponent

    if fixed_opponent is not None:
        controller = load_opponent_controller(*fixed_opponent, deterministic=det)
        return DogfightEnv(env_cfg, opponent_controller=controller)

    if stage == "b" and pool_dir:
        pool = CheckpointPool(pool_dir)
        return DogfightEnv(
            env_cfg,
            opponent_pool=pool,
            opponent_latest_prob=env_cfg.opponent_latest_prob,
        )

    # Stage A: kolaydan zora mufredat
    return DogfightEnv(env_cfg, opponent_curriculum=OpponentCurriculum(env_cfg))


def make_dogfight_training_vec_env(env_cfg, n_envs: int, stage: str, pool_dir: str,
                                   training: bool, norm_reward: bool,
                                   clip_obs: float = 10.0, clip_reward: float = 10.0,
                                   vec: str = "auto", seed=None,
                                   vecnormalize_path: str = None):
    """DUZELTME (resume destegi): vecnormalize_path verilirse, YENI bir
    VecNormalize olusturmak yerine diskteki kaydedilmis normalizasyon
    istatistikleri (mean/var vb.) YUKLENIR. Bu olmadan resume edilen bir
    egitim, sifirdan sifirlanmis (yanlis olcekli) bir gozlem
    normalizasyonuyla devam eder - bu da modelin ogrendigi politikayla
    UYUMSUZ girdi dagilimina yol acar (fiilen egitim bozulur)."""
    def _make(rank):
        def _init():
            env = make_dogfight_env(env_cfg, stage=stage, pool_dir=pool_dir)
            if seed is not None:
                env.reset(seed=int(seed) + rank)
            return Monitor(env)
        return _init

    fns = [_make(i) for i in range(n_envs)]

    if vec == "auto":
        vec = "subproc" if n_envs > 1 else "dummy"

    if vec == "subproc" and n_envs > 1:
        try:
            venv = SubprocVecEnv(fns, start_method="spawn")
        except Exception as e:  # Colab/Windows gibi ortamlarda guvenli geri donus
            print(f"[env_factory] SubprocVecEnv basarisiz ({e}); DummyVecEnv'e dusuluyor.")
            venv = DummyVecEnv(fns)
    else:
        venv = DummyVecEnv(fns)

    if vecnormalize_path:
        venv = VecNormalize.load(vecnormalize_path, venv)
        venv.training = training
        venv.norm_reward = norm_reward
        venv.clip_obs = clip_obs
        venv.clip_reward = clip_reward
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=norm_reward,
                            clip_obs=clip_obs, clip_reward=clip_reward, training=training)
    return venv
