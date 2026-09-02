"""Merkezi config yukleme. YAML dosyasindaki degerleri dataclass'lara donusturur.

configs/ppo_hover.yaml gibi bir dosyayi okuyup Config nesnesine cevirir.
--config verilmezse (path=None), kod-ici varsayilan degerlerle bir Config
dondurulur - bu varsayilanlar, refactor oncesi f450_env.py/train.py'de
hardcoded olan degerlerle AYNIDIR, yani --config kullanilmadigi surece
davranis degismez.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List
import yaml


@dataclass
class EnvConfig:
    target_altitude_ft: float = 30.0
    episode_seconds: float = 20.0
    control_hz: int = 20
    physics_hz: int = 240
    hover_throttle: float = 0.420
    throttle_range: float = 0.25
    reward_alt_weight: float = 0.10
    reward_tilt_weight: float = 0.50
    reward_spin_weight: float = 0.10
    reward_jerk_weight: float = 0.05
    crash_penalty: float = 50.0
    crash_min_alt_ft: float = 1.0
    crash_max_alt_offset_ft: float = 60.0
    crash_max_tilt_rad: float = 1.0


@dataclass
class PPOConfig:
    policy: str = "MlpPolicy"
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    ent_coef: float = 0.0
    # YENI: custom actor-critic mimarisi (opsiyonel).
    # None birakilirsa SB3'un kendi varsayilani kullanilir (pi=[64,64], vf=[64,64], Tanh)
    # -- yani eski config'ler / eski calistirmalar HICBIR SEKILDE etkilenmez.
    net_arch_pi: Optional[List[int]] = None   # actor (policy) agi katman boyutlari
    net_arch_vf: Optional[List[int]] = None   # critic (value) agi katman boyutlari
    activation_fn: Optional[str] = None       # "tanh" veya "relu" (None -> SB3 varsayilani: Tanh)


@dataclass
class SACConfig:
    """SAC (Soft Actor-Critic) hiperparametreleri.
    Varsayilanlar stable-baselines3'un kendi varsayilanlariyla ayni,
    boylece --config verilmeden --algo sac calistirilirsa da makul bir
    davranis elde edilir."""
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: Union[str, float] = "auto"
    # YENI: SAC icin de ayni mantik, opsiyonel ve varsayilani None
    net_arch_pi: Optional[List[int]] = None
    net_arch_vf: Optional[List[int]] = None
    activation_fn: Optional[str] = None


@dataclass
class TrainConfig:
    timesteps: int = 300_000
    n_envs: int = 4


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: Optional[str]) -> Config:
    """Verilen yaml dosyasini okuyup Config nesnesine donusturur.
    path None ise, tamamen varsayilan degerlerle bir Config doner
    (eski hardcoded davranisla ayni)."""
    if path is None:
        return Config()

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return Config(
        env=EnvConfig(**raw.get("env", {})),
        ppo=PPOConfig(**raw.get("ppo", {})),
        sac=SACConfig(**raw.get("sac", {})),
        train=TrainConfig(**raw.get("train", {})),
    )
