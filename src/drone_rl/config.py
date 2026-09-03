"""Merkezi config yukleme. YAML dosyasindaki degerleri dataclass'lara donusturur."""

from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class EnvConfig:
    """F450HoverEnv icin ayarlar (hedef irtifa sabit)."""
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
class FlightEnvConfig:
    """F450FlightEnv icin ayarlar (hedef irtifa + hedef yon, ikisi de
    her episode'da rastgele). EnvConfig'ten BAGIMSIZ, ayri bir dataclass -
    hover config'ini hic etkilemez."""
    target_altitude_min_ft: float = 20.0
    target_altitude_max_ft: float = 45.0
    target_speed_fps: float = 6.0
    episode_seconds: float = 60.0
    control_hz: int = 20
    physics_hz: int = 240
    hover_throttle: float = 0.420
    throttle_range: float = 0.25
    reward_alt_weight: float = 0.10
    reward_heading_weight: float = 0.08
    reward_tilt_weight: float = 0.05
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
    flight_env: FlightEnvConfig = field(default_factory=FlightEnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: Optional[str]) -> Config:
    if path is None:
        return Config()

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return Config(
        env=EnvConfig(**raw.get("env", {})),
        flight_env=FlightEnvConfig(**raw.get("flight_env", {})),
        ppo=PPOConfig(**raw.get("ppo", {})),
        train=TrainConfig(**raw.get("train", {})),
    )




