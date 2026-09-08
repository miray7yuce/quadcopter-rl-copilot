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
    her episode'da rastgele). EnvConfig'ten BAGIMSIZ, ayri bir dataclass.

    v6:
    - max_horizontal_range_ft: drone'un baslangic noktasindan
      uzaklasabilecegi MAKSIMUM yatay mesafe - asilirsa crash sayilir.
      Amac: (1) durum uzayini sinirlayip egitimi kolaylastirmak, (2)
      simulator ekranindaki grid'i SABIT/GARANTILI bir sinir haline
      getirmek (kayan/sonsuz grid hack'i yerine).
    - reward_heading_weight 0.08->0.14: yon takibinin daha belirgin/
      amacli gorunmesi icin guclendirildi (onceki agirlik cok zayifti,
      "amacsiz/rastgele" gorunumune katkida bulunuyordu).
    - reward_jerk_weight 0.05->0.08: komut seviyesinde ek pürüzsüzlük -
      fiziksel yuzey yumusatmasina (control_surface_tau_s) EK olarak.
    v5: reward_yawrate_weight, roll/pitch/yaw_authority,
        control_surface_tau_s, crash_max_tilt_rad=0.6, crash_max_yawrate_rps.
    v3: reward_hdot_weight, hdot_damping_min_factor, success_hdot_tol_fps.
    """
    target_altitude_min_ft: float = 20.0
    target_altitude_max_ft: float = 45.0
    target_speed_fps: float = 6.0
    episode_seconds: float = 60.0
    control_hz: int = 20
    physics_hz: int = 240
    hover_throttle: float = 0.420
    throttle_range: float = 0.25
    reward_alt_weight: float = 0.10
    reward_heading_weight: float = 0.14
    reward_tilt_weight: float = 0.05
    reward_spin_weight: float = 0.10
    reward_jerk_weight: float = 0.08
    crash_penalty: float = 50.0
    crash_min_alt_ft: float = 1.0
    crash_max_alt_offset_ft: float = 60.0
    crash_max_tilt_rad: float = 0.6
    altitude_start_offset_ft: float = 25.0
    altitude_start_jitter_ft: float = 2.0
    success_alt_tol_ft: float = 1.5
    success_hold_seconds: float = 1.0
    success_bonus: float = 20.0
    reward_hdot_weight: float = 0.12
    hdot_damping_min_factor: float = 0.3
    success_hdot_tol_fps: float = 1.0
    reward_yawrate_weight: float = 0.06
    roll_authority: float = 0.6
    pitch_authority: float = 0.6
    yaw_authority: float = 0.45
    control_surface_tau_s: float = 0.08
    crash_max_yawrate_rps: float = 20.0
    # --- YENI (v6) ---
    max_horizontal_range_ft: float = 90.0


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
    use_custom_extractor: bool = False
    features_dim: int = 64


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
