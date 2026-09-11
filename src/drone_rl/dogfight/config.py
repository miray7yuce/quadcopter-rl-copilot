from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class DogfightEnvConfig:
    episode_seconds: float = 45.0
    physics_hz: int = 240
    control_hz: int = 20
    hover_throttle: float = 0.420
    throttle_range: float = 0.25

    roll_authority: float = 0.6
    pitch_authority: float = 0.6
    yaw_authority: float = 0.45
    control_surface_tau_s: float = 0.08

    # v6: koni kisaltildi (30deg/70ft -> 20deg/45ft) - gorsel + oyun mantigi
    cone_half_angle_deg: float = 20.0
    cone_range_ft: float = 45.0

    standoff_target_ft: float = 40.0
    standoff_weight_start: float = 0.02
    standoff_weight_end: float = 0.08
    standoff_ramp_steps: int = 400_000
    standoff_penalty_cap: float = 2.0

    reward_align_weight: float = 0.20
    reward_exposure_weight: float = 0.15
    reward_cone_hold: float = 0.08
    reward_tilt_weight: float = 0.03
    reward_spin_weight: float = 0.06
    reward_yawrate_weight: float = 0.04
    reward_jerk_weight: float = 0.05
    opponent_fault_bonus: float = 5.0

    crash_penalty: float = 30.0
    crash_min_alt_ft: float = 5.0
    crash_max_alt_ft: float = 250.0
    # v6: 0.7 -> 0.9 rad. Eski limit dogfight manevralarinda gereksiz
    # sikligarr "crash" tetikliyordu; caydiriciligi bozmadan biraz gevsetildi.
    crash_max_tilt_rad: float = 0.9
    crash_max_yawrate_rps: float = 20.0
    # v6: 220 -> 300ft. Asil sorun bu limitin kendisinden cok, asagidaki
    # _is_out_of_bounds icinde YANLIS referans noktasina gore olculmesiydi
    # (bkz. dogfight_env.py). Referans duzeltilip sinir da biraz genisletildi.
    max_horizontal_range_ft: float = 300.0
    min_separation_ft: float = 10.0

    # YENI: sinira yaklasildikca kademeli, yumusak bir ceza uygulanir -
    # aninda "crash" yerine dronun sinirdan uzak durmayi OGRENMESINI
    # saglar, ama cok kucuk agirlikta oldugu icin pasiflige/kacmaya
    # itmez (asil odul bilesenleri hala saldirgan ucusu tesvik ediyor).
    boundary_soft_margin_ft: float = 70.0
    boundary_soft_weight: float = 0.06

    base_altitude_ft: float = 150.0
    altitude_jitter_ft: float = 15.0
    spawn_range_min_ft: float = 60.0
    spawn_range_max_ft: float = 150.0

    opponent_latest_prob: float = 0.7


@dataclass
class PPOConfig:
    policy: str = "MlpPolicy"
    n_steps: int = 2048
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
    timesteps: int = 500_000
    n_envs: int = 4


@dataclass
class PromotionConfig:
    eval_freq: int = 30_000
    n_eval_episodes: int = 30
    win_rate_threshold: float = 0.55
    mean_reward_improve_pct: float = 8.0
    consecutive_passes_required: int = 3


@dataclass
class DogfightConfig:
    env: DogfightEnvConfig = field(default_factory=DogfightEnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)


def load_dogfight_config(path: Optional[str]) -> DogfightConfig:
    if path is None:
        return DogfightConfig()
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return DogfightConfig(
        env=DogfightEnvConfig(**raw.get("env", {})),
        ppo=PPOConfig(**raw.get("ppo", {})),
        train=TrainConfig(**raw.get("train", {})),
        promotion=PromotionConfig(**raw.get("promotion", {})),
    )



