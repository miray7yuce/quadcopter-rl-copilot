"""Dogfight konfigurasyonu - v8.

v8'de ne degisti (ozet):
  * Odul fonksiyonu tamamen yeniden tasarlandi: "burun hizalama" (align)
    yerine HIZ VEKTORU tabanli ATA/AA geometrisi + Gaussian shaping
    (Benati 2025 tezi, Bolum 4.3.6/4.5.6), mesafe potansiyeli (r_dist),
    yaklasma hizi odulu (r_close), HP/WEZ hasar modeli ve terminal
    kazanma odulu (Chen et al. 2025 - aerospace-12-00265).
  * standoff_* parametreleri KALDIRILDI (yerine dist/too_close terimleri).
  * crash mantigi yumusatildi: egim/donus hizi artik ANINDA sonlandirma
    degil, kademeli ceza; sert sonlandirma sadece gercekten kurtarilamaz
    durumlarda (cok buyuk egim, yer/tavan, arena disi).
  * shaped odul agirligi egitim boyunca sonumleniyor (AOS makalesindeki
    lambda_r decay fikri) - boylece terminal (kazanma) sinyali gitgide
    baskin hale geliyor.
  * gamma 0.99 -> 0.995 (20 Hz'de 2.5 s -> ~5 s efektif ufuk).
  * ent_coef 0 -> 0.005, lineer lr sonumlemesi.
  * Mufredat (kappa-PPG benzeri) parametreleri eklendi.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class DogfightEnvConfig:
    # ------------------------------------------------------------------
    # Zamanlama
    # ------------------------------------------------------------------
    episode_seconds: float = 45.0
    physics_hz: int = 240
    control_hz: int = 20

    # ------------------------------------------------------------------
    # Kontrol
    # ------------------------------------------------------------------
    hover_throttle: float = 0.420
    throttle_range: float = 0.25
    roll_authority: float = 0.6
    pitch_authority: float = 0.6
    yaw_authority: float = 0.45
    control_surface_tau_s: float = 0.08

    # YENI: pitch_cmd'nin HANGI isaretinin dronu BURNU YONUNDE
    # ilerlettigi. F450 + ScasEngage kombinasyonunda bu isaret model
    # dosyasina bagli oldugu icin varsayim yapmiyoruz.
    # `python main.py calibrate` komutu bu degeri olcup soyler.
    forward_pitch_sign: float = 1.0

    # ------------------------------------------------------------------
    # Gozlem normalizasyonu (O1)
    # ------------------------------------------------------------------
    velocity_norm_fps: float = 30.0
    range_norm_ft: float = 100.0
    closing_norm_fps: float = 20.0
    dz_norm_ft: float = 50.0
    # Hiz cok dusukken (hover) hiz vektorunun yonu tanimsizdir; bu esigin
    # altinda angajman ekseni yumusak sekilde burun vektorune kayar.
    engage_axis_blend_fps: float = 8.0

    # ------------------------------------------------------------------
    # WEZ / koni
    # ------------------------------------------------------------------
    cone_half_angle_deg: float = 25.0
    cone_range_ft: float = 60.0

    # ------------------------------------------------------------------
    # Odul: takip geometrisi (R1 - Gaussian ATA/AA shaping)
    # ------------------------------------------------------------------
    reward_track_weight: float = 0.50
    reward_threat_weight: float = 0.35
    # Benati tezindeki grid-search sonucu: s = 0.8 rad (~36 deg).
    track_sigma_rad: float = 0.80

    # ------------------------------------------------------------------
    # Odul: mesafe / yaklasma (R2)
    # ------------------------------------------------------------------
    reward_dist_weight: float = 0.15
    dist_ref_ft: float = 150.0
    reward_close_weight: float = 0.12
    closing_ref_fps: float = 15.0
    too_close_ft: float = 25.0
    reward_too_close_weight: float = 0.30

    # ------------------------------------------------------------------
    # Odul: kilitlenme / maruz kalma
    # ------------------------------------------------------------------
    reward_cone_hold: float = 0.15

    # ------------------------------------------------------------------
    # Odul: kontrol duzgunlugu (kucultuldu - eskiden pasifligi tesvik
    # edecek kadar buyuktu)
    # ------------------------------------------------------------------
    reward_tilt_weight: float = 0.02
    reward_spin_weight: float = 0.03
    reward_yawrate_weight: float = 0.02
    reward_jerk_weight: float = 0.03

    # ------------------------------------------------------------------
    # HP / WEZ hasar modeli + terminal kazanma (R3)
    # ------------------------------------------------------------------
    hp_initial: float = 3.0            # "kac saniye kilitte kalinca duser"
    hp_damage_rate: float = 1.0        # HP/saniye (yakinlikla olceklenir)
    win_bonus: float = 50.0            # dusurme / dusurulme
    timeout_hp_bonus: float = 15.0     # sure dolunca HP farkina gore

    # ------------------------------------------------------------------
    # Guvenlik (R4) - yumusak cezalar + sadece kurtarilamaz durumda
    # sert sonlandirma
    # ------------------------------------------------------------------
    crash_penalty: float = 25.0
    opponent_fault_bonus: float = 10.0
    # DUZELTME (yer carpmasi sorunu): eskiden 8 ft idi - sert tabanla
    # arasinda neredeyse hic tepki payi yoktu. 20 ft'e cikarildi.
    crash_min_alt_ft: float = 20.0
    crash_max_alt_ft: float = 300.0
    crash_max_tilt_rad: float = 1.40   # ~80 deg: gercekten kurtarilamaz
    tilt_soft_rad: float = 0.60        # bu acinin uzerinde kademeli ceza
    tilt_soft_weight: float = 0.25
    yawrate_soft_rps: float = 4.0
    yawrate_soft_weight: float = 0.10
    max_horizontal_range_ft: float = 400.0
    boundary_soft_margin_ft: float = 120.0
    boundary_soft_weight: float = 0.25
    min_separation_ft: float = 8.0

    # YENI: irtifa tabani/tavani icin de tilt/yawrate/boundary'deki gibi
    # KADEMELI ceza. Eskiden bu SADECE sert sinirdi (crash_min_alt_ft/
    # crash_max_alt_ft) - ajan yere yaklastigini hic 'hissetmeden' aniden
    # carpiyordu, cunku hicbir erken uyari sinyali yoktu (diger 3 guvenlik
    # terimi icin vardi, bu bir eksiklikti). Taban icin marj daha genis
    # tutuldu cunku yere carpma cok daha sik/tehlikeli.
    alt_floor_soft_margin_ft: float = 60.0
    alt_floor_soft_weight: float = 0.35
    alt_ceiling_soft_margin_ft: float = 40.0
    alt_ceiling_soft_weight: float = 0.15

    # YENI: hizli inis (yuksek negatif dikey hiz) dogrudan cezalandirilir -
    # ozellikle rakip asagidayken 'menzili kapatma' odulu dalisi tesvik
    # ediyordu; bu terim dalis HIZINI irtifadan bagimsiz olarak sinirlar.
    descent_rate_soft_fps: float = 12.0
    descent_rate_soft_weight: float = 0.12

    # ------------------------------------------------------------------
    # Shaped odul sonumlemesi + mufredat ilerlemesi
    # ------------------------------------------------------------------
    shaped_weight_start: float = 1.00
    shaped_weight_end: float = 0.45
    shaped_ramp_steps: int = 400_000
    curriculum_ramp_steps: int = 250_000

    # ------------------------------------------------------------------
    # Spawn randomizasyonu (O3)
    # ------------------------------------------------------------------
    base_altitude_ft: float = 150.0
    altitude_jitter_ft: float = 25.0
    spawn_range_min_ft: float = 80.0
    spawn_range_max_ft: float = 220.0
    spawn_speed_max_fps: float = 18.0
    spawn_attitude_jitter_rad: float = 0.12

    # ------------------------------------------------------------------
    # Self-play
    # ------------------------------------------------------------------
    opponent_latest_prob: float = 0.60
    # PFSP-lite: eski checkpoint'ler arasinda win_rate'e gore softmax
    # agirlikli ornekleme sicakligi. Kucuk deger = guclu rakiplere daha
    # cok agirlik, buyuk deger = uniform'a yakin.
    opponent_pfsp_temp: float = 0.25
    # Egitim sirasinda rakip politikayi stokastik calistirmak cesitliligi
    # artirir (degerlendirmede yine deterministic kullanilir).
    opponent_deterministic: bool = False


@dataclass
class PPOConfig:
    policy: str = "MlpPolicy"
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.995          # 20 Hz -> ~5 s efektif ufuk
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    lr_final_frac: float = 0.1    # lineer sonumleme hedefi
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.05
    clip_reward: float = 10.0
    net_arch_pi: Optional[List[int]] = None
    net_arch_vf: Optional[List[int]] = None
    activation_fn: Optional[str] = None


@dataclass
class TrainConfig:
    timesteps: int = 500_000
    n_envs: int = 4
    # "subproc" | "dummy" | "auto" (auto: n_envs > 1 ise subproc)
    vec: str = "auto"
    seed: Optional[int] = None


@dataclass
class PromotionConfig:
    eval_freq: int = 30_000
    n_eval_episodes: int = 20
    win_rate_threshold: float = 0.55
    mean_reward_improve_pct: float = 5.0
    consecutive_passes_required: int = 2


@dataclass
class DogfightConfig:
    env: DogfightEnvConfig = field(default_factory=DogfightEnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)


_LEGACY_ENV_KEYS = {
    "standoff_target_ft", "standoff_weight_start", "standoff_weight_end",
    "standoff_ramp_steps", "standoff_penalty_cap",
    "reward_align_weight", "reward_exposure_weight", "reward_closing_weight",
    "crash_max_yawrate_rps",
}


def _filter_env(raw_env: dict) -> dict:
    """Eski (v7 ve oncesi) yaml dosyalarinda kalmis, artik kullanilmayan
    anahtarlari sessizce atar - boylece eski bir config dosyasi
    TypeError ile patlamaz, sadece uyari basar."""
    unknown = [k for k in raw_env if k in _LEGACY_ENV_KEYS]
    if unknown:
        print(f"[config] UYARI: artik kullanilmayan env anahtarlari yok "
              f"sayildi (v8'de odul fonksiyonu degisti): {sorted(unknown)}")
    return {k: v for k, v in raw_env.items() if k not in _LEGACY_ENV_KEYS}


def load_dogfight_config(path: Optional[str]) -> DogfightConfig:
    if path is None:
        return DogfightConfig()
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return DogfightConfig(
        env=DogfightEnvConfig(**_filter_env(raw.get("env", {}))),
        ppo=PPOConfig(**raw.get("ppo", {})),
        train=TrainConfig(**raw.get("train", {})),
        promotion=PromotionConfig(**raw.get("promotion", {})),
    )




