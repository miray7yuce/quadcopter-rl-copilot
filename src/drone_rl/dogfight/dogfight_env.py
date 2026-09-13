"""Iki F450 arasinda 'dogfight' gorevi - TAM 3D fizik.

================================================================
v8 - "NEDEN BIRBIRLERINI TAKIP ETMIYORLARDI" DUZELTMELERI
================================================================

(1) AJAN HEDEFIN HANGI YONDE OLDUGUNU BILMIYORDU.  [KRITIK]
    Eski gozlemde yon bilgisi olarak SADECE `align_owner = cos(ATA)`
    vardi. Kosinus simetriktir: hedef 30 derece SOLDA da 30 derece
    SAGDA da ayni degeri uretir. Yani ajan "ne kadar sapmisim"i
    goruyor ama "hangi yone donmeliyim"i GOREMIYORDU. Tek kareden
    (Markov) dogru karar vermesi matematiksel olarak imkansizdi;
    ancak salinarak/deneyerek arayabiliyordu.
    COZUM: LOS (line-of-sight) vektoru artik GOVDE EKSENINDE, ISARETLI
    3 bilesen olarak veriliyor (los_bx = on, los_by = sag, los_bz =
    asagi). "Sag tarafta" ile "sol tarafta" artik farkli isaretli.

(2) GOZLEMDE KENDI HIZI YOKTU.  [KRITIK]
    17 boyutlu eski vektorde `hdot` disinda hicbir hiz yoktu: u, v, w
    (govde eksen hizlari) yok, rakibin hizi/yonu yok. Bir multikopter
    icin bu gozu kapali ucmaktir - ileri gidip gitmedigini bilemez,
    kesme (lead pursuit) yapmasi imkansizdir.
    COZUM: kendi govde hizlari (u,v,w), hiz buyuklugu ve RAKIBIN hiz
    vektoru (kendi govde ekseninde) gozleme eklendi.

(3) `closing_n` GOZLEMDE HER ZAMAN SIFIRDI.  [GERCEK BUG]
    step() icinde `self._prev_range_ft = rng_ft` atamasi, sonundaki
    `_get_obs_for()` cagrisindan ONCE yapiliyordu; _get_obs_for menzili
    yeniden hesaplayip AYNI _prev_range_ft'ten cikariyordu -> (x-x)/dt
    = 0. info["closing_fps"] dogruydu ama POLITIKAYA giden ozellik
    oluydu. Ayrica _get_obs_for rakip icin closing'i acikca 0.0
    sabitliyordu -> self-play'de dagilim kaymasi.
    COZUM: closing bir kez step()/reset() icinde dogru hesaplanip
    self._closing_fps'te saklaniyor; her iki taraf da ayni (fiziksel
    olarak simetrik) gercek degeri goruyor.

(4) "NISAN AL" ODULU QUADROTOR FIZIGIYLE CELISIYORDU.  [KRITIK]
    _nose_vector burnu psi+theta'dan uretiyordu. Bir F450 ilerlemek
    icin burnunu ASAGI egmek zorundadir (theta < 0). Hedef ayni
    irtifadaysa, yaklasmak icin pitch yaptigin anda align_cos DUSER.
    Yani odul fonksiyonu fiilen "yaklasma, sadece burnunu cevir"
    diyordu; ajanin buldugu lokal optimum tam olarak buydu.
    COZUM: ATA artik BURUN ile degil HIZ VEKTORU ile LOS arasinda
    olculuyor (Benati 2025 tezi, Bolum 4.3.2 - "ATA: angle between the
    agent's velocity vector and the line-of-sight"). Koni/WEZ de ayni
    "angajman ekseni" uzerinde tanimli. Hiz cok dusukken (hover) yon
    tanimsiz oldugu icin eksen yumusak sekilde burun vektorune kayar.

Ikincil duzeltmeler:
  * r_dist (mesafe potansiyeli) + r_close (yaklasma hizi) odulleri
    eklendi - eskiden menzili kapatmak icin net gradyan yoktu.
  * HP / WEZ hasar modeli + terminal kazanma-kaybetme odulu eklendi -
    eskiden "kazanmak" diye bir kavram yoktu (Chen et al. 2025,
    Denklem 3; Benati 2025, Bolum 4.5).
  * Egim ve donus hizi artik ANINDA "crash" degil, kademeli ceza.
    Sert sonlandirma sadece kurtarilamaz durumlarda.
  * Spawn'da hiz + yonelim randomizasyonu (her episode hover'dan
    baslamiyor).
  * Kolaydan zora rakip mufredati (HoverOpponent -> daire -> kappa-PPG
    saf takip), AOS makalesindeki kappa-PPG fikrinin sade hali.

--- KIM RL ILE CALISIYOR? ---
- fdm_self: HER ZAMAN disaridan (PPO) gelen action ile suruluyor.
- fdm_opp: self.opponent_controller uzerinden - Stage A'da scripted,
  Stage B'de dondurulmus/inference-only bir PPO modeli.
- reward SADECE fdm_self icin ogrenme sinyalidir. info'daki opp_reward
  sadece demo arayuzu icin hesaplanan simetrik bir gosterim degeridir.
"""

import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import jsbsim

LAT0_DEG = 0.0
LON0_DEG = 0.0
FT_PER_DEG_LAT = 364567.2

# Gozlem boyutu. v7'de 17 idi -> v8'de 30.
# DIKKAT: bu degisiklik ESKI checkpoint'leri ve vecnormalize.pkl
# dosyalarini GECERSIZ kilar. Havuzu silip sifirdan egitmek gerekir.
OBS_DIM = 30
ACT_DIM = 4


# ======================================================================
# Kucuk matematik yardimcilari
# ======================================================================

def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _norm3(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


def _world_to_body(phi, theta, psi, n, e, up):
    """(kuzey, dogu, yukari) dunya vektorunu govde eksenine cevirir.
    Govde ekseni standart havacilik konvansiyonu: x = on, y = sag,
    z = asagi.  R_ned->body = Rx(phi) * Ry(theta) * Rz(psi)."""
    d = -up  # NED'de asagi pozitif
    cps, sps = math.cos(psi), math.sin(psi)
    cth, sth = math.cos(theta), math.sin(theta)
    cph, sph = math.cos(phi), math.sin(phi)

    x1 = n * cps + e * sps
    y1 = -n * sps + e * cps
    z1 = d

    x2 = x1 * cth - z1 * sth
    y2 = y1
    z2 = x1 * sth + z1 * cth

    xb = x2
    yb = y2 * cph + z2 * sph
    zb = -y2 * sph + z2 * cph
    return xb, yb, zb


def _gauss(angle_rad: float, sigma: float) -> float:
    """Benati 2025, Denklem 4.9: exp(-angle^2 / s^2).
    Mutlak-deger yerine karesel (Gaussian) form, kucuk acilarda daha
    duzgun turev ve daha hizli yakinsama veriyor (tezde grid-search ile
    s = 0.8 rad ~ 36 derece en iyi bulunmus)."""
    s = max(sigma, 1e-6)
    return math.exp(-(angle_rad * angle_rad) / (s * s))


# ======================================================================
# Rakip kontrolculeri
# ======================================================================

class BaseOpponentController:
    name = "base"

    def reset(self):
        pass

    def compute_action(self, env: "DogfightEnv") -> np.ndarray:
        raise NotImplementedError


def _pitch_cmd_for_target_theta(cfg, theta_now, q_now, theta_des,
                                kp=2.5, kd=0.45) -> float:
    """pitch_cmd'nin isaret semantigi model dosyasina bagli oldugu icin
    (bkz. cfg.forward_pitch_sign) tum scripted kontrolculer pitch'i
    BURADAN uretir. cfg.forward_pitch_sign, "pozitif pitch_cmd dronu
    burnu yonunde ilerletir" (yani burnu asagi eger) anlamina gelir;
    dolayisiyla theta'yi AZALTMAK icin pozitif komut gerekir."""
    s = cfg.forward_pitch_sign
    cmd = s * (kp * (theta_now - theta_des) + kd * q_now)
    return float(np.clip(cmd, -1.0, 1.0))


class HoverOpponent(BaseOpponentController):
    """Mufredatin en kolay seviyesi: yerinde durur, seviyeli kalir.
    Ajanin once "yaklas ve nisan al"i ogrenmesi icin."""

    name = "hover"

    def __init__(self, cfg, kp_alt=0.10, kd_alt=0.30, kp_roll=3.0, kd_roll=0.5):
        self.cfg = cfg
        self.kp_alt, self.kd_alt = kp_alt, kd_alt
        self.kp_roll, self.kd_roll = kp_roll, kd_roll
        self._target_alt = None

    def reset(self):
        self._target_alt = None

    def compute_action(self, env):
        f = env.fdm_opp
        if self._target_alt is None:
            self._target_alt = f["position/h-agl-ft"]

        phi = f["attitude/phi-rad"]
        theta = f["attitude/theta-rad"]
        p = f["velocities/p-rad_sec"]
        q = f["velocities/q-rad_sec"]

        roll_cmd = float(np.clip(self.kp_roll * (0.0 - phi) - self.kd_roll * p, -1.0, 1.0))
        pitch_cmd = _pitch_cmd_for_target_theta(self.cfg, theta, q, 0.0)
        yaw_cmd = 0.0
        alt_err = self._target_alt - f["position/h-agl-ft"]
        throttle_cmd = float(np.clip(
            self.kp_alt * alt_err - self.kd_alt * f["velocities/h-dot-fps"], -1.0, 1.0))
        return np.array([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=np.float32)


class ScriptedCircleOpponent(BaseOpponentController):
    """Sabit bankli daire - mufredatin ikinci seviyesi. Kovalamaz, ama
    hareketli bir hedef olarak ajana 'lead pursuit' gerektirir."""

    name = "circle"

    def __init__(self, cfg, bank_deg=15.0, kp_roll=3.5, kd_roll=0.3,
                 kp_alt=0.12, kd_alt=0.35):
        self.cfg = cfg
        self.bank_deg = bank_deg
        self.kp_roll, self.kd_roll = kp_roll, kd_roll
        self.kp_alt, self.kd_alt = kp_alt, kd_alt
        self._target_alt_ft = None

    def reset(self):
        self._target_alt_ft = None

    def compute_action(self, env):
        f = env.fdm_opp
        if self._target_alt_ft is None:
            self._target_alt_ft = f["position/h-agl-ft"]

        phi = f["attitude/phi-rad"]
        theta = f["attitude/theta-rad"]
        p = f["velocities/p-rad_sec"]
        q = f["velocities/q-rad_sec"]

        target_roll = math.radians(self.bank_deg)
        roll_cmd = float(np.clip(
            self.kp_roll * (target_roll - phi) - self.kd_roll * p, -1.0, 1.0))
        # hafif ileri egim -> daire cizerken gercekten yol alsin
        pitch_cmd = _pitch_cmd_for_target_theta(self.cfg, theta, q, -0.12)
        yaw_cmd = 0.0
        alt_err = self._target_alt_ft - f["position/h-agl-ft"]
        throttle_cmd = float(np.clip(
            self.kp_alt * alt_err - self.kd_alt * f["velocities/h-dot-fps"], -1.0, 1.0))
        return np.array([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=np.float32)


class KappaPursuitOpponent(BaseOpponentController):
    """kappa-PPG: kappa olasilikla RASTGELE bir yone, aksi halde LOS
    uzerine (saf takip) ucar. Chen et al. 2025 (aerospace-12-00265,
    Denklem 23-24) icindeki kappa-PPG mufredat politikasinin sade hali.
    kappa buyudukce rakip zayiflar/ongorulemezlesir, kucuk kappa ise
    gercek bir kovalayicidir.

    Hiz kontrolu ACIK DONGU DEGIL: istenen ileri hiz ile fiili ileri
    hiz arasindaki hatadan bir hedef egim acisi uretilir, boylece
    pitch isaret/olcek belirsizligi kendini duzeltir."""

    def __init__(self, cfg, kappa=0.3, redirect_period_s=2.5,
                 speed_gain=0.25, max_speed_fps=25.0, max_tilt_rad=0.45,
                 kp_roll=3.5, kd_roll=0.45, kp_yaw=1.2, kd_yaw=0.15,
                 bank_limit_deg=45.0, kp_alt=0.10, kd_alt=0.30,
                 speed_err_gain=0.05):
        self.cfg = cfg
        self.kappa = float(kappa)
        self.redirect_period_s = redirect_period_s
        self.speed_gain = speed_gain
        self.max_speed_fps = max_speed_fps
        self.max_tilt_rad = max_tilt_rad
        self.kp_roll, self.kd_roll = kp_roll, kd_roll
        self.kp_yaw, self.kd_yaw = kp_yaw, kd_yaw
        self.bank_limit = math.radians(bank_limit_deg)
        self.kp_alt, self.kd_alt = kp_alt, kd_alt
        self.speed_err_gain = speed_err_gain
        self.name = f"kappa{int(round(kappa * 100)):03d}"
        self._rand_dir = None
        self._t_since_redirect = 1e9
        self._rng = np.random.default_rng()

    def seed(self, rng):
        self._rng = rng

    def reset(self):
        self._rand_dir = None
        self._t_since_redirect = 1e9

    def _maybe_redirect(self, dt):
        self._t_since_redirect += dt
        if self._t_since_redirect < self.redirect_period_s:
            return
        self._t_since_redirect = 0.0
        if self._rng.random() < self.kappa:
            yaw = self._rng.uniform(-math.pi, math.pi)
            pitch = self._rng.uniform(-0.25, 0.25)
            self._rand_dir = (
                math.cos(pitch) * math.cos(yaw),
                math.cos(pitch) * math.sin(yaw),
                math.sin(pitch),
            )
        else:
            self._rand_dir = None  # saf takip

    def compute_action(self, env):
        f = env.fdm_opp
        self._maybe_redirect(env.control_dt)

        if self._rand_dir is not None:
            dn, de, dup = self._rand_dir
            target_alt = f["position/h-agl-ft"] + dup * 40.0
        else:
            on, oe, oalt = env._position(env.fdm_opp)
            sn, se, salt = env._position(env.fdm_self)
            dn, de, dup = sn - on, se - oe, salt - oalt
            target_alt = salt

        horiz = max(math.hypot(dn, de), 1e-6)
        desired_yaw = math.atan2(de, dn)

        phi = f["attitude/phi-rad"]
        theta = f["attitude/theta-rad"]
        psi = f["attitude/psi-rad"]
        p = f["velocities/p-rad_sec"]
        q = f["velocities/q-rad_sec"]
        r = f["velocities/r-rad_sec"]

        yaw_err = _wrap_pi(desired_yaw - psi)

        # Koordineli donus: yaw hatasina orantili banka acisi
        target_roll = float(np.clip(yaw_err * 1.2, -self.bank_limit, self.bank_limit))
        roll_cmd = float(np.clip(
            self.kp_roll * (target_roll - phi) - self.kd_roll * p, -1.0, 1.0))
        yaw_cmd = float(np.clip(self.kp_yaw * yaw_err - self.kd_yaw * r, -1.0, 1.0))

        # Ileri hiz kontrolu (kapali dongu)
        vn = f["velocities/v-north-fps"]
        ve = f["velocities/v-east-fps"]
        fwd_speed = (vn * dn + ve * de) / horiz
        v_des = float(np.clip(self.speed_gain * horiz, 0.0, self.max_speed_fps))
        v_des *= max(math.cos(yaw_err), 0.0)  # cok sapmisken once don
        tilt_des = float(np.clip(
            self.speed_err_gain * (v_des - fwd_speed), -0.15, self.max_tilt_rad))
        theta_des = -tilt_des  # burun asagi = ileri
        pitch_cmd = _pitch_cmd_for_target_theta(self.cfg, theta, q, theta_des)

        alt_err = target_alt - f["position/h-agl-ft"]
        throttle_cmd = float(np.clip(
            self.kp_alt * alt_err - self.kd_alt * f["velocities/h-dot-fps"], -1.0, 1.0))

        return np.array([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=np.float32)


class OpponentCurriculum:
    """Kolaydan zora rakip havuzu. Egitim ilerledikce daha zor
    seviyelerin 'kilidi acilir'; her reset()'te agirlikli olarak en zor
    acik seviye, bazen de daha kolay seviyeler secilir (AOS makalesinde
    kappa = 0.95 kopyasinin havuzda tutulmasiyla ayni amac: onceki
    becerileri unutmamak)."""

    def __init__(self, cfg, hardest_prob: float = 0.6):
        self.cfg = cfg
        self.hardest_prob = hardest_prob
        self.levels = [
            HoverOpponent(cfg),
            ScriptedCircleOpponent(cfg),
            KappaPursuitOpponent(cfg, kappa=0.70),
            KappaPursuitOpponent(cfg, kappa=0.35),
            KappaPursuitOpponent(cfg, kappa=0.05),
        ]

    def sample(self, rng, progress: float) -> BaseOpponentController:
        n = len(self.levels)
        unlocked = 1 + int(round(float(np.clip(progress, 0.0, 1.0)) * (n - 1)))
        unlocked = int(np.clip(unlocked, 1, n))
        if unlocked == 1 or rng.random() < self.hardest_prob:
            idx = unlocked - 1
        else:
            idx = int(rng.integers(0, unlocked))
        ctrl = self.levels[idx]
        if hasattr(ctrl, "seed"):
            ctrl.seed(rng)
        return ctrl


class NormalizerStats:
    def __init__(self, vecnorm):
        self.mean = vecnorm.obs_rms.mean.astype(np.float32)
        self.var = vecnorm.obs_rms.var.astype(np.float32)
        self.epsilon = vecnorm.epsilon
        self.clip_obs = vecnorm.clip_obs

    @property
    def obs_dim(self) -> int:
        return int(self.mean.shape[-1])

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        normed = (obs - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normed, -self.clip_obs, self.clip_obs).astype(np.float32)


class PPOOpponentController(BaseOpponentController):
    name = "ppo"

    def __init__(self, model, stats: NormalizerStats, deterministic: bool = False):
        self.model = model
        self.stats = stats
        self.deterministic = deterministic

    def compute_action(self, env: "DogfightEnv") -> np.ndarray:
        obs = env._get_obs_for(env.fdm_opp, env.fdm_self, env.prev_action_opp)
        norm_obs = self.stats.normalize(obs).reshape(1, -1)
        action, _ = self.model.predict(norm_obs, deterministic=self.deterministic)
        return action[0]


# ======================================================================
# Ortam
# ======================================================================

class DogfightEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg, opponent_controller: Optional[BaseOpponentController] = None,
                 opponent_pool=None, opponent_latest_prob: float = 0.6,
                 opponent_curriculum: Optional[OpponentCurriculum] = None):
        super().__init__()
        self.cfg = cfg

        self.physics_hz = int(cfg.physics_hz)
        self.control_hz = int(cfg.control_hz)
        self.physics_dt = 1.0 / self.physics_hz
        self.substeps = self.physics_hz // self.control_hz
        self.max_steps = int(cfg.episode_seconds * self.control_hz)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self.fdm_self = jsbsim.FGFDMExec(None)
        self.fdm_self.set_debug_level(0)
        if not self.fdm_self.load_model("F450"):
            raise RuntimeError("F450 (self) yuklenemedi")
        self.fdm_self.set_dt(self.physics_dt)

        self.fdm_opp = jsbsim.FGFDMExec(None)
        self.fdm_opp.set_debug_level(0)
        if not self.fdm_opp.load_model("F450"):
            raise RuntimeError("F450 (opp) yuklenemedi")
        self.fdm_opp.set_dt(self.physics_dt)

        self.opponent_curriculum = opponent_curriculum
        self.opponent_controller = opponent_controller or HoverOpponent(cfg)
        self.opponent_pool = opponent_pool
        self.opponent_latest_prob = opponent_latest_prob

        self._surface_self = np.zeros(3, dtype=np.float64)
        self._surface_opp = np.zeros(3, dtype=np.float64)
        self.prev_action_self = np.zeros(ACT_DIM, dtype=np.float32)
        self.prev_action_opp = np.zeros(ACT_DIM, dtype=np.float32)

        self.step_count = 0
        self.my_score = 0
        self.opp_score = 0
        self.hp_self = float(cfg.hp_initial)
        self.hp_opp = float(cfg.hp_initial)
        self._prev_range_ft = None
        self._closing_fps = 0.0

        # Egitim ilerlemesine bagli olarak disaridan set edilir
        self.shaped_weight = float(cfg.shaped_weight_start)
        self.curriculum_progress = 0.0

        self._arena_center_n = 0.0
        self._arena_center_e = 0.0

    # ------------------------------------------------------------------
    @property
    def control_dt(self):
        return self.substeps * self.physics_dt

    def set_shaped_weight(self, w: float):
        """Shaped (yogun) odulun agirligi. Egitim ilerledikce
        sonumlenir -> terminal kazanma sinyali baskin hale gelir."""
        self.shaped_weight = float(w)

    def set_curriculum_progress(self, p: float):
        self.curriculum_progress = float(np.clip(p, 0.0, 1.0))

    def set_opponent_controller(self, controller: BaseOpponentController):
        self.opponent_controller = controller

    def set_opponent_controller_from_pool(self, model_vecnorm_tuple):
        from drone_rl.dogfight.env_factory import load_opponent_controller
        model_path, vecnorm_path = model_vecnorm_tuple
        self.opponent_controller = load_opponent_controller(
            model_path, vecnorm_path, deterministic=self.cfg.opponent_deterministic)

    # ------------------------------------------------------------------
    # Kinematik yardimcilari
    # ------------------------------------------------------------------
    def _attitude(self, fdm):
        return (fdm["attitude/phi-rad"], fdm["attitude/theta-rad"], fdm["attitude/psi-rad"])

    def _nose_vector(self, fdm):
        _, theta, psi = self._attitude(fdm)
        return (math.cos(theta) * math.cos(psi),
                math.cos(theta) * math.sin(psi),
                math.sin(theta))

    def _world_velocity(self, fdm):
        """(kuzey, dogu, yukari) ft/s."""
        return (fdm["velocities/v-north-fps"],
                fdm["velocities/v-east-fps"],
                -fdm["velocities/v-down-fps"])

    def _engage_axis(self, fdm):
        """DUZELTME (4): angajman ekseni artik BURUN degil HIZ VEKTORU.
        Bir quadrotor ilerlemek icin burnunu asagi eger; burun tabanli
        nisan alma ile yaklasma hareketi birbirini iptal ediyordu.
        Hiz cok dusukken (hover, |v| -> 0) yon tanimsiz olacagi icin
        eksen yumusak sekilde burun vektorune kayar - boylece odulde
        sicrama/sureksizlik olmaz."""
        vn, ve, vu = self._world_velocity(fdm)
        speed = _norm3(vn, ve, vu)
        nn, ne, nu = self._nose_vector(fdm)
        blend = max(self.cfg.engage_axis_blend_fps - speed, 0.0)
        ax, ay, az = vn + blend * nn, ve + blend * ne, vu + blend * nu
        mag = _norm3(ax, ay, az)
        if mag < 1e-6:
            return (nn, ne, nu), speed
        return (ax / mag, ay / mag, az / mag), speed

    def _position(self, fdm):
        lat = fdm["position/lat-gc-deg"]
        lon = fdm["position/long-gc-deg"]
        ft_per_deg_lon = FT_PER_DEG_LAT * math.cos(math.radians(LAT0_DEG))
        north_ft = (lat - LAT0_DEG) * FT_PER_DEG_LAT
        east_ft = (lon - LON0_DEG) * ft_per_deg_lon
        alt_ft = fdm["position/h-agl-ft"]
        return north_ft, east_ft, alt_ft

    def _geom(self, fdm_a, fdm_b):
        """a -> b yonunde tam angajman geometrisi."""
        na, ea, ua = self._position(fdm_a)
        nb, eb, ub = self._position(fdm_b)
        dn, de, dz = nb - na, eb - ea, ub - ua
        rng_ft = max(_norm3(dn, de, dz), 1e-3)
        los = (dn / rng_ft, de / rng_ft, dz / rng_ft)

        axis_a, speed_a = self._engage_axis(fdm_a)
        axis_b, speed_b = self._engage_axis(fdm_b)

        # ATA: a'nin hiz vektoru ile LOS arasindaki aci
        cos_ata = float(np.clip(axis_a[0] * los[0] + axis_a[1] * los[1] + axis_a[2] * los[2],
                                -1.0, 1.0))
        # AA (aspect angle): LOS ile b'nin hiz yonu arasindaki aci.
        # 0 -> b bizden kaciyor (biz onun kuyrugundayiz) = ideal.
        cos_aa = float(np.clip(los[0] * axis_b[0] + los[1] * axis_b[1] + los[2] * axis_b[2],
                               -1.0, 1.0))
        return {
            "rng": rng_ft, "los": los, "dn": dn, "de": de, "dz": dz,
            "axis_a": axis_a, "axis_b": axis_b,
            "speed_a": speed_a, "speed_b": speed_b,
            "cos_ata": cos_ata, "cos_aa": cos_aa,
        }

    def _in_cone(self, cos_axis_los: float, rng_ft: float) -> bool:
        cone_half_cos = math.cos(math.radians(self.cfg.cone_half_angle_deg))
        return bool(cos_axis_los >= cone_half_cos and rng_ft <= self.cfg.cone_range_ft)

    def _boundary_dist(self, fdm) -> float:
        n_ft, e_ft, _ = self._position(fdm)
        return math.hypot(n_ft - self._arena_center_n, e_ft - self._arena_center_e)

    # ------------------------------------------------------------------
    # Gozlem
    # ------------------------------------------------------------------
    def _get_obs_for(self, fdm_owner, fdm_other, prev_action_owner):
        cfg = self.cfg
        g = self._geom(fdm_owner, fdm_other)
        phi, theta, psi = self._attitude(fdm_owner)

        # (1) Gövde eksenindeki ISARETLI LOS - "hangi yone donmeliyim"
        los_bx, los_by, los_bz = _world_to_body(phi, theta, psi, *g["los"])

        # (2) Kendi govde hizlari + rakibin hizi (kendi govde ekseninde)
        vnorm = max(cfg.velocity_norm_fps, 1e-6)
        u = fdm_owner["velocities/u-fps"] / vnorm
        v = fdm_owner["velocities/v-fps"] / vnorm
        w = fdm_owner["velocities/w-fps"] / vnorm
        speed_n = g["speed_a"] / vnorm

        ovn, ove, ovu = self._world_velocity(fdm_other)
        ov_bx, ov_by, ov_bz = _world_to_body(phi, theta, psi, ovn, ove, ovu)
        ov_bx, ov_by, ov_bz = ov_bx / vnorm, ov_by / vnorm, ov_bz / vnorm

        # (3) Gercek yaklasma hizi (artik sifir degil)
        closing_n = self._closing_fps / max(cfg.closing_norm_fps, 1e-6)

        cos_ata = g["cos_ata"]
        cos_aa = g["cos_aa"]
        in_my_cone = 1.0 if self._in_cone(cos_ata, g["rng"]) else 0.0
        # rakibin ATA'si: kendi ekseni ile BANA giden LOS (-los) arasinda
        cos_ata_other = -cos_aa
        in_other_cone = 1.0 if self._in_cone(cos_ata_other, g["rng"]) else 0.0

        alt = fdm_owner["position/h-agl-ft"]
        alt_n = (alt - cfg.base_altitude_ft) / max(cfg.base_altitude_ft, 1e-6)
        boundary_n = self._boundary_dist(fdm_owner) / max(cfg.max_horizontal_range_ft, 1e-6)

        if fdm_owner is self.fdm_self:
            hp_own, hp_other = self.hp_self, self.hp_opp
        else:
            hp_own, hp_other = self.hp_opp, self.hp_self
        hp0 = max(cfg.hp_initial, 1e-6)

        obs = np.array([
            phi, theta,                                     # 0-1
            fdm_owner["velocities/p-rad_sec"] / 5.0,        # 2
            fdm_owner["velocities/q-rad_sec"] / 5.0,        # 3
            fdm_owner["velocities/r-rad_sec"] / 5.0,        # 4
            u, v, w,                                        # 5-7
            speed_n,                                        # 8
            los_bx, los_by, los_bz,                         # 9-11
            g["rng"] / max(cfg.range_norm_ft, 1e-6),        # 12
            closing_n,                                      # 13
            g["dz"] / max(cfg.dz_norm_ft, 1e-6),            # 14
            cos_ata,                                        # 15
            cos_aa,                                         # 16
            ov_bx, ov_by, ov_bz,                            # 17-19
            in_my_cone, in_other_cone,                      # 20-21
            alt_n,                                          # 22
            boundary_n,                                     # 23
            hp_own / hp0, hp_other / hp0,                   # 24-25
            *prev_action_owner,                             # 26-29
        ], dtype=np.float32)
        return obs

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _init_fdm(self, fdm, alt_ft, heading_deg, north_ft=0.0, east_ft=0.0,
                  speed_fps=0.0, phi=0.0, theta=0.0):
        ft_per_deg_lon = FT_PER_DEG_LAT * math.cos(math.radians(LAT0_DEG))
        fdm["ic/lat-gc-deg"] = LAT0_DEG + north_ft / FT_PER_DEG_LAT
        fdm["ic/long-gc-deg"] = LON0_DEG + east_ft / ft_per_deg_lon
        fdm["ic/h-agl-ft"] = alt_ft
        # (O3) hover'dan degil, govde-ileri bir hizla basla
        fdm["ic/u-fps"] = float(speed_fps)
        fdm["ic/v-fps"] = 0.0
        fdm["ic/w-fps"] = 0.0
        fdm["ic/phi-rad"] = float(phi)
        fdm["ic/theta-rad"] = float(theta)
        fdm["ic/psi-true-rad"] = math.radians(heading_deg)
        fdm.run_ic()
        for i in range(4):
            fdm[f"propulsion/engine[{i}]/set-running"] = 1
        fdm["fcs/ScasEngage"] = 1
        fdm["fcs/aileron-cmd-norm"] = 0.0
        fdm["fcs/elevator-cmd-norm"] = 0.0
        fdm["fcs/rudder-cmd-norm"] = 0.0
        fdm["fcs/throttle-cmd-norm"] = self.cfg.hover_throttle

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.cfg
        rng = self.np_random

        # Stage B: havuzdan rakip ornekle
        if self.opponent_pool is not None:
            sampled = self.opponent_pool.sample(
                cfg.opponent_latest_prob, rng=rng, temperature=cfg.opponent_pfsp_temp)
            if sampled is not None:
                from drone_rl.dogfight.env_factory import load_opponent_controller
                self.opponent_controller = load_opponent_controller(
                    *sampled, deterministic=cfg.opponent_deterministic)
        # Stage A: mufredattan rakip ornekle
        elif self.opponent_curriculum is not None:
            self.opponent_controller = self.opponent_curriculum.sample(
                rng, self.curriculum_progress)

        rng_range = rng.uniform(cfg.spawn_range_min_ft, cfg.spawn_range_max_ft)
        bearing_deg = rng.uniform(0.0, 360.0)
        alt_self = cfg.base_altitude_ft + rng.uniform(-cfg.altitude_jitter_ft, cfg.altitude_jitter_ft)
        alt_opp = cfg.base_altitude_ft + rng.uniform(-cfg.altitude_jitter_ft, cfg.altitude_jitter_ft)
        heading_self = rng.uniform(0.0, 360.0)
        heading_opp = rng.uniform(0.0, 360.0)

        jit = cfg.spawn_attitude_jitter_rad
        spd = cfg.spawn_speed_max_fps

        dn_target = rng_range * math.cos(math.radians(bearing_deg))
        de_target = rng_range * math.sin(math.radians(bearing_deg))

        self._arena_center_n = dn_target / 2.0
        self._arena_center_e = de_target / 2.0

        self._init_fdm(self.fdm_self, alt_self, heading_self,
                       north_ft=0.0, east_ft=0.0,
                       speed_fps=rng.uniform(0.0, spd),
                       phi=rng.uniform(-jit, jit), theta=rng.uniform(-jit, jit))
        self._init_fdm(self.fdm_opp, alt_opp, heading_opp,
                       north_ft=dn_target, east_ft=de_target,
                       speed_fps=rng.uniform(0.0, spd),
                       phi=rng.uniform(-jit, jit), theta=rng.uniform(-jit, jit))

        self._surface_self[:] = 0.0
        self._surface_opp[:] = 0.0
        self.prev_action_self = np.zeros(ACT_DIM, dtype=np.float32)
        self.prev_action_opp = np.zeros(ACT_DIM, dtype=np.float32)
        self.step_count = 0
        self.my_score = 0
        self.opp_score = 0
        self.hp_self = float(cfg.hp_initial)
        self.hp_opp = float(cfg.hp_initial)
        self.opponent_controller.reset()

        g = self._geom(self.fdm_self, self.fdm_opp)
        self._prev_range_ft = g["rng"]
        self._closing_fps = 0.0

        return self._get_obs_for(self.fdm_self, self.fdm_opp, self.prev_action_self), {}

    # ------------------------------------------------------------------
    # Aksiyon
    # ------------------------------------------------------------------
    def _apply_action(self, fdm, surface_state, action):
        roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd = action
        aileron_t = float(np.clip(roll_cmd * self.cfg.roll_authority, -1.0, 1.0))
        elevator_t = float(np.clip(-pitch_cmd * self.cfg.pitch_authority, -1.0, 1.0))
        rudder_t = float(np.clip(yaw_cmd * self.cfg.yaw_authority, -1.0, 1.0))
        throttle = float(np.clip(
            self.cfg.hover_throttle + throttle_cmd * self.cfg.throttle_range, 0.0, 1.0))
        targets = np.array([aileron_t, elevator_t, rudder_t])
        alpha = self.physics_dt / (self.cfg.control_surface_tau_s + self.physics_dt)
        surface_state += alpha * (targets - surface_state)
        fdm["fcs/aileron-cmd-norm"] = float(surface_state[0])
        fdm["fcs/elevator-cmd-norm"] = float(surface_state[1])
        fdm["fcs/rudder-cmd-norm"] = float(surface_state[2])
        fdm["fcs/throttle-cmd-norm"] = throttle

    # ------------------------------------------------------------------
    # Sonlandirma
    # ------------------------------------------------------------------
    def _hard_terminate(self, fdm) -> Optional[str]:
        """DUZELTME (R4): egim ve donus hizi artik burada DEGIL, kademeli
        ceza olarak ele aliniyor. Eskiden 0.9 rad (51 derece) egim aninda
        -30 ceza + episode sonu demekti; bu, agresif manevrayi olumcul
        kilip ajani 'duz uc, yaklasma' politikasina itiyordu."""
        alt = fdm["position/h-agl-ft"]
        if alt < self.cfg.crash_min_alt_ft:
            return "ground"
        if alt > self.cfg.crash_max_alt_ft:
            return "ceiling"
        phi, theta, _ = self._attitude(fdm)
        if abs(phi) > self.cfg.crash_max_tilt_rad or abs(theta) > self.cfg.crash_max_tilt_rad:
            return "tumble"
        if self._boundary_dist(fdm) > self.cfg.max_horizontal_range_ft:
            return "boundary"
        return None

    def _soft_safety_penalty(self, fdm) -> float:
        cfg = self.cfg
        phi, theta, _ = self._attitude(fdm)
        tilt = max(abs(phi), abs(theta))
        tilt_excess = max(tilt - cfg.tilt_soft_rad, 0.0)
        tilt_span = max(cfg.crash_max_tilt_rad - cfg.tilt_soft_rad, 1e-3)
        tilt_pen = cfg.tilt_soft_weight * min(tilt_excess / tilt_span, 1.0) ** 2

        yaw_excess = max(abs(fdm["velocities/r-rad_sec"]) - cfg.yawrate_soft_rps, 0.0)
        yaw_pen = cfg.yawrate_soft_weight * min(yaw_excess / max(cfg.yawrate_soft_rps, 1e-3), 1.0) ** 2

        soft_edge = cfg.max_horizontal_range_ft - cfg.boundary_soft_margin_ft
        margin = max(cfg.boundary_soft_margin_ft, 1e-3)
        bnd_excess = max(self._boundary_dist(fdm) - soft_edge, 0.0)
        bnd_pen = cfg.boundary_soft_weight * min(bnd_excess / margin, 1.0) ** 2

        return tilt_pen + yaw_pen + bnd_pen

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        cfg = self.cfg
        action = np.asarray(action, dtype=np.float32).reshape(ACT_DIM)
        opp_action = np.asarray(self.opponent_controller.compute_action(self),
                                dtype=np.float32).reshape(ACT_DIM)

        for _ in range(self.substeps):
            self._apply_action(self.fdm_self, self._surface_self, action)
            self._apply_action(self.fdm_opp, self._surface_opp, opp_action)
            self.fdm_self.run()
            self.fdm_opp.run()

        self.step_count += 1
        dt = self.control_dt

        g = self._geom(self.fdm_self, self.fdm_opp)
        rng_ft = g["rng"]
        cos_ata = g["cos_ata"]          # benim hiz vektorum vs LOS
        cos_aa = g["cos_aa"]            # LOS vs rakibin hiz vektoru
        cos_ata_opp = -cos_aa           # rakibin ATA'si
        cos_aa_opp = -cos_ata           # rakibin gordugu AA

        # (3) closing bir kez, dogru sekilde
        closing_fps = (self._prev_range_ft - rng_ft) / dt
        self._closing_fps = closing_fps
        self._prev_range_ft = rng_ft

        ata = math.acos(float(np.clip(cos_ata, -1.0, 1.0)))
        aa = math.acos(float(np.clip(cos_aa, -1.0, 1.0)))
        ata_opp = math.acos(float(np.clip(cos_ata_opp, -1.0, 1.0)))
        aa_opp = math.acos(float(np.clip(cos_aa_opp, -1.0, 1.0)))

        # --- R1: Gaussian ATA/AA takip odulu -------------------------
        s = cfg.track_sigma_rad
        track_self = _gauss(ata, s) * _gauss(aa, s)
        track_opp = _gauss(ata_opp, s) * _gauss(aa_opp, s)
        r_track = cfg.reward_track_weight * track_self
        r_threat = cfg.reward_threat_weight * track_opp

        # --- R2: mesafe + yaklasma -----------------------------------
        r_dist = cfg.reward_dist_weight * min(rng_ft / max(cfg.dist_ref_ft, 1e-6), 2.0)
        close_gate = 1.0 if rng_ft > cfg.too_close_ft else 0.0
        closing_n = float(np.clip(closing_fps / max(cfg.closing_ref_fps, 1e-6), -1.0, 1.0))
        r_close = cfg.reward_close_weight * closing_n * close_gate
        r_close_opp = cfg.reward_close_weight * closing_n * close_gate
        too_close = max(cfg.too_close_ft - rng_ft, 0.0) / max(cfg.too_close_ft, 1e-6)
        r_tooclose = cfg.reward_too_close_weight * (too_close ** 2)

        # --- Koni / kilitlenme ---------------------------------------
        opp_in_my_cone = self._in_cone(cos_ata, rng_ft)
        me_in_opp_cone = self._in_cone(cos_ata_opp, rng_ft)
        r_lock = cfg.reward_cone_hold if opp_in_my_cone else 0.0
        r_exposed = cfg.reward_cone_hold if me_in_opp_cone else 0.0
        if opp_in_my_cone:
            self.my_score += 1
        if me_in_opp_cone:
            self.opp_score += 1

        # --- R3: HP / WEZ hasar modeli -------------------------------
        dmg_to_opp = 0.0
        dmg_to_self = 0.0
        if opp_in_my_cone:
            prox = 1.0 - 0.5 * min(rng_ft / max(cfg.cone_range_ft, 1e-6), 1.0)
            dmg_to_opp = cfg.hp_damage_rate * prox * dt
            self.hp_opp = max(self.hp_opp - dmg_to_opp, 0.0)
        if me_in_opp_cone:
            prox = 1.0 - 0.5 * min(rng_ft / max(cfg.cone_range_ft, 1e-6), 1.0)
            dmg_to_self = cfg.hp_damage_rate * prox * dt
            self.hp_self = max(self.hp_self - dmg_to_self, 0.0)

        # --- Kontrol duzgunlugu --------------------------------------
        def _control_pen(fdm, act, prev_act):
            phi, theta, _ = self._attitude(fdm)
            tilt = abs(phi) + abs(theta)
            spin = abs(fdm["velocities/p-rad_sec"]) + abs(fdm["velocities/q-rad_sec"])
            yawr = abs(fdm["velocities/r-rad_sec"])
            jerk = float(np.sum(np.abs(act - prev_act)))
            return (cfg.reward_tilt_weight * tilt + cfg.reward_spin_weight * spin
                    + cfg.reward_yawrate_weight * yawr + cfg.reward_jerk_weight * jerk)

        control_penalty = _control_pen(self.fdm_self, action, self.prev_action_self)
        opp_control_penalty = _control_pen(self.fdm_opp, opp_action, self.prev_action_opp)

        safety_self = self._soft_safety_penalty(self.fdm_self)
        safety_opp = self._soft_safety_penalty(self.fdm_opp)

        # --- Shaped toplam -------------------------------------------
        shaped_self = (r_track - r_threat - r_dist + r_close - r_tooclose
                       + r_lock - r_exposed - control_penalty - safety_self)
        shaped_opp = (cfg.reward_track_weight * track_opp
                      - cfg.reward_threat_weight * track_self
                      - r_dist + r_close_opp - r_tooclose
                      + r_exposed - r_lock - opp_control_penalty - safety_opp)

        reward = self.shaped_weight * shaped_self
        opp_reward = self.shaped_weight * shaped_opp

        # --- Terminal olaylar (shaped_weight ile OLCEKLENMEZ) --------
        self_fault = self._hard_terminate(self.fdm_self)
        opp_fault = self._hard_terminate(self.fdm_opp)
        collided = rng_ft < cfg.min_separation_ft

        crashed = False
        terminated = False
        reset_reason = None

        if collided:
            reward -= cfg.crash_penalty
            opp_reward -= cfg.crash_penalty
            crashed = True
            terminated = True
            reset_reason = "collision"
        elif self.hp_self <= 0.0 and self.hp_opp <= 0.0:
            terminated = True
            reset_reason = "mutual_down"
        elif self.hp_opp <= 0.0:
            reward += cfg.win_bonus
            opp_reward -= cfg.win_bonus
            terminated = True
            reset_reason = "opponent_down"
        elif self.hp_self <= 0.0:
            reward -= cfg.win_bonus
            opp_reward += cfg.win_bonus
            terminated = True
            reset_reason = "self_down"
        elif self_fault is not None:
            reward -= cfg.crash_penalty
            opp_reward += cfg.opponent_fault_bonus
            crashed = True
            terminated = True
            reset_reason = f"self_{self_fault}"
        elif opp_fault is not None:
            reward += cfg.opponent_fault_bonus
            opp_reward -= cfg.crash_penalty
            terminated = True
            reset_reason = f"opponent_{opp_fault}"

        self.prev_action_self = action.copy()
        self.prev_action_opp = opp_action.copy()

        truncated = bool(self.step_count >= self.max_steps)
        if truncated and reset_reason is None:
            reset_reason = "timeout"
            hp_edge = (self.hp_self - self.hp_opp) / max(cfg.hp_initial, 1e-6)
            reward += cfg.timeout_hp_bonus * hp_edge
            opp_reward -= cfg.timeout_hp_bonus * hp_edge

        obs = self._get_obs_for(self.fdm_self, self.fdm_opp, self.prev_action_self)

        axis_self = g["axis_a"]
        axis_opp = g["axis_b"]

        info = {
            # geometri
            "range_ft": rng_ft,
            "closing_fps": closing_fps,
            "ata_deg": math.degrees(ata),
            "aa_deg": math.degrees(aa),
            "opp_ata_deg": math.degrees(ata_opp),
            "cos_ata": cos_ata,
            "cos_aa": cos_aa,
            "self_speed_fps": g["speed_a"],
            "opp_speed_fps": g["speed_b"],
            # DUZELTME (gorsellestirme): koni artik Euler acilarindan
            # degil, dogrudan angajman ekseni vektorunden cizilir.
            "self_axis": axis_self,
            "opp_axis": axis_opp,
            # durum
            "opp_in_my_cone": bool(opp_in_my_cone),
            "me_in_opp_cone": bool(me_in_opp_cone),
            "my_score": self.my_score,
            "opp_score": self.opp_score,
            "hp_self": float(self.hp_self),
            "hp_opp": float(self.hp_opp),
            "hp_initial": float(cfg.hp_initial),
            "crashed": crashed,
            "reset_reason": reset_reason,
            "shaped_weight": float(self.shaped_weight),
            "curriculum_progress": float(self.curriculum_progress),
            "opponent_name": getattr(self.opponent_controller, "name", "?"),
            # odul kirilimi (self)
            "track_reward": r_track,
            "threat_penalty": r_threat,
            "dist_penalty": r_dist,
            "close_reward": r_close,
            "tooclose_penalty": r_tooclose,
            "lock_reward": r_lock,
            "exposed_penalty": r_exposed,
            "control_penalty": control_penalty,
            "safety_penalty": safety_self,
            # odul kirilimi (opp - sadece gosterim)
            "opp_track_reward": cfg.reward_track_weight * track_opp,
            "opp_threat_penalty": cfg.reward_threat_weight * track_self,
            "opp_dist_penalty": r_dist,
            "opp_close_reward": r_close_opp,
            "opp_control_penalty": opp_control_penalty,
            "opp_safety_penalty": safety_opp,
            "self_reward": float(reward),
            "opp_reward": float(opp_reward),
            # kinematik (demo)
            "self_hdot_fps": float(self.fdm_self["velocities/h-dot-fps"]),
            "opp_hdot_fps": float(self.fdm_opp["velocities/h-dot-fps"]),
            "self_pos": self._position(self.fdm_self),
            "opp_pos": self._position(self.fdm_opp),
            "self_attitude": self._attitude(self.fdm_self),
            "opp_attitude": self._attitude(self.fdm_opp),
        }

        return obs, float(reward), terminated, truncated, info
