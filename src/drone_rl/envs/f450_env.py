"""F450 quadcopter icin hover gorevi ortami."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import jsbsim


class F450HoverEnv(gym.Env):
    """JSBSim F450 modeli uzerinde sabit irtifada durma (hover) gorevi.

    Tum sayisal ayarlar (hedef irtifa, hover gazi, odul agirliklari,
    crash esikleri) constructor parametreleridir ve
    configs/ppo_hover.yaml + drone_rl.config.load_config ile
    doldurulabilir. Parametre verilmezse, refactor-oncesi kodda
    hardcoded olan degerlerle AYNI varsayilanlar kullanilir.

    step() her adimda info sozlugunde ham telemetri dondurur. Bunun
    sebebi: VecEnv'ler episode bittiginde alt ortami OTOMATIK reset
    eder, dolayisiyla disaridan `env.fdm[...]` okuyan kod son adimda
    zaten resetlenmis (yeni episode'un IC'leri) degerleri gorur.
    info ise reset'ten etkilenmez, dogru kareyi tasir.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        target_altitude_ft=30.0,
        episode_seconds=20.0,
        physics_hz=240,
        control_hz=20,
        hover_throttle=0.420,
        throttle_range=0.25,
        reward_alt_weight=0.10,
        reward_tilt_weight=0.50,
        reward_spin_weight=0.10,
        reward_jerk_weight=0.05,
        crash_penalty=50.0,
        crash_min_alt_ft=1.0,
        crash_max_alt_offset_ft=60.0,
        crash_max_tilt_rad=1.0,
    ):
        super().__init__()

        physics_hz = int(physics_hz)
        control_hz = int(control_hz)
        if physics_hz <= 0 or control_hz <= 0:
            raise ValueError("physics_hz ve control_hz pozitif olmali")
        if physics_hz % control_hz != 0:
            # Aksi halde substeps asagi yuvarlanir, gercek kontrol frekansi
            # istenenden farkli olur ama max_steps hala istenen degere gore
            # hesaplanir -> episode suresi sessizce yanlis olur.
            raise ValueError(
                f"physics_hz ({physics_hz}) control_hz'e ({control_hz}) tam "
                "bolunmeli. Or: 240/20=12 OK, 240/50 HATALI."
            )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32)

        self.target_altitude = target_altitude_ft
        self.physics_hz = physics_hz
        self.physics_dt = 1.0 / physics_hz
        self.control_hz = control_hz
        self.substeps = physics_hz // control_hz
        self.max_steps = int(episode_seconds * control_hz)

        self.hover_throttle = hover_throttle
        self.throttle_range = throttle_range

        self.reward_alt_weight = reward_alt_weight
        self.reward_tilt_weight = reward_tilt_weight
        self.reward_spin_weight = reward_spin_weight
        self.reward_jerk_weight = reward_jerk_weight
        self.crash_penalty = crash_penalty
        self.crash_min_alt_ft = crash_min_alt_ft
        self.crash_max_alt_offset_ft = crash_max_alt_offset_ft
        self.crash_max_tilt_rad = crash_max_tilt_rad

        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_debug_level(0)
        if not self.fdm.load_model("F450"):
            raise RuntimeError("F450 modeli yuklenemedi")
        self.fdm.set_dt(self.physics_dt)

        self.step_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)

    @property
    def control_dt(self):
        """Bir 'step()' cagrisinin temsil ettigi sure (saniye).
        evaluate.py gibi diger kodun control_hz'i elle tekrar
        hesaplamasina gerek kalmaz, dogrudan buradan okur."""
        return self.substeps * self.physics_dt

    def _apply_initial_conditions(self):
        h0 = self.target_altitude + self.np_random.uniform(-3.0, 3.0)
        self.fdm["ic/h-agl-ft"] = h0
        self.fdm["ic/u-fps"] = self.np_random.uniform(-1.0, 1.0)
        self.fdm["ic/v-fps"] = self.np_random.uniform(-1.0, 1.0)
        self.fdm["ic/w-fps"] = self.np_random.uniform(-1.0, 1.0)
        self.fdm["ic/phi-rad"] = self.np_random.uniform(-0.05, 0.05)
        self.fdm["ic/theta-rad"] = self.np_random.uniform(-0.05, 0.05)
        self.fdm["ic/psi-true-rad"] = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._apply_initial_conditions()
        self.fdm.run_ic()

        for i in range(4):
            self.fdm[f"propulsion/engine[{i}]/set-running"] = 1
        self.fdm["fcs/ScasEngage"] = 0

        for i in range(4):
            self.fdm[f"fcs/throttle-cmd-norm[{i}]"] = self.hover_throttle

        self.step_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)

        return self._get_obs(), {}

    def _get_obs(self):
        f = self.fdm
        alt_err = (f["position/h-agl-ft"] - self.target_altitude) / 10.0
        hdot = f["velocities/h-dot-fps"] / 10.0
        u = f["velocities/u-fps"] / 10.0
        v = f["velocities/v-fps"] / 10.0
        roll = f["attitude/phi-rad"]
        pitch = f["attitude/theta-rad"]
        p = f["velocities/p-rad_sec"] / 5.0
        q = f["velocities/q-rad_sec"] / 5.0
        r = f["velocities/r-rad_sec"] / 5.0

        return np.array(
            [alt_err, hdot, u, v, roll, pitch, p, q, r, *self.prev_action],
            dtype=np.float32,
        )

    def _get_telemetry(self, crashed):
        """Disaridaki kayit/gorsellestirme kodu icin ham durum.
        VecEnv auto-reset'inden ONCE, step() icinde toplanir."""
        f = self.fdm
        alt_agl_ft = float(f["position/h-agl-ft"])
        return {
            "alt_ft": alt_agl_ft,
            "alt_sl_ft": float(f["position/h-sl-ft"]),      # ACMI icin MSL irtifa
            "alt_err_ft": abs(alt_agl_ft - self.target_altitude),
            "lat_deg": float(f["position/lat-geod-deg"]),
            "lon_deg": float(f["position/long-gc-deg"]),
            "roll_rad": float(f["attitude/phi-rad"]),
            "pitch_rad": float(f["attitude/theta-rad"]),
            "yaw_rad": float(f["attitude/psi-rad"]),
            "crashed": bool(crashed),
        }

    def _is_crashed(self):
        alt = self.fdm["position/h-agl-ft"]
        return (
            alt < self.crash_min_alt_ft
            or alt > self.target_altitude + self.crash_max_alt_offset_ft
            or abs(self.fdm["attitude/phi-rad"]) > self.crash_max_tilt_rad
            or abs(self.fdm["attitude/theta-rad"]) > self.crash_max_tilt_rad
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(4)

        throttles = np.clip(
            self.hover_throttle + action * self.throttle_range, 0.0, 1.0
        )

        for _ in range(self.substeps):
            for i in range(4):
                self.fdm[f"fcs/throttle-cmd-norm[{i}]"] = float(throttles[i])
            self.fdm.run()

        self.step_count += 1
        obs = self._get_obs()

        alt_err_ft = abs(self.fdm["position/h-agl-ft"] - self.target_altitude)
        tilt = abs(self.fdm["attitude/phi-rad"]) + abs(self.fdm["attitude/theta-rad"])
        spin = abs(self.fdm["velocities/p-rad_sec"]) + abs(self.fdm["velocities/q-rad_sec"])
        jerk = float(np.sum(np.abs(action - self.prev_action)))

        reward = (
            1.0
            - self.reward_alt_weight * alt_err_ft
            - self.reward_tilt_weight * tilt
            - self.reward_spin_weight * spin
            - self.reward_jerk_weight * jerk
        )

        crashed = self._is_crashed()
        if crashed:
            reward -= self.crash_penalty

        info = self._get_telemetry(crashed)

        self.prev_action = action.copy()

        terminated = bool(crashed)
        truncated = bool(self.step_count >= self.max_steps)

        return obs, float(reward), terminated, truncated, info
