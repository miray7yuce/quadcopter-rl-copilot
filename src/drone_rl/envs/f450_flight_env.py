"""F450 quadcopter icin hedef irtifa + hedef yon (heading) takip gorevi.

DUZELTME (v6):
1. max_horizontal_range_ft: YENI bir crash kosulu - drone artik
   baslangic noktasindan belirli bir yatay mesafenin (varsayilan 90ft)
   OTESINE GECEMEZ (gecerse crash sayilir). Iki amaci var:
   (a) egitim: durum uzayini sinirlayarak ogrenmeyi kolaylastirir,
   (b) gorsellestirme: simulator ekranindaki grid artik SABIT ve bu
   sinirla eslesecek boyutta cizilebilir - drone'un gridi asmasi
   FIZIKSEL OLARAK IMKANSIZ hale gelir (kozmetik/kayan-grid hack'i
   yerine gercek bir garanti).
2. reward_heading_weight ve reward_jerk_weight guclendirildi (config'te,
   varsayilan degerler yukseltildi) - yon takibinin daha amacli
   gorunmesi ve komut seviyesinde ek puruzsuzluk icin.

ONEMLI: max_horizontal_range_ft yeni bir TERMINATION kosulu oldugu
icin onceki egitimli model (flight_ppo_v5) bu ortamla FIZIKSEL OLARAK
uyumsuzdur (farkli bir gorev/state-space) - YENIDEN EGITIM sarttir.

--- (v5 notlari, hala gecerli) ---
reward_yawrate_weight (yaw acisal hizina ceza), roll/pitch/yaw_authority
(kontrol yetkisi kisitlama), control_surface_tau_s (yuzey yumusatma/
slew-rate), crash_max_tilt_rad=0.6, crash_max_yawrate_rps (guvenlik agi).

--- (v4 notlari, hala gecerli) ---
Gercek kontrol arayuzu: fcs/aileron-cmd-norm (roll), fcs/elevator-cmd-norm
(pitch, TERS isaretli: elevator=-pitch_cmd), fcs/rudder-cmd-norm (yaw),
fcs/throttle-cmd-norm SKALER (kolektif) - fcs/ScasEngage=1 sart.
Action: [roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], pitch_cmd
pozitif=ILERI, roll_cmd pozitif=SAGA.

--- (v3 notlari, hala gecerli) ---
reward_hdot_weight: hedefe yaklastikca guclenen irtifa sonumleme cezasi.
success artik hem irtifa toleransi hem dusuk dikey hiz gerektirir.
"""

import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import jsbsim

FT_PER_M = 3.28084


class F450FlightEnv(gym.Env):
    """JSBSim F450 modeli uzerinde: hedef irtifaya TIRMAN, tirmanirken
    yavasca hedef yone don, hedef irtifaya ulasip OTURUNCA episode'u
    basariyla bitir (reset) gorevi. Yatay hareket, max_horizontal_range_ft
    ile SINIRLANDIRILMISTIR (v6).

    Action (4,): [roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], -1..1.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        target_altitude_min_ft=20.0,
        target_altitude_max_ft=45.0,
        target_speed_fps=6.0,
        episode_seconds=60.0,
        physics_hz=240,
        control_hz=20,
        hover_throttle=0.420,
        throttle_range=0.25,
        reward_alt_weight=0.10,
        reward_heading_weight=0.14,
        reward_tilt_weight=0.05,
        reward_spin_weight=0.10,
        reward_jerk_weight=0.08,
        crash_penalty=50.0,
        crash_min_alt_ft=1.0,
        crash_max_alt_offset_ft=60.0,
        crash_max_tilt_rad=0.6,
        altitude_start_offset_ft=25.0,
        altitude_start_jitter_ft=2.0,
        success_alt_tol_ft=1.5,
        success_hold_seconds=1.0,
        success_bonus=20.0,
        reward_hdot_weight=0.12,
        hdot_damping_min_factor=0.3,
        success_hdot_tol_fps=1.0,
        reward_yawrate_weight=0.06,
        roll_authority=0.6,
        pitch_authority=0.6,
        yaw_authority=0.45,
        control_surface_tau_s=0.08,
        crash_max_yawrate_rps=20.0,
        # --- YENI (v6) ---
        max_horizontal_range_ft=90.0,
    ):
        super().__init__()

        physics_hz = int(physics_hz)
        control_hz = int(control_hz)
        if physics_hz <= 0 or control_hz <= 0:
            raise ValueError("physics_hz ve control_hz pozitif olmali")
        if physics_hz % control_hz != 0:
            raise ValueError(
                f"physics_hz ({physics_hz}) control_hz'e ({control_hz}) tam "
                "bolunmeli. Or: 240/20=12 OK, 240/50 HATALI."
            )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)

        self.target_altitude_min_ft = target_altitude_min_ft
        self.target_altitude_max_ft = target_altitude_max_ft
        self.target_speed_fps = target_speed_fps

        self.physics_hz = physics_hz
        self.physics_dt = 1.0 / physics_hz
        self.control_hz = control_hz
        self.substeps = physics_hz // control_hz
        self.max_steps = int(episode_seconds * control_hz)

        self.hover_throttle = hover_throttle
        self.throttle_range = throttle_range

        self.reward_alt_weight = reward_alt_weight
        self.reward_heading_weight = reward_heading_weight
        self.reward_tilt_weight = reward_tilt_weight
        self.reward_spin_weight = reward_spin_weight
        self.reward_jerk_weight = reward_jerk_weight
        self.crash_penalty = crash_penalty
        self.crash_min_alt_ft = crash_min_alt_ft
        self.crash_max_alt_offset_ft = crash_max_alt_offset_ft
        self.crash_max_tilt_rad = crash_max_tilt_rad

        self.altitude_start_offset_ft = altitude_start_offset_ft
        self.altitude_start_jitter_ft = altitude_start_jitter_ft
        self.success_alt_tol_ft = success_alt_tol_ft
        self.success_hold_steps = max(1, int(success_hold_seconds * control_hz))
        self.success_bonus = success_bonus
        self._success_counter = 0
        self._initial_alt_err_ft = 1.0

        self.reward_hdot_weight = reward_hdot_weight
        self.hdot_damping_min_factor = hdot_damping_min_factor
        self.success_hdot_tol_fps = success_hdot_tol_fps

        self.reward_yawrate_weight = reward_yawrate_weight
        self.roll_authority = float(np.clip(roll_authority, 0.0, 1.0))
        self.pitch_authority = float(np.clip(pitch_authority, 0.0, 1.0))
        self.yaw_authority = float(np.clip(yaw_authority, 0.0, 1.0))
        self.control_surface_tau_s = max(control_surface_tau_s, 1e-4)
        self.crash_max_yawrate_rps = crash_max_yawrate_rps
        self._surface_state = np.zeros(3, dtype=np.float64)

        # --- YENI (v6) ---
        self.max_horizontal_range_ft = max_horizontal_range_ft

        self.target_altitude = (target_altitude_min_ft + target_altitude_max_ft) / 2.0
        self.target_heading = 0.0

        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_debug_level(0)
        if not self.fdm.load_model("F450"):
            raise RuntimeError("F450 modeli yuklenemedi")
        self.fdm.set_dt(self.physics_dt)

        self.step_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)

    @property
    def control_dt(self):
        return self.substeps * self.physics_dt

    def _apply_initial_conditions(self):
        jitter = self.np_random.uniform(
            -self.altitude_start_jitter_ft, self.altitude_start_jitter_ft
        )
        h0 = self.target_altitude - self.altitude_start_offset_ft + jitter
        h0 = max(h0, self.crash_min_alt_ft + 3.0)
        self.fdm["ic/h-agl-ft"] = h0

        self.fdm["ic/u-fps"] = self.np_random.uniform(-1.0, 1.0)
        self.fdm["ic/v-fps"] = self.np_random.uniform(-1.0, 1.0)
        self.fdm["ic/w-fps"] = self.np_random.uniform(-1.0, 1.0)
        self.fdm["ic/phi-rad"] = self.np_random.uniform(-0.05, 0.05)
        self.fdm["ic/theta-rad"] = self.np_random.uniform(-0.05, 0.05)
        self.fdm["ic/psi-true-rad"] = 0.0

        return abs(h0 - self.target_altitude)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.target_altitude = self.np_random.uniform(
            self.target_altitude_min_ft, self.target_altitude_max_ft
        )
        self.target_heading = self.np_random.uniform(0.0, 2.0 * np.pi)

        initial_alt_err = self._apply_initial_conditions()
        self._initial_alt_err_ft = max(initial_alt_err, 1e-3)
        self.fdm.run_ic()

        for i in range(4):
            self.fdm[f"propulsion/engine[{i}]/set-running"] = 1

        self.fdm["fcs/ScasEngage"] = 1
        self.fdm["fcs/aileron-cmd-norm"] = 0.0
        self.fdm["fcs/elevator-cmd-norm"] = 0.0
        self.fdm["fcs/rudder-cmd-norm"] = 0.0
        self.fdm["fcs/throttle-cmd-norm"] = self.hover_throttle

        self._surface_state[:] = 0.0

        self.step_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self._success_counter = 0

        return self._get_obs(), {}

    def _world_frame_velocity(self):
        f = self.fdm
        u = f["velocities/u-fps"]
        v = f["velocities/v-fps"]
        psi = f["attitude/psi-rad"]
        north_vel = u * np.cos(psi) - v * np.sin(psi)
        east_vel = u * np.sin(psi) + v * np.cos(psi)
        return north_vel, east_vel

    def _along_cross_track(self):
        north_vel, east_vel = self._world_frame_velocity()
        h = self.target_heading
        along = north_vel * np.cos(h) + east_vel * np.sin(h)
        cross = -north_vel * np.sin(h) + east_vel * np.cos(h)
        return along, cross

    def _climb_progress(self, alt_err_ft):
        progress = 1.0 - (alt_err_ft / self._initial_alt_err_ft)
        return float(np.clip(progress, 0.0, 1.0))

    def _horizontal_dist_ft(self):
        x_m = self.fdm["position/distance-from-start-lon-mt"]
        y_m = self.fdm["position/distance-from-start-lat-mt"]
        return math.hypot(x_m, y_m) * FT_PER_M

    def _get_obs(self):
        f = self.fdm
        alt_err = (f["position/h-agl-ft"] - self.target_altitude) / 10.0
        hdot = f["velocities/h-dot-fps"] / 10.0

        along, cross = self._along_cross_track()
        along_n = along / 10.0
        cross_n = cross / 10.0

        roll = f["attitude/phi-rad"]
        pitch = f["attitude/theta-rad"]
        p = f["velocities/p-rad_sec"] / 5.0
        q = f["velocities/q-rad_sec"] / 5.0
        r = f["velocities/r-rad_sec"] / 5.0

        sin_h = np.sin(self.target_heading)
        cos_h = np.cos(self.target_heading)

        return np.array(
            [alt_err, hdot, along_n, cross_n, roll, pitch, p, q, r,
             sin_h, cos_h, *self.prev_action],
            dtype=np.float32,
        )

    def _get_telemetry(self, crashed, reached):
        f = self.fdm
        alt_agl_ft = float(f["position/h-agl-ft"])
        along, cross = self._along_cross_track()
        motor_throttle = [float(f[f"fcs/throttle-pos-norm[{i}]"]) for i in range(4)]
        return {
            "alt_ft": alt_agl_ft,
            "alt_sl_ft": float(f["position/h-sl-ft"]),
            "alt_err_ft": abs(alt_agl_ft - self.target_altitude),
            "hdot_fps": float(f["velocities/h-dot-fps"]),
            "lat_deg": float(f["position/lat-geod-deg"]),
            "lon_deg": float(f["position/long-gc-deg"]),
            "x_m": float(f["position/distance-from-start-lon-mt"]),
            "y_m": float(f["position/distance-from-start-lat-mt"]),
            "roll_rad": float(f["attitude/phi-rad"]),
            "pitch_rad": float(f["attitude/theta-rad"]),
            "yaw_rad": float(f["attitude/psi-rad"]),
            "yaw_rate_rps": float(f["velocities/r-rad_sec"]),
            "target_heading_rad": float(self.target_heading),
            "along_track_fps": float(along),
            "cross_track_fps": float(cross),
            "target_speed_fps": float(self.target_speed_fps),
            "motor_throttle": motor_throttle,
            "crashed": bool(crashed),
            "reached_target": bool(reached),
        }

    def _is_crashed(self):
        alt = self.fdm["position/h-agl-ft"]
        yaw_rate = abs(self.fdm["velocities/r-rad_sec"])
        return (
            alt < self.crash_min_alt_ft
            or alt > self.target_altitude + self.crash_max_alt_offset_ft
            or abs(self.fdm["attitude/phi-rad"]) > self.crash_max_tilt_rad
            or abs(self.fdm["attitude/theta-rad"]) > self.crash_max_tilt_rad
            or yaw_rate > self.crash_max_yawrate_rps
            # YENI (v6): yatay sinir asimi da crash sayilir.
            or self._horizontal_dist_ft() > self.max_horizontal_range_ft
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(4)
        roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd = action

        aileron_target = float(np.clip(roll_cmd * self.roll_authority, -1.0, 1.0))
        elevator_target = float(np.clip(-pitch_cmd * self.pitch_authority, -1.0, 1.0))
        rudder_target = float(np.clip(yaw_cmd * self.yaw_authority, -1.0, 1.0))
        throttle = float(np.clip(
            self.hover_throttle + throttle_cmd * self.throttle_range, 0.0, 1.0
        ))

        targets = np.array([aileron_target, elevator_target, rudder_target])
        alpha = self.physics_dt / (self.control_surface_tau_s + self.physics_dt)

        for _ in range(self.substeps):
            self._surface_state += alpha * (targets - self._surface_state)
            self.fdm["fcs/aileron-cmd-norm"] = float(self._surface_state[0])
            self.fdm["fcs/elevator-cmd-norm"] = float(self._surface_state[1])
            self.fdm["fcs/rudder-cmd-norm"] = float(self._surface_state[2])
            self.fdm["fcs/throttle-cmd-norm"] = throttle
            self.fdm.run()

        self.step_count += 1
        obs = self._get_obs()

        alt_err_ft = abs(self.fdm["position/h-agl-ft"] - self.target_altitude)
        hdot_fps = self.fdm["velocities/h-dot-fps"]
        along, cross = self._along_cross_track()

        progress = self._climb_progress(alt_err_ft)
        heading_err = abs(self.target_speed_fps - along) + abs(cross)

        tilt = abs(self.fdm["attitude/phi-rad"]) + abs(self.fdm["attitude/theta-rad"])
        spin = abs(self.fdm["velocities/p-rad_sec"]) + abs(self.fdm["velocities/q-rad_sec"])
        yaw_rate_penalty = abs(self.fdm["velocities/r-rad_sec"])
        jerk = float(np.sum(np.abs(action - self.prev_action)))

        damping_factor = self.hdot_damping_min_factor + (
            1.0 - self.hdot_damping_min_factor
        ) * progress
        hdot_penalty = self.reward_hdot_weight * damping_factor * abs(hdot_fps)

        reward = (
            1.0
            - self.reward_alt_weight * alt_err_ft
            - hdot_penalty
            - self.reward_heading_weight * progress * heading_err
            - self.reward_tilt_weight * tilt
            - self.reward_spin_weight * spin
            - self.reward_yawrate_weight * yaw_rate_penalty
            - self.reward_jerk_weight * jerk
        )

        crashed = self._is_crashed()
        if crashed:
            reward -= self.crash_penalty

        alt_ok = alt_err_ft < self.success_alt_tol_ft
        hdot_ok = abs(hdot_fps) < self.success_hdot_tol_fps
        if alt_ok and hdot_ok:
            self._success_counter += 1
        else:
            self._success_counter = 0

        reached = self._success_counter >= self.success_hold_steps
        if reached:
            reward += self.success_bonus

        info = self._get_telemetry(crashed, reached)

        self.prev_action = action.copy()

        terminated = bool(crashed or reached)
        truncated = bool(self.step_count >= self.max_steps)

        return obs, float(reward), terminated, truncated, info
