"""F450 quadcopter icin hedef irtifa + hedef yon (heading) takip gorevi.

f450_env.py'deki F450HoverEnv'e HIC dokunulmadan, ayri bir env olarak
eklenmistir.

DUZELTME (v4) - KRITIK KONTROL ARAYUZU HATASI:
Onceki surumler action'i `fcs/throttle-cmd-norm[i]` (motor-bazli, index'li)
property'sine yaziyordu. F450.xml'in FlightControl.xml'i incelendiginde
(ve gercek JSBSim fizigiyle ampirik test edildiginde) su ortaya cikti:
  1. Bu aircraft'in FCS'i motor throttle'larini KENDI ic mixer'i
     (Control Mixer + Effectors/ESC actuator zinciri) uzerinden
     hesapliyor; bu zincir per-motor throttle-cmd-norm[i] degerlerini
     DEGIL, SKALER fcs/throttle-cmd-norm (indexsiz, sadece [0]'a alias)
     + fcs/cmdRoll_rps/cmdPitch_rps/cmdYaw_rps (PID cikislari) degerlerini
     okuyor.
  2. fcs/cmdRoll_rps/cmdPitch_rps/cmdYaw_rps, fcs/ScasEngage kazanciyla
     capraziliyor - onceki kod bunu 0 yapiyordu, yani bu ic dongu
     TAMAMEN devre disiydi.
  Sonuc: eskiden action'in 4 boyutu da (motor sirasi ne olursa olsun)
  SADECE ortak/kolektif throttle'i (irtifayi) etkiliyordu; roll/pitch/yaw
  icin FIZIKSEL OLARAK HICBIR ETKI yoktu. Bu, hem irtifa salinimini
  (PPO tek eksenli bir sistemi kontrol etmeye calisiyordu) hem de
  "WASD/AD sadece irtifayi degistiriyor" sikayetini birebir acikliyor.

  DOGRU arayuz (ampirik olarak dogrulandi):
    fcs/aileron-cmd-norm   -> roll  (pozitif = SAGA hareket)
    fcs/elevator-cmd-norm  -> pitch (pozitif = GERIYE hareket)
    fcs/rudder-cmd-norm    -> yaw
    fcs/throttle-cmd-norm (SKALER, indexsiz) -> kolektif/dikey itki
    fcs/ScasEngage = 1     -> yukaridaki komutlarin isleyebilmesi icin
                               ZORUNLU (rate-based inner-loop'u aktif eder)

  Action semantigi, RL/manuel kontrol tarafinda sezgisel kalsin diye
  [roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd] olarak TANIMLANDI (hepsi
  -1..1, throttle_cmd hover etrafinda olceklenir). pitch_cmd pozitif =
  ILERI (sezgisel) olsun diye, elevator-cmd-norm = -pitch_cmd olarak
  ters cevriliyor (olcum: elevator+ -> geriye gidiyor).

  ONEMLI: Bu, action'in fizige BAGLANMA SEKLINI degistirir. Onceki
  egitilmis model (model_final.zip) eski/kopuk arayuzle egitildigi icin
  bu degisiklikten sonra GECERSIZ olur - YENIDEN EGITIM sart.

DUZELTME (v3) - IRTIFA SONUMLEME (damping) [v4 uzerine tasindi]:
- reward_hdot_weight: hedefe yaklastikca (progress->1) guclenen bir
  dikey-hiz sonumleme cezasi. Tirmanma basinda (progress~0) zayif -
  dinamik/yay seklinde tirmanmaya izin verir; hedefe yaklasinca guclu -
  gercekten OTURMAYI ogretir. Bu, "sadece pozisyon hatasina ceza var,
  hiz cezasi yok" eksikliginden kaynaklanan salinimi giderir.
- success artik SADECE irtifa toleransina degil, dusuk dikey hiza da
  bakiyor (success_hdot_tol_fps) - drone hedef bandi hizla GECEREK
  degil, gercekten YAVASLAYIP OTURARAK basari kazaniyor.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import jsbsim


class F450FlightEnv(gym.Env):
    """JSBSim F450 modeli uzerinde: hedef irtifaya TIRMAN, tirmanirken
    yavasca hedef yone (dunya cercevesinde, pusula konvansiyonu:
    0=Kuzey, 90=Dogu) don, hedef irtifaya ulasip OTURUNCA episode'u
    basariyla bitir (reset) gorevi.

    Action (4,): [roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], hepsi
    yaklasik -1..1 araliginda. Gercek FCS komutlarina donusum icin
    step() icindeki aciklamaya bakin.

    Her episode'da target_altitude VE target_heading RASTGELE secilir
    (F450HoverEnv'de target_altitude sabitti). Model bu ikisini gozlem
    olarak alir (alt_err + sin/cos(heading)), yani 'goal-conditioned' bir
    politika ogrenir.
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
        reward_heading_weight=0.08,
        reward_tilt_weight=0.05,
        reward_spin_weight=0.10,
        reward_jerk_weight=0.05,
        crash_penalty=50.0,
        crash_min_alt_ft=1.0,
        crash_max_alt_offset_ft=60.0,
        crash_max_tilt_rad=1.0,
        altitude_start_offset_ft=25.0,
        altitude_start_jitter_ft=2.0,
        success_alt_tol_ft=1.5,
        success_hold_seconds=1.0,
        success_bonus=20.0,
        # --- irtifa sonumleme (damping) parametreleri ---
        reward_hdot_weight=0.12,
        hdot_damping_min_factor=0.3,
        success_hdot_tol_fps=1.0,
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

        # Gozlem: alt_err, hdot, along_track, cross_track, roll, pitch,
        #         p, q, r, sin(heading), cos(heading), prev_action(4) = 15
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
        self._initial_alt_err_ft = 1.0  # reset()'te gercek degerle guncellenir

        self.reward_hdot_weight = reward_hdot_weight
        self.hdot_damping_min_factor = hdot_damping_min_factor
        self.success_hdot_tol_fps = success_hdot_tol_fps

        # Her episode'da rastgele secilecek hedefler - reset()'te doldurulur
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

        # DUZELTME (v4): ScasEngage=1 OLMAK ZORUNDA. 0 iken aileron/
        # elevator/rudder komutlari FCS'in ic PID zincirinde (gain=
        # ScasEngage) sifirlaniyor ve motorlara HICBIR farkli komut
        # ulasmiyordu - eskiden buradaki deger 0'di, bu YUZDEN roll/
        # pitch fiziksel olarak calismiyordu.
        self.fdm["fcs/ScasEngage"] = 1

        # Kontrol yuzeylerini notr, kolektif throttle'i hover'a ayarla.
        # DUZELTME (v4): artik per-motor fcs/throttle-cmd-norm[i] DEGIL,
        # aircraft'in gercekten okudugu SKALER fcs/throttle-cmd-norm.
        self.fdm["fcs/aileron-cmd-norm"] = 0.0
        self.fdm["fcs/elevator-cmd-norm"] = 0.0
        self.fdm["fcs/rudder-cmd-norm"] = 0.0
        self.fdm["fcs/throttle-cmd-norm"] = self.hover_throttle

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
        """0 (hala baslangic irtifasinda) -> 1 (hedef irtifaya ulasti)
        arasinda bir ilerleme skoru. Hem heading odulunu hem de h-dot
        sonumleme cezasini bununla carparak: drone tirmanirken odul/ceza
        zayif (dinamik/yay hareketine izin ver), hedefe yaklastikca
        guclu (yon dogrulugu VE irtifa sonumleme onceliklenir)."""
        progress = 1.0 - (alt_err_ft / self._initial_alt_err_ft)
        return float(np.clip(progress, 0.0, 1.0))

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
        # Gercek motor pos-norm degerleri (JSBSim engine sirasi:
        # 0=front right, 1=aft left, 2=front left, 3=aft right - bkz.
        # Propulsion.xml). Artik action'dan TAHMIN edilmiyor, dogrudan
        # FCS'in gercekten uyguladigi degerler okunuyor.
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
        return (
            alt < self.crash_min_alt_ft
            or alt > self.target_altitude + self.crash_max_alt_offset_ft
            or abs(self.fdm["attitude/phi-rad"]) > self.crash_max_tilt_rad
            or abs(self.fdm["attitude/theta-rad"]) > self.crash_max_tilt_rad
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(4)
        roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd = action

        # DUZELTME (v4): aircraft'in GERCEKTEN okudugu FCS komutlari.
        # aileron/elevator/rudder-cmd-norm dogrudan -1..1 araliginda
        # (JSBSim'in pilotRoll/Pitch/Yaw_norm summer'lari zaten bu
        # araliga clip ediyor). elevator-cmd-norm = -pitch_cmd: cunku
        # ampirik testte elevator POZITIF verildiginde drone GERIYE
        # gidiyor (u-fps negatif oluyor) - pitch_cmd'i "ileri=pozitif"
        # sezgisiyle tanimladigimiz icin burada isareti ceviriyoruz.
        # aileron-cmd-norm = +roll_cmd: ampirik testte aileron POZITIF
        # verildiginde drone SAGA gidiyor (v-fps pozitif) - roll_cmd
        # "saga=pozitif" sezgisiyle zaten dogrudan uyuyor.
        aileron = float(np.clip(roll_cmd, -1.0, 1.0))
        elevator = float(np.clip(-pitch_cmd, -1.0, 1.0))
        rudder = float(np.clip(yaw_cmd, -1.0, 1.0))
        throttle = float(np.clip(
            self.hover_throttle + throttle_cmd * self.throttle_range, 0.0, 1.0
        ))

        for _ in range(self.substeps):
            self.fdm["fcs/aileron-cmd-norm"] = aileron
            self.fdm["fcs/elevator-cmd-norm"] = elevator
            self.fdm["fcs/rudder-cmd-norm"] = rudder
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
        jerk = float(np.sum(np.abs(action - self.prev_action)))

        # irtifa sonumleme (damping) cezasi - progress ile guclenir.
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
            - self.reward_jerk_weight * jerk
        )

        crashed = self._is_crashed()
        if crashed:
            reward -= self.crash_penalty

        # basari: irtifa toleransi ICINDE VE dikey hiz yeterince dusuk
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
