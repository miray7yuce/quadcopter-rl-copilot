"""Iki F450 arasinda 'dogfight' gorevi - birbirini kovalayip radar
konisine alma. TAM 3D fizik.

v5: Stage B'de rakip HER reset()'te havuzdan yeniden orneklenir.
v4: KRITIK konum duzeltmesi - mutlak enlem/boylam (ic/lat-gc-deg,
    ic/long-gc-deg) kullaniliyor. Eskiden "distance-from-start-*"
    property'leri kullaniliyordu, bunlar HER FDM'IN KENDI baslangicina
    gore olcum yapiyordu (iki FDM arasi PAYLASILAN referans DEGIL) -
    iki drone pratikte HEP ayni noktada spawn oluyordu. Ampirik JSBSim
    testiyle dogrulanan duzeltme.
v3: info dict'e her iki drone icin kinematik + odul kirilimi eklendi.

--- KIM RL ILE CALISIYOR? ---
- fdm_self: HER ZAMAN dis taraftan (PPO) gelen action ile suruluyor.
- fdm_opp: self.opponent_controller uzerinden - Stage A'da scripted
  (RL degil), Stage B'de dondurulmus/inference-only bir PPO modeli.
- reward SADECE fdm_self icin hesaplanir, opp hicbir zaman bu adimda
  ogrenmez (frozen opponent self-play deseni).
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


class BaseOpponentController:
    def reset(self):
        pass

    def compute_action(self, env: "DogfightEnv") -> np.ndarray:
        raise NotImplementedError


class ScriptedCircleOpponent(BaseOpponentController):
    def __init__(self, bank_deg=15.0, kp_roll=3.5, kd_roll=0.3, kp_alt=0.12, kd_alt=0.35):
        self.bank_deg = bank_deg
        self.kp_roll = kp_roll
        self.kd_roll = kd_roll
        self.kp_alt = kp_alt
        self.kd_alt = kd_alt
        self._target_alt_ft = None

    def reset(self):
        self._target_alt_ft = None

    def compute_action(self, env: "DogfightEnv") -> np.ndarray:
        f = env.fdm_opp
        if self._target_alt_ft is None:
            self._target_alt_ft = f["position/h-agl-ft"]

        roll_now = f["attitude/phi-rad"]
        p_now = f["velocities/p-rad_sec"]
        target_roll = math.radians(self.bank_deg)
        roll_cmd = float(np.clip(
            self.kp_roll * (target_roll - roll_now) - self.kd_roll * p_now, -1.0, 1.0
        ))
        pitch_cmd = 0.0

        alt_err = self._target_alt_ft - f["position/h-agl-ft"]
        hdot = f["velocities/h-dot-fps"]
        throttle_cmd = float(np.clip(self.kp_alt * alt_err - self.kd_alt * hdot, -1.0, 1.0))
        yaw_cmd = 0.0

        return np.array([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=np.float32)


class NormalizerStats:
    def __init__(self, vecnorm):
        self.mean = vecnorm.obs_rms.mean.astype(np.float32)
        self.var = vecnorm.obs_rms.var.astype(np.float32)
        self.epsilon = vecnorm.epsilon
        self.clip_obs = vecnorm.clip_obs

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        normed = (obs - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normed, -self.clip_obs, self.clip_obs).astype(np.float32)


class PPOOpponentController(BaseOpponentController):
    def __init__(self, model, stats: NormalizerStats):
        self.model = model
        self.stats = stats

    def compute_action(self, env: "DogfightEnv") -> np.ndarray:
        obs = env._get_obs_for(env.fdm_opp, env.fdm_self, env.prev_action_opp)
        norm_obs = self.stats.normalize(obs).reshape(1, -1)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        return action[0]


class DogfightEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg, opponent_controller: Optional[BaseOpponentController] = None,
                 opponent_pool=None, opponent_latest_prob: float = 0.7):
        super().__init__()
        self.cfg = cfg

        self.physics_hz = int(cfg.physics_hz)
        self.control_hz = int(cfg.control_hz)
        self.physics_dt = 1.0 / self.physics_hz
        self.substeps = self.physics_hz // self.control_hz
        self.max_steps = int(cfg.episode_seconds * self.control_hz)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float32)

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

        self.opponent_controller = opponent_controller or ScriptedCircleOpponent()
        self.opponent_pool = opponent_pool
        self.opponent_latest_prob = opponent_latest_prob

        self._surface_self = np.zeros(3, dtype=np.float64)
        self._surface_opp = np.zeros(3, dtype=np.float64)
        self.prev_action_self = np.zeros(4, dtype=np.float32)
        self.prev_action_opp = np.zeros(4, dtype=np.float32)

        self.step_count = 0
        self.my_score = 0
        self.opp_score = 0
        self._prev_range_ft = None

        self.standoff_weight = cfg.standoff_weight_start

    @property
    def control_dt(self):
        return self.substeps * self.physics_dt

    def set_standoff_weight(self, w: float):
        self.standoff_weight = float(w)

    def set_opponent_controller(self, controller: BaseOpponentController):
        self.opponent_controller = controller

    def set_opponent_controller_from_pool(self, model_vecnorm_tuple):
        from drone_rl.dogfight.env_factory import load_opponent_controller
        model_path, vecnorm_path = model_vecnorm_tuple
        self.opponent_controller = load_opponent_controller(model_path, vecnorm_path)

    def _init_fdm(self, fdm, alt_ft, heading_deg, north_ft=0.0, east_ft=0.0):
        ft_per_deg_lon = FT_PER_DEG_LAT * math.cos(math.radians(LAT0_DEG))
        fdm["ic/lat-gc-deg"] = LAT0_DEG + north_ft / FT_PER_DEG_LAT
        fdm["ic/long-gc-deg"] = LON0_DEG + east_ft / ft_per_deg_lon
        fdm["ic/h-agl-ft"] = alt_ft
        fdm["ic/u-fps"] = 0.0
        fdm["ic/v-fps"] = 0.0
        fdm["ic/w-fps"] = 0.0
        fdm["ic/phi-rad"] = 0.0
        fdm["ic/theta-rad"] = 0.0
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

        if self.opponent_pool is not None:
            sampled = self.opponent_pool.sample(self.opponent_latest_prob, rng=self.np_random)
            if sampled is not None:
                from drone_rl.dogfight.env_factory import load_opponent_controller
                self.opponent_controller = load_opponent_controller(*sampled)

        rng = self.np_random
        rng_range = rng.uniform(cfg.spawn_range_min_ft, cfg.spawn_range_max_ft)
        bearing_deg = rng.uniform(0.0, 360.0)
        alt_self = cfg.base_altitude_ft + rng.uniform(-cfg.altitude_jitter_ft, cfg.altitude_jitter_ft)
        alt_opp = cfg.base_altitude_ft + rng.uniform(-cfg.altitude_jitter_ft, cfg.altitude_jitter_ft)
        heading_self = rng.uniform(0.0, 360.0)
        heading_opp = rng.uniform(0.0, 360.0)

        dn_target = rng_range * math.cos(math.radians(bearing_deg))
        de_target = rng_range * math.sin(math.radians(bearing_deg))

        self._init_fdm(self.fdm_self, alt_self, heading_self, north_ft=0.0, east_ft=0.0)
        self._init_fdm(self.fdm_opp, alt_opp, heading_opp, north_ft=dn_target, east_ft=de_target)

        self._surface_self[:] = 0.0
        self._surface_opp[:] = 0.0
        self.prev_action_self = np.zeros(4, dtype=np.float32)
        self.prev_action_opp = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.my_score = 0
        self.opp_score = 0
        self.opponent_controller.reset()

        rng_ft, _, _, _, _ = self._relative_geom(self.fdm_self, self.fdm_opp)
        self._prev_range_ft = rng_ft

        return self._get_obs_for(self.fdm_self, self.fdm_opp, self.prev_action_self), {}

    def _nose_vector(self, fdm):
        psi = fdm["attitude/psi-rad"]
        theta = fdm["attitude/theta-rad"]
        n = math.cos(theta) * math.cos(psi)
        e = math.cos(theta) * math.sin(psi)
        u = math.sin(theta)
        return n, e, u

    def _position(self, fdm):
        lat = fdm["position/lat-gc-deg"]
        lon = fdm["position/long-gc-deg"]
        ft_per_deg_lon = FT_PER_DEG_LAT * math.cos(math.radians(LAT0_DEG))
        north_ft = (lat - LAT0_DEG) * FT_PER_DEG_LAT
        east_ft = (lon - LON0_DEG) * ft_per_deg_lon
        alt_ft = fdm["position/h-agl-ft"]
        return north_ft, east_ft, alt_ft

    def _relative_geom(self, fdm_a, fdm_b):
        na, ea, ua = self._position(fdm_a)
        nb, eb, ub = self._position(fdm_b)
        dn, de, dz = nb - na, eb - ea, ub - ua
        rng_ft = max(math.sqrt(dn * dn + de * de + dz * dz), 1e-3)
        los = (dn / rng_ft, de / rng_ft, dz / rng_ft)
        nose = self._nose_vector(fdm_a)
        align_cos = nose[0] * los[0] + nose[1] * los[1] + nose[2] * los[2]
        return rng_ft, align_cos, dn, de, dz

    def _get_obs_for(self, fdm_owner, fdm_other, prev_action_owner):
        roll = fdm_owner["attitude/phi-rad"]
        pitch = fdm_owner["attitude/theta-rad"]
        p = fdm_owner["velocities/p-rad_sec"] / 5.0
        q = fdm_owner["velocities/q-rad_sec"] / 5.0
        r = fdm_owner["velocities/r-rad_sec"] / 5.0
        hdot = fdm_owner["velocities/h-dot-fps"] / 10.0

        rng_ft, align_owner, dn, de, dz = self._relative_geom(fdm_owner, fdm_other)
        _, align_other, _, _, _ = self._relative_geom(fdm_other, fdm_owner)

        if fdm_owner is self.fdm_self:
            closing_fps = (self._prev_range_ft - rng_ft) / self.control_dt if self._prev_range_ft else 0.0
        else:
            closing_fps = 0.0

        range_n = rng_ft / 100.0
        closing_n = closing_fps / 20.0
        dz_n = dz / 50.0

        cone_half_cos = math.cos(math.radians(self.cfg.cone_half_angle_deg))
        other_in_owner_cone = 1.0 if (align_owner >= cone_half_cos and rng_ft <= self.cfg.cone_range_ft) else 0.0
        owner_in_other_cone = 1.0 if (align_other >= cone_half_cos and rng_ft <= self.cfg.cone_range_ft) else 0.0

        return np.array([
            roll, pitch, p, q, r, hdot,
            range_n, closing_n,
            align_owner, align_other,
            dz_n,
            other_in_owner_cone, owner_in_other_cone,
            *prev_action_owner,
        ], dtype=np.float32)

    def _apply_action(self, fdm, surface_state, action):
        roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd = action
        aileron_t = float(np.clip(roll_cmd * self.cfg.roll_authority, -1.0, 1.0))
        elevator_t = float(np.clip(-pitch_cmd * self.cfg.pitch_authority, -1.0, 1.0))
        rudder_t = float(np.clip(yaw_cmd * self.cfg.yaw_authority, -1.0, 1.0))
        throttle = float(np.clip(
            self.cfg.hover_throttle + throttle_cmd * self.cfg.throttle_range, 0.0, 1.0
        ))
        targets = np.array([aileron_t, elevator_t, rudder_t])
        alpha = self.physics_dt / (self.cfg.control_surface_tau_s + self.physics_dt)
        surface_state += alpha * (targets - surface_state)
        fdm["fcs/aileron-cmd-norm"] = float(surface_state[0])
        fdm["fcs/elevator-cmd-norm"] = float(surface_state[1])
        fdm["fcs/rudder-cmd-norm"] = float(surface_state[2])
        fdm["fcs/throttle-cmd-norm"] = throttle

    def _is_out_of_bounds(self, fdm):
        alt = fdm["position/h-agl-ft"]
        n_ft, e_ft, _ = self._position(fdm)
        horiz = math.hypot(n_ft, e_ft)
        yaw_rate = abs(fdm["velocities/r-rad_sec"])
        return (
            alt < self.cfg.crash_min_alt_ft
            or alt > self.cfg.crash_max_alt_ft
            or abs(fdm["attitude/phi-rad"]) > self.cfg.crash_max_tilt_rad
            or abs(fdm["attitude/theta-rad"]) > self.cfg.crash_max_tilt_rad
            or yaw_rate > self.cfg.crash_max_yawrate_rps
            or horiz > self.cfg.max_horizontal_range_ft
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(4)
        opp_action = self.opponent_controller.compute_action(self)
        opp_action = np.asarray(opp_action, dtype=np.float32).reshape(4)

        for _ in range(self.substeps):
            self._apply_action(self.fdm_self, self._surface_self, action)
            self._apply_action(self.fdm_opp, self._surface_opp, opp_action)
            self.fdm_self.run()
            self.fdm_opp.run()

        self.step_count += 1

        rng_ft, align_mine, dn, de, dz = self._relative_geom(self.fdm_self, self.fdm_opp)
        _, align_opp_to_me, _, _, _ = self._relative_geom(self.fdm_opp, self.fdm_self)
        closing_fps = (self._prev_range_ft - rng_ft) / self.control_dt
        self._prev_range_ft = rng_ft

        cone_half_cos = math.cos(math.radians(self.cfg.cone_half_angle_deg))
        opp_in_my_cone = align_mine >= cone_half_cos and rng_ft <= self.cfg.cone_range_ft
        me_in_opp_cone = align_opp_to_me >= cone_half_cos and rng_ft <= self.cfg.cone_range_ft

        align_reward = self.cfg.reward_align_weight * align_mine
        exposure_penalty = self.cfg.reward_exposure_weight * align_opp_to_me
        standoff_raw = ((rng_ft - self.cfg.standoff_target_ft) / self.cfg.standoff_target_ft) ** 2
        standoff_penalty = self.standoff_weight * min(standoff_raw, self.cfg.standoff_penalty_cap)

        tilt = abs(self.fdm_self["attitude/phi-rad"]) + abs(self.fdm_self["attitude/theta-rad"])
        spin = abs(self.fdm_self["velocities/p-rad_sec"]) + abs(self.fdm_self["velocities/q-rad_sec"])
        yaw_rate_pen = abs(self.fdm_self["velocities/r-rad_sec"])
        jerk = float(np.sum(np.abs(action - self.prev_action_self)))
        control_penalty = (
            self.cfg.reward_tilt_weight * tilt
            + self.cfg.reward_spin_weight * spin
            + self.cfg.reward_yawrate_weight * yaw_rate_pen
            + self.cfg.reward_jerk_weight * jerk
        )

        opp_tilt = abs(self.fdm_opp["attitude/phi-rad"]) + abs(self.fdm_opp["attitude/theta-rad"])
        opp_spin = abs(self.fdm_opp["velocities/p-rad_sec"]) + abs(self.fdm_opp["velocities/q-rad_sec"])
        opp_yaw_rate_pen = abs(self.fdm_opp["velocities/r-rad_sec"])
        opp_jerk = float(np.sum(np.abs(opp_action - self.prev_action_opp)))
        opp_control_penalty = (
            self.cfg.reward_tilt_weight * opp_tilt
            + self.cfg.reward_spin_weight * opp_spin
            + self.cfg.reward_yawrate_weight * opp_yaw_rate_pen
            + self.cfg.reward_jerk_weight * opp_jerk
        )
        opp_align_reward = self.cfg.reward_align_weight * align_opp_to_me
        opp_exposure_penalty = self.cfg.reward_exposure_weight * align_mine

        cone_net = 0.0
        if opp_in_my_cone:
            cone_net += self.cfg.reward_cone_hold
            self.my_score += 1
        if me_in_opp_cone:
            cone_net -= self.cfg.reward_cone_hold
            self.opp_score += 1

        reward = align_reward - exposure_penalty - standoff_penalty - control_penalty + cone_net

        self_oob = self._is_out_of_bounds(self.fdm_self)
        opp_oob = self._is_out_of_bounds(self.fdm_opp)
        collided = rng_ft < self.cfg.min_separation_ft

        crashed = False
        terminated = False
        reset_reason = None
        if collided:
            reward -= self.cfg.crash_penalty
            crashed = True
            terminated = True
            reset_reason = "collision"
        elif self_oob:
            reward -= self.cfg.crash_penalty
            crashed = True
            terminated = True
            reset_reason = "self_crash"
        elif opp_oob:
            reward += self.cfg.opponent_fault_bonus
            terminated = True
            reset_reason = "opponent_crash"

        self.prev_action_self = action.copy()
        self.prev_action_opp = opp_action.copy()

        truncated = bool(self.step_count >= self.max_steps)
        if truncated and reset_reason is None:
            reset_reason = "timeout"

        obs = self._get_obs_for(self.fdm_self, self.fdm_opp, self.prev_action_self)

        info = {
            "range_ft": rng_ft,
            "closing_fps": closing_fps,
            "align_mine": align_mine,
            "align_opp": align_opp_to_me,
            "opp_in_my_cone": bool(opp_in_my_cone),
            "me_in_opp_cone": bool(me_in_opp_cone),
            "my_score": self.my_score,
            "opp_score": self.opp_score,
            "crashed": crashed,
            "reset_reason": reset_reason,
            "align_reward": align_reward,
            "exposure_penalty": exposure_penalty,
            "standoff_penalty": standoff_penalty,
            "control_penalty": control_penalty,
            "cone_net": cone_net,
            "opp_align_reward": opp_align_reward,
            "opp_exposure_penalty": opp_exposure_penalty,
            "opp_standoff_penalty": standoff_penalty,
            "opp_control_penalty": opp_control_penalty,
            "self_hdot_fps": float(self.fdm_self["velocities/h-dot-fps"]),
            "opp_hdot_fps": float(self.fdm_opp["velocities/h-dot-fps"]),
            "self_pos": self._position(self.fdm_self),
            "opp_pos": self._position(self.fdm_opp),
            "self_attitude": (
                self.fdm_self["attitude/phi-rad"],
                self.fdm_self["attitude/theta-rad"],
                self.fdm_self["attitude/psi-rad"],
            ),
            "opp_attitude": (
                self.fdm_opp["attitude/phi-rad"],
                self.fdm_opp["attitude/theta-rad"],
                self.fdm_opp["attitude/psi-rad"],
            ),
        }

        return obs, float(reward), terminated, truncated, info
