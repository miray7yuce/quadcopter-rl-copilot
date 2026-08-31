"""F450 quadcopter icin hover gorevi ortami."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import jsbsim

HOVER_THROTTLE = 0.420
THROTTLE_RANGE = 0.25


class F450HoverEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, target_altitude_ft=30.0, episode_seconds=20.0,
                 physics_hz=240, control_hz=20):
        super().__init__()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32)

        self.target_altitude = target_altitude_ft
        self.physics_dt = 1.0 / physics_hz
        self.substeps = int(physics_hz / control_hz)
        self.max_steps = int(episode_seconds * control_hz)

        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_debug_level(0)
        if not self.fdm.load_model("F450"):
            raise RuntimeError("F450 modeli yuklenemedi")
        self.fdm.set_dt(self.physics_dt)

        self.step_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)

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
            self.fdm[f"fcs/throttle-cmd-norm[{i}]"] = HOVER_THROTTLE

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

    def step(self, action):
        #action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        throttles = np.clip(HOVER_THROTTLE + action * THROTTLE_RANGE, 0.0, 1.0)

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
            - 0.10 * alt_err_ft
            - 0.50 * tilt
            - 0.10 * spin
            - 0.05 * jerk
        )

        crashed = (
            self.fdm["position/h-agl-ft"] < 1.0
            or self.fdm["position/h-agl-ft"] > self.target_altitude + 60.0
            or abs(self.fdm["attitude/phi-rad"]) > 1.0
            or abs(self.fdm["attitude/theta-rad"]) > 1.0
        )
        if crashed:
            reward -= 50.0

        self.prev_action = action.copy()

        terminated = bool(crashed)
        truncated = bool(self.step_count >= self.max_steps)

        return obs, float(reward), terminated, truncated, {}
