"""F450 flight - gercek zamanli (WASDEF ile oynanabilir) backend.

DUZELTME (v6):
- SUCCESS_LINGER_SECONDS: 3.0 -> 5.0 (basari sonrasi irtifa-koruma
  gozlemi icin istenen sure).
- payload'a "max_horizontal_range_ft" eklendi - frontend artik grid/
  kamera boyutunu bu SABIT, fiziksel sinirdan turetiyor (kayan/sonsuz
  grid hack'i yerine).

--- (v5 notlari, hala gecerli) ---
Manuel kontrol ACI-MODU (angle-mode/self-leveling): tusa basildigi an
kontrol TAMAMEN VE ANINDA ele geciriliyor, PPO'nun biraktigi durumdan
BAGIMSIZ. Yaw'da tus yok ama kalinti spin'i sonduren damper her zaman
aktif.
"""

import asyncio
import json
import math
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from drone_rl.config import load_config
from drone_rl.env_factory import make_flight_env, make_flight_eval_vec_env
from drone_rl.evaluate import resolve_model_paths


MAX_ROLL_RAD = math.radians(20.0)
MAX_PITCH_RAD = math.radians(20.0)
ANGLE_KP = 3.5
RATE_KD = 0.30
YAW_DAMP_KP = 0.9

ALT_HOLD_KP = 0.08
ALT_HOLD_MAX = 0.5
ALT_MAG = 0.65


class ManualController:
    """Aci-modu (angle-mode/self-leveling) manuel kontrol."""

    def __init__(self):
        self.alt_lock = None

    def compute_action(self, keys: dict, env) -> np.ndarray:
        f = env.fdm
        roll_now = f["attitude/phi-rad"]
        pitch_now = f["attitude/theta-rad"]
        p_now = f["velocities/p-rad_sec"]
        q_now = f["velocities/q-rad_sec"]
        r_now = f["velocities/r-rad_sec"]
        alt_now = f["position/h-agl-ft"]

        roll_stick = 1.0 if keys.get("d") else (-1.0 if keys.get("a") else 0.0)
        pitch_stick = 1.0 if keys.get("w") else (-1.0 if keys.get("s") else 0.0)

        target_roll = MAX_ROLL_RAD * roll_stick
        roll_cmd = float(np.clip(
            ANGLE_KP * (target_roll - roll_now) - RATE_KD * p_now, -1.0, 1.0
        ))

        target_pitch = -MAX_PITCH_RAD * pitch_stick
        pitch_cmd = float(np.clip(
            ANGLE_KP * (pitch_now - target_pitch) + RATE_KD * q_now, -1.0, 1.0
        ))

        yaw_cmd = float(np.clip(-YAW_DAMP_KP * r_now, -1.0, 1.0))

        if keys.get("e") or keys.get("f"):
            self.alt_lock = None
            throttle_cmd = ALT_MAG if keys.get("e") else -ALT_MAG
        else:
            if self.alt_lock is None:
                self.alt_lock = alt_now
            throttle_cmd = float(np.clip(
                ALT_HOLD_KP * (self.alt_lock - alt_now), -ALT_HOLD_MAX, ALT_HOLD_MAX
            ))

        return np.array([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=np.float32)

    def reset_lock(self):
        self.alt_lock = None


def any_key_pressed(keys: dict) -> bool:
    return any(keys.get(k) for k in ("w", "a", "s", "d", "e", "f"))


class NormalizerStats:
    def __init__(self, vecnorm: VecNormalize):
        self.mean = vecnorm.obs_rms.mean.astype(np.float32)
        self.var = vecnorm.obs_rms.var.astype(np.float32)
        self.epsilon = vecnorm.epsilon
        self.clip_obs = vecnorm.clip_obs

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        normed = (obs - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normed, -self.clip_obs, self.clip_obs).astype(np.float32)


def load_policy(run: str, config: str, use_best: bool = False):
    cfg = load_config(config)
    run_path = Path(run)
    model_path, vecnorm_path = resolve_model_paths(run_path, use_best)

    if not vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize dosyasi bulunamadi: {vecnorm_path}")

    dummy_venv = make_flight_eval_vec_env(cfg.flight_env)
    vecnorm = VecNormalize.load(str(vecnorm_path), dummy_venv)
    stats = NormalizerStats(vecnorm)

    model = PPO.load(str(model_path), device="cpu")
    return model, stats, cfg


app = FastAPI()

STATE = {"model": None, "stats": None, "cfg": None, "html_path": None}


@app.get("/")
def index():
    return FileResponse(STATE["html_path"])


# DUZELTME (v6): 3.0 -> 5.0 - basari sonrasi irtifa-koruma gozlemi.
SUCCESS_LINGER_SECONDS = 5.0


@app.websocket("/ws")
async def flight_loop(websocket: WebSocket):
    await websocket.accept()

    model = STATE["model"]
    stats = STATE["stats"]
    cfg = STATE["cfg"]

    env = make_flight_env(cfg.flight_env)
    obs, _ = env.reset()

    linger_steps_total = max(1, int(SUCCESS_LINGER_SECONDS / env.control_dt))
    linger_remaining = None
    manual_ctrl = ManualController()

    current_keys = {}

    async def receiver():
        nonlocal current_keys
        try:
            while True:
                raw = await websocket.receive_text()
                current_keys = json.loads(raw) if raw else {}
        except WebSocketDisconnect:
            pass

    receiver_task = asyncio.create_task(receiver())

    try:
        while True:
            keys = current_keys

            if any_key_pressed(keys):
                action = manual_ctrl.compute_action(keys, env)
                mode = "manual"
            else:
                manual_ctrl.reset_lock()
                norm_obs = stats.normalize(obs).reshape(1, -1)
                action, _ = model.predict(norm_obs, deterministic=True)
                action = action[0]
                mode = "auto"

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reset = False

            if linger_remaining is not None:
                linger_remaining -= 1
                if info.get("crashed") or linger_remaining <= 0:
                    obs, _ = env.reset()
                    episode_reset = True
                    linger_remaining = None
            elif terminated or truncated:
                if info.get("reached_target") and not info.get("crashed"):
                    linger_remaining = linger_steps_total
                else:
                    obs, _ = env.reset()
                    episode_reset = True

            payload = dict(info)
            payload["mode"] = mode
            payload["episode_reset"] = episode_reset
            payload["target_altitude_ft"] = env.target_altitude
            payload["control_dt"] = env.control_dt
            # YENI (v6): frontend, grid/kamerayi buradan turetiyor.
            payload["max_horizontal_range_ft"] = env.max_horizontal_range_ft

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(env.control_dt)

    except WebSocketDisconnect:
        pass
    finally:
        receiver_task.cancel()


def start_server(run: str, config: str, html_path: str, use_best: bool = False, port: int = 8000):
    import threading
    import uvicorn

    model, stats, cfg = load_policy(run, config, use_best)
    STATE["model"] = model
    STATE["stats"] = stats
    STATE["cfg"] = cfg
    STATE["html_path"] = html_path

    thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning"),
        daemon=True,
    )
    thread.start()
    print(f"Sunucu baslatildi (arka planda, port {port}).")
    print("Simdi asagidaki hucreyi calistirip acilan pencereye/linke tikla.")
