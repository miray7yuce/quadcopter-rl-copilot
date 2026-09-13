"""Gercek zamanli dogfight demo sunucusu - v8.

Degisiklikler:
  * Payload yeni odul/geometri anahtarlarina gore guncellendi
    (ATA/AA, HP, angajman ekseni, track/threat/dist/close kirilimi).
  * Koni yonelimi artik Euler acilarindan degil, ortamin hesapladigi
    ANGAJMAN EKSENI vektorunden (self_axis/opp_axis) cizilir - hem
    isaret/siralama hatalarini ortadan kaldirir hem de ekranda gorulen
    koni ile odulde kullanilan koni BIREBIR ayni olur.
  * VecNormalize yuklenirken artik gercek bir DogfightEnv (2 JSBSim
    motoru) kurulmuyor; _DummyObsEnv kullaniliyor.
"""

import asyncio
import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from drone_rl.dogfight.config import load_dogfight_config
from drone_rl.dogfight.dogfight_env import DogfightEnv, NormalizerStats, OBS_DIM
from drone_rl.dogfight.env_factory import load_opponent_controller, make_dummy_vecnorm_env
from drone_rl.dogfight.checkpoint_pool import CheckpointPool

app = FastAPI()
STATE = {
    "html_path": None, "env": None, "training_controller": None,
    "demo_max_steps": None, "live_snapshot_dir": None, "pool_dir": None,
    "reload_interval_s": 2.0, "_last_training_mtime": None,
    "_last_pool_version": None, "_last_event": None,
    "training_timesteps": None,
}


@app.get("/")
def index():
    return FileResponse(STATE["html_path"])


def _load_training_controller(model_path, vecnorm_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    vecnorm = VecNormalize.load(vecnorm_path, make_dummy_vecnorm_env())
    stats = NormalizerStats(vecnorm)
    if stats.obs_dim != OBS_DIM:
        raise ValueError(
            f"Snapshot {stats.obs_dim} boyutlu, bu surum {OBS_DIM} bekliyor "
            f"({vecnorm_path}). Eski run'lar v8 ile uyumlu degil."
        )
    model = PPO.load(model_path, device="cpu")
    return _TrainingSelfController(model, stats)


def _read_training_meta():
    """Hafif json okuma - model reload'undan bagimsiz, boylece timestep
    sayaci arayuzde akici ilerler."""
    meta_path = os.path.join(STATE["live_snapshot_dir"], "meta.json")
    if not os.path.exists(meta_path):
        return
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        STATE["training_timesteps"] = meta.get("num_timesteps")
    except Exception:
        pass


def _check_and_reload_training():
    """DIKKAT: agir, bloklayan islemler icerir. reload_watcher() bunu
    asyncio.to_thread ile ayri thread'de calistirir."""
    _read_training_meta()

    model_path = os.path.join(STATE["live_snapshot_dir"], "model.zip")
    vecnorm_path = os.path.join(STATE["live_snapshot_dir"], "vecnormalize.pkl")
    if not (os.path.exists(model_path) and os.path.exists(vecnorm_path)):
        return
    mtime = os.path.getmtime(model_path)
    if STATE["_last_training_mtime"] is not None and mtime <= STATE["_last_training_mtime"]:
        return

    STATE["training_controller"] = _load_training_controller(model_path, vecnorm_path)
    STATE["_last_training_mtime"] = mtime
    STATE["_last_event"] = {"type": "training_updated"}
    print("[reload] TRAINING modeli guncellendi")


def _check_and_reload_best():
    pool = CheckpointPool(STATE["pool_dir"])
    version = pool.latest_version()
    if version is None:
        return
    if STATE["_last_pool_version"] is not None and version <= STATE["_last_pool_version"]:
        return

    new_controller = load_opponent_controller(*pool.latest(), deterministic=True)
    STATE["env"].set_opponent_controller(new_controller)
    STATE["_last_pool_version"] = version
    STATE["_last_event"] = {"type": "best_updated", "version": version}
    print(f"[reload] BEST modeli guncellendi -> v{version}")


async def reload_watcher():
    while True:
        await asyncio.sleep(STATE["reload_interval_s"])
        try:
            if STATE["env"] is not None:
                await asyncio.to_thread(_check_and_reload_training)
                await asyncio.to_thread(_check_and_reload_best)
        except Exception as e:
            print(f"[reload] hata (atlaniyor): {e}")


@app.on_event("startup")
async def _on_startup():
    asyncio.create_task(reload_watcher())


@app.websocket("/ws")
async def dogfight_loop(websocket: WebSocket):
    await websocket.accept()
    env: DogfightEnv = STATE["env"]
    demo_max_steps = STATE["demo_max_steps"]

    obs, _ = env.reset()
    total_steps = 0
    finished = False
    cumulative_my_score = 0
    cumulative_opp_score = 0
    episode_count = 0
    last_reset_reason = None
    info = None

    try:
        while True:
            if not finished:
                training_action = STATE["training_controller"].compute_action_for_self(env)
                obs, reward, terminated, truncated, info = env.step(training_action)
                total_steps += 1

                if terminated or truncated:
                    cumulative_my_score += info["my_score"]
                    cumulative_opp_score += info["opp_score"]
                    episode_count += 1
                    last_reset_reason = info.get("reset_reason")
                    obs, _ = env.reset()

                if demo_max_steps and total_steps >= demo_max_steps:
                    finished = True

            payload = {
                "self_pos": info["self_pos"], "opp_pos": info["opp_pos"],
                "self_attitude": info["self_attitude"], "opp_attitude": info["opp_attitude"],
                "self_axis": list(info["self_axis"]), "opp_axis": list(info["opp_axis"]),
                "self_hdot_fps": info["self_hdot_fps"], "opp_hdot_fps": info["opp_hdot_fps"],
                "self_speed_fps": info["self_speed_fps"], "opp_speed_fps": info["opp_speed_fps"],
                "range_ft": info["range_ft"], "closing_fps": info["closing_fps"],
                "ata_deg": info["ata_deg"], "aa_deg": info["aa_deg"],
                "opp_ata_deg": info["opp_ata_deg"],
                "opp_in_my_cone": info["opp_in_my_cone"], "me_in_opp_cone": info["me_in_opp_cone"],
                "hp_self": info["hp_self"], "hp_opp": info["hp_opp"],
                "hp_initial": info["hp_initial"],
                # odul kirilimi
                "track_reward": info["track_reward"], "threat_penalty": info["threat_penalty"],
                "dist_penalty": info["dist_penalty"], "close_reward": info["close_reward"],
                "lock_reward": info["lock_reward"], "exposed_penalty": info["exposed_penalty"],
                "control_penalty": info["control_penalty"], "safety_penalty": info["safety_penalty"],
                "opp_track_reward": info["opp_track_reward"],
                "opp_threat_penalty": info["opp_threat_penalty"],
                "opp_dist_penalty": info["opp_dist_penalty"],
                "opp_close_reward": info["opp_close_reward"],
                "opp_control_penalty": info["opp_control_penalty"],
                "opp_safety_penalty": info["opp_safety_penalty"],
                "self_reward": info["self_reward"], "opp_reward": info["opp_reward"],
                "shaped_weight": info["shaped_weight"],
                "training_timesteps": STATE["training_timesteps"],
                "my_score": cumulative_my_score + info["my_score"],
                "opp_score": cumulative_opp_score + info["opp_score"],
                "episode_count": episode_count, "last_reset_reason": last_reset_reason,
                "cone_half_angle_deg": env.cfg.cone_half_angle_deg,
                "cone_range_ft": env.cfg.cone_range_ft,
                "step": total_steps, "finished": finished,
            }
            if STATE["_last_event"] is not None:
                payload["event"] = STATE["_last_event"]
                STATE["_last_event"] = None

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(env.control_dt if not finished else 1.0)

    except WebSocketDisconnect:
        pass


class _TrainingSelfController:
    def __init__(self, model, stats):
        self.model = model
        self.stats = stats

    def compute_action_for_self(self, env: DogfightEnv):
        obs = env._get_obs_for(env.fdm_self, env.fdm_opp, env.prev_action_self)
        norm_obs = self.stats.normalize(obs).reshape(1, -1)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        return action[0]


def start_server(live_snapshot_dir: str, pool_dir: str, config_path: str, html_path: str,
                 demo_max_steps: int = None, reload_interval_s: float = 2.0, port: int = 8020):
    import threading
    import uvicorn

    cfg = load_dogfight_config(config_path)

    model_path = os.path.join(live_snapshot_dir, "model.zip")
    vecnorm_path = os.path.join(live_snapshot_dir, "vecnormalize.pkl")
    if not (os.path.exists(model_path) and os.path.exists(vecnorm_path)):
        raise FileNotFoundError(f"Henuz bir egitim anlik goruntusu yok: {live_snapshot_dir}")

    pool = CheckpointPool(pool_dir)
    if len(pool) == 0:
        raise FileNotFoundError(f"Havuz bos: {pool_dir}. Once --seed-pool calistirin.")

    opp_controller = load_opponent_controller(*pool.latest(), deterministic=True)
    env = DogfightEnv(cfg.env, opponent_controller=opp_controller)
    env.set_shaped_weight(cfg.env.shaped_weight_end)
    env.set_curriculum_progress(1.0)

    STATE["env"] = env
    STATE["training_controller"] = _load_training_controller(model_path, vecnorm_path)
    STATE["html_path"] = html_path
    STATE["demo_max_steps"] = demo_max_steps
    STATE["live_snapshot_dir"] = live_snapshot_dir
    STATE["pool_dir"] = pool_dir
    STATE["reload_interval_s"] = reload_interval_s
    STATE["_last_training_mtime"] = os.path.getmtime(model_path)
    STATE["_last_pool_version"] = pool.latest_version()
    STATE["_last_event"] = None

    meta_path = os.path.join(live_snapshot_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                STATE["training_timesteps"] = json.load(f).get("num_timesteps")
        except Exception:
            STATE["training_timesteps"] = None

    thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning"),
        daemon=True,
    )
    thread.start()
    print(f"Dogfight sunucusu baslatildi (port {port}). Her {reload_interval_s}s kontrol edilecek.")
