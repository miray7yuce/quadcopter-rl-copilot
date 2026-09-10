import asyncio
import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from drone_rl.dogfight.config import load_dogfight_config
from drone_rl.dogfight.dogfight_env import DogfightEnv, NormalizerStats
from drone_rl.dogfight.env_factory import load_opponent_controller
from drone_rl.dogfight.checkpoint_pool import CheckpointPool

app = FastAPI()
STATE = {
    "html_path": None, "env": None, "training_controller": None,
    "demo_max_steps": None, "live_snapshot_dir": None, "pool_dir": None,
    "reload_interval_s": 15.0, "_last_training_mtime": None,
    "_last_pool_version": None, "_last_event": None,
}


@app.get("/")
def index():
    return FileResponse(STATE["html_path"])


def _check_and_reload_training():
    model_path = os.path.join(STATE["live_snapshot_dir"], "model.zip")
    vecnorm_path = os.path.join(STATE["live_snapshot_dir"], "vecnormalize.pkl")
    if not (os.path.exists(model_path) and os.path.exists(vecnorm_path)):
        return
    mtime = os.path.getmtime(model_path)
    if STATE["_last_training_mtime"] is not None and mtime <= STATE["_last_training_mtime"]:
        return

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    dummy = DummyVecEnv([lambda: Monitor(DogfightEnv(STATE["env"].cfg))])
    vecnorm = VecNormalize.load(vecnorm_path, dummy)
    stats = NormalizerStats(vecnorm)
    model = PPO.load(model_path, device="cpu")

    STATE["training_controller"] = _TrainingSelfController(model, stats)
    STATE["_last_training_mtime"] = mtime
    STATE["_last_event"] = {"type": "training_updated"}
    print("[reload] TRAINING modeli guncellendi")


def _check_and_reload_best():
    pool = CheckpointPool(STATE["pool_dir"])
    if len(pool) == 0:
        return
    version = pool.entries[-1]["version"]
    if STATE["_last_pool_version"] is not None and version <= STATE["_last_pool_version"]:
        return

    new_controller = load_opponent_controller(*pool.latest())
    STATE["env"].set_opponent_controller(new_controller)
    STATE["_last_pool_version"] = version
    STATE["_last_event"] = {"type": "best_updated", "version": version}
    print(f"[reload] BEST modeli guncellendi -> v{version}")


async def reload_watcher():
    while True:
        await asyncio.sleep(STATE["reload_interval_s"])
        try:
            if STATE["env"] is not None:
                _check_and_reload_training()
                _check_and_reload_best()
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
                "self_hdot_fps": info["self_hdot_fps"], "opp_hdot_fps": info["opp_hdot_fps"],
                "range_ft": info["range_ft"], "closing_fps": info["closing_fps"],
                "opp_in_my_cone": info["opp_in_my_cone"], "me_in_opp_cone": info["me_in_opp_cone"],
                "align_reward": info["align_reward"], "exposure_penalty": info["exposure_penalty"],
                "standoff_penalty": info["standoff_penalty"], "control_penalty": info["control_penalty"],
                "opp_align_reward": info["opp_align_reward"], "opp_exposure_penalty": info["opp_exposure_penalty"],
                "opp_standoff_penalty": info["opp_standoff_penalty"], "opp_control_penalty": info["opp_control_penalty"],
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
                  demo_max_steps: int = None, reload_interval_s: float = 15.0, port: int = 8020):
    import threading
    import uvicorn
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    cfg = load_dogfight_config(config_path)

    model_path = os.path.join(live_snapshot_dir, "model.zip")
    vecnorm_path = os.path.join(live_snapshot_dir, "vecnormalize.pkl")
    if not (os.path.exists(model_path) and os.path.exists(vecnorm_path)):
        raise FileNotFoundError(f"Henuz bir egitim anlik goruntusu yok: {live_snapshot_dir}")

    pool = CheckpointPool(pool_dir)
    if len(pool) == 0:
        raise FileNotFoundError(f"Havuz bos: {pool_dir}. Once --seed-pool calistirin.")

    opp_controller = load_opponent_controller(*pool.latest())
    env = DogfightEnv(cfg.env, opponent_controller=opp_controller)

    dummy = DummyVecEnv([lambda: Monitor(DogfightEnv(cfg.env))])
    vecnorm = VecNormalize.load(vecnorm_path, dummy)
    stats = NormalizerStats(vecnorm)
    model = PPO.load(model_path, device="cpu")

    STATE["env"] = env
    STATE["training_controller"] = _TrainingSelfController(model, stats)
    STATE["html_path"] = html_path
    STATE["demo_max_steps"] = demo_max_steps
    STATE["live_snapshot_dir"] = live_snapshot_dir
    STATE["pool_dir"] = pool_dir
    STATE["reload_interval_s"] = reload_interval_s
    STATE["_last_training_mtime"] = os.path.getmtime(model_path)
    STATE["_last_pool_version"] = pool.entries[-1]["version"]
    STATE["_last_event"] = None

    thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning"),
        daemon=True,
    )
    thread.start()
    print(f"Dogfight sunucusu baslatildi (port {port}). Her {reload_interval_s}s kontrol edilecek.")
