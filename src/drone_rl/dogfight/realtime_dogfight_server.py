"""Dogfight canli demo sunucusu.

DUZELTME (v2):
- Skor artik SUNUCU TARAFINDA KUMULATIF tutuluyor (cumulative_my_score/
  cumulative_opp_score) - env.reset() her episode'da kendi ic sayacini
  sifirlasa da (RL egitimi icin DOGRU davranis), demo ekranindaki skor
  artik episode gecisleri arasinda KORUNUYOR.
- payload'a "reset_reason" ve "episode_count" eklendi - HANGI sebeple
  (collision/self_crash/opponent_crash/timeout) resetlendigini
  gosteriyor.
"""

import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from drone_rl.dogfight.config import load_dogfight_config
from drone_rl.dogfight.dogfight_env import DogfightEnv
from drone_rl.dogfight.env_factory import load_opponent_controller

app = FastAPI()
STATE = {"html_path": None, "env": None, "training_controller": None,
         "demo_max_steps": None}


@app.get("/")
def index():
    return FileResponse(STATE["html_path"])


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
                "self_pos": info["self_pos"],
                "opp_pos": info["opp_pos"],
                "self_attitude": info["self_attitude"],
                "opp_attitude": info["opp_attitude"],
                "range_ft": info["range_ft"],
                "opp_in_my_cone": info["opp_in_my_cone"],
                "me_in_opp_cone": info["me_in_opp_cone"],
                "my_score": cumulative_my_score + info["my_score"],
                "opp_score": cumulative_opp_score + info["opp_score"],
                "episode_count": episode_count,
                "last_reset_reason": last_reset_reason,
                "cone_half_angle_deg": env.cfg.cone_half_angle_deg,
                "cone_range_ft": env.cfg.cone_range_ft,
                "step": total_steps,
                "finished": finished,
            }
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


def start_server(training_model_path: str, training_vecnorm_path: str,
                  best_model_path: str, best_vecnorm_path: str,
                  config_path: str, html_path: str,
                  demo_max_steps: int = 6000, port: int = 8020):
    import threading
    import uvicorn
    from drone_rl.dogfight.dogfight_env import NormalizerStats
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    cfg = load_dogfight_config(config_path)

    opp_controller = load_opponent_controller(best_model_path, best_vecnorm_path)
    env = DogfightEnv(cfg.env, opponent_controller=opp_controller)

    dummy = DummyVecEnv([lambda: Monitor(DogfightEnv(cfg.env))])
    vecnorm_train = VecNormalize.load(training_vecnorm_path, dummy)
    stats_train = NormalizerStats(vecnorm_train)
    model_train = PPO.load(training_model_path, device="cpu")

    STATE["env"] = env
    STATE["training_controller"] = _TrainingSelfController(model_train, stats_train)
    STATE["html_path"] = html_path
    STATE["demo_max_steps"] = demo_max_steps

    thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning"),
        daemon=True,
    )
    thread.start()
    print(f"Dogfight sunucusu baslatildi (port {port}).")

