"""Egitilmis PPO politikasini calistirip 3D animasyon icin telemetri toplar."""

import json
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

from drone_rl.config import load_config
from drone_rl.env_factory import make_eval_vec_env, make_flight_eval_vec_env
from drone_rl.evaluate import resolve_model_paths, ALGO_CLASSES


def export_telemetry_json(telem, path):
    serializable = {}
    for k, v in telem.items():
        if hasattr(v, "tolist"):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)
    print(f"Telemetri JSON kaydedildi: {path}")


def run_inference_episode(algo, run, config=None, task="hover", use_best=False, deterministic=True):
    """task='hover' (varsayilan, eski davranis) veya task='flight'."""
    cfg = load_config(config)
    from pathlib import Path
    run = Path(run)
    model_path, vecnorm_path = resolve_model_paths(run, use_best)

    if not vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize dosyasi bulunamadi: {vecnorm_path}")

    if task == "hover":
        venv = make_eval_vec_env(cfg.env)
    else:
        venv = make_flight_eval_vec_env(cfg.flight_env)

    venv = VecNormalize.load(str(vecnorm_path), venv)
    venv.training = False
    venv.norm_reward = False

    model = ALGO_CLASSES[algo].load(str(model_path), device="cpu")
    raw = venv.envs[0]
    control_dt = raw.control_dt

    records = []
    obs = venv.reset()
    t = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, done, infos = venv.step(action)
        telem = infos[0]
        rec = {
            "t": t,
            "x_m": telem["x_m"],
            "y_m": telem["y_m"],
            "alt_ft": telem["alt_ft"],
            "roll_rad": telem["roll_rad"],
            "pitch_rad": telem["pitch_rad"],
            "yaw_rad": telem["yaw_rad"],
            "alt_err_ft": telem["alt_err_ft"],
            "crashed": telem["crashed"],
        }
        if "target_heading_rad" in telem:
            rec["target_heading_rad"] = telem["target_heading_rad"]
            rec["along_track_fps"] = telem["along_track_fps"]
            rec["cross_track_fps"] = telem["cross_track_fps"]
        records.append(rec)
        t += control_dt
        if done[0]:
            break

    out = {k: np.array([r[k] for r in records]) for k in records[0].keys()}
    out["control_dt"] = control_dt
    out["target_altitude_ft"] = raw.target_altitude
    return out

