"""Egitilmis PPO politikasini calistir; CSV telemetri ve/veya
gercek ACMI (Tacview) dosyasi olarak kaydet.

--task hover  -> F450HoverEnv
--task flight -> F450FlightEnv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from drone_rl.config import load_config
from drone_rl.env_factory import make_eval_vec_env, make_flight_eval_vec_env
from drone_rl.utils.units import ft_to_m
from drone_rl.acmi_writer import ACMIWriter


ALGO_CLASSES = {"ppo": PPO}


def resolve_model_paths(run: Path, use_best: bool):
    if use_best:
        return run / "best_model" / "best_model", run / "best_model" / "vecnormalize_best.pkl"
    return run / "model_final", run / "vecnormalize.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", type=str, choices=["ppo"], default="ppo")
    ap.add_argument("--task", type=str, choices=["hover", "flight"], default="hover")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--run", type=str, default="/content/repo/runs/run")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--output", type=str, default="/content/telemetry.csv")
    ap.add_argument("--acmi-output", type=str, default=None)
    ap.add_argument("--use-best", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run = Path(args.run)
    model_path, vecnorm_path = resolve_model_paths(run, args.use_best)

    if not vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize dosyasi bulunamadi: {vecnorm_path}")

    if args.task == "hover":
        venv = make_eval_vec_env(cfg.env)
    else:
        venv = make_flight_eval_vec_env(cfg.flight_env)

    venv = VecNormalize.load(str(vecnorm_path), venv)
    venv.training = False
    venv.norm_reward = False

    algo_cls = ALGO_CLASSES[args.algo]
    model = algo_cls.load(str(model_path), device="cpu")
    raw = venv.envs[0]
    control_dt = raw.control_dt

    all_telemetry = []
    acmi = ACMIWriter(name="F450", obj_type="Air+Rotorcraft+UAV", color="Blue") \
        if args.acmi_output else None

    global_t = 0.0

    for ep in range(args.episodes):
        obs = venv.reset()
        t = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = venv.step(action)
            telem = infos[0]

            data = {
                "timestamp": round(t, 4),
                "episode": ep,
                "alt_ft": round(telem["alt_ft"], 4),
                "roll_rad": round(telem["roll_rad"], 6),
                "pitch_rad": round(telem["pitch_rad"], 6),
                "yaw_rad": round(telem["yaw_rad"], 6),
                "alt_err_ft": round(telem["alt_err_ft"], 4),
                "crashed": telem["crashed"],
            }
            if "target_heading_rad" in telem:
                data["target_heading_rad"] = round(telem["target_heading_rad"], 4)
                data["along_track_fps"] = round(telem["along_track_fps"], 4)
                data["cross_track_fps"] = round(telem["cross_track_fps"], 4)
            all_telemetry.append(data)

            if acmi is not None:
                alt_m = ft_to_m(telem["alt_sl_ft"])
                acmi.add_frame(
                    t=global_t,
                    lon_deg=telem["lon_deg"],
                    lat_deg=telem["lat_deg"],
                    alt_m=alt_m,
                    roll_deg=np.degrees(telem["roll_rad"]),
                    pitch_deg=np.degrees(telem["pitch_rad"]),
                    yaw_deg=np.degrees(telem["yaw_rad"]),
                )

            t += control_dt
            global_t += control_dt
            if done[0]:
                break

    df = pd.DataFrame(all_telemetry)
    df.to_csv(args.output, index=False, header=True)
    print(f"CSV telemetri kaydedildi: {args.output}")
    print(df.head())

    if acmi is not None:
        saved_path = acmi.save(args.acmi_output)
        print(f"ACMI (Tacview) dosyasi kaydedildi: {saved_path}")


if __name__ == '__main__':
    main()
