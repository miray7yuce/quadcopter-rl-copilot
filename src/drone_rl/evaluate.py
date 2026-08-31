"""Egitilmis politikayi calistir ve ACME telemetry formatinda kaydet."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_rl.envs.f450_env import F450HoverEnv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="/content/repo/runs/hover_v1")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--output", type=str, default="/content/acme_telemetry.csv")
    args = ap.parse_args()

    run = Path(args.run)
    venv = DummyVecEnv([lambda: F450HoverEnv()])
    venv = VecNormalize.load(str(run / "vecnormalize.pkl"), venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(str(run / "model"), device="cpu")
    raw = venv.envs[0]
    all_telemetry = []

    for ep in range(args.episodes):
        obs = venv.reset()
        t = 0.0
        dt = 1.0 / 20.0  # control_hz
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = venv.step(action)
            
            # ACME Telemetry Data Map with Headers
            data = {
                "timestamp": round(t, 4),
                "episode": ep,
                "pos_x_ft": round(float(raw.fdm["position/h-agl-ft"]), 4),
                "pos_y_ft": 0.0,
                "pos_z_ft": round(float(raw.fdm["position/h-agl-ft"]), 4),
                "roll_rad": round(float(raw.fdm["attitude/phi-rad"]), 6),
                "pitch_rad": round(float(raw.fdm["attitude/theta-rad"]), 6),
                "yaw_rad": round(float(raw.fdm["attitude/psi-true-rad"]), 6),
                "alt_err": round(abs(raw.fdm["position/h-agl-ft"] - raw.target_altitude), 4)
            }
            all_telemetry.append(data)
            t += dt
            if done[0]:
                break #episode length

    df = pd.DataFrame(all_telemetry)
    # Save with headers
    df.to_csv(args.output, index=False, header=True)
    print(f"ACME Telemetry kaydedildi (Headerlar eklendi): {args.output}")
    print(df.head())

if __name__ == '__main__':
    main()
