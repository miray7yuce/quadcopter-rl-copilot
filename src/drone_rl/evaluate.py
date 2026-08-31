"""Egitilmis politikayi calistir; CSV telemetry + gercek ACMI (Tacview 2.2) kaydet."""

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_rl.envs.f450_env import F450HoverEnv

def safe_fdm(fdm, prop, default=0.0):
    """JSBSim property okurken hata verirse default dondur."""
    try:
        return float(fdm[prop])
    except (KeyError, TypeError):
        return default

def rad2deg(x):
    return x * 180.0 / math.pi

class AcmiWriter:
    """Tacview ACMI 2.2 dosyasi yazan minimal writer."""
    def __init__(self, path, reference_time_iso="2024-01-01T00:00:00Z"):
        self.path = Path(path)
        self.f = open(self.path, "w", encoding="utf-8")
        self.f.write("FileType=text/acmi/tacview\n")
        self.f.write("FileVersion=2.2\n")
        self.f.write(f"0,ReferenceTime={reference_time_iso}\n")
        self._last_time = None
        self._known_ids = set()

    def new_time_frame(self, t_seconds):
        if self._last_time != t_seconds:
            self.f.write(f"#{t_seconds:.4f}\n")
            self._last_time = t_seconds

    def update_object(self, obj_id, lon_deg, lat_deg, alt_m, roll_deg, pitch_deg, yaw_deg,
                      name=None, obj_type=None):
        t_field = f"T={lon_deg:.7f}|{lat_deg:.7f}|{alt_m:.2f}|{roll_deg:.3f}|{pitch_deg:.3f}|{yaw_deg:.3f}"
        parts = [str(obj_id), t_field]
        if obj_id not in self._known_ids:
            if name: parts.append(f"Name={name}")
            if obj_type: parts.append(f"Type={obj_type}")
            self._known_ids.add(obj_id)
        self.f.write(",".join(parts) + "\n")

    def remove_object(self, obj_id):
        self.f.write(f"-{obj_id}\n")
        self._known_ids.discard(obj_id)

    def close(self):
        self.f.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="/content/repo/runs/hover_v1")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--output", type=str, default="/content/iz.csv")
    ap.add_argument("--acmi_output", type=str, default="/content/telemetry.acmi")
    args = ap.parse_args()

    run = Path(args.run)
    venv = DummyVecEnv([lambda: F450HoverEnv()])
    venv = VecNormalize.load(str(run / "vecnormalize.pkl"), venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(str(run / "model"), device="cpu")
    raw = venv.envs[0]
    all_telemetry = []
    acmi = AcmiWriter(args.acmi_output)

    for ep in range(args.episodes):
        obs = venv.reset()
        t = 0.0
        dt = 1.0 / 20.0
        obj_id = f"{1000 + ep:X}"
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = venv.step(action)

            # Corrected property: attitude/heading-true-rad
            yaw_val = safe_fdm(raw.fdm, "attitude/heading-true-rad")

            data = {
                "timestamp": round(t, 4),
                "episode": ep,
                "pos_x_ft": round(float(raw.fdm["position/h-agl-ft"]), 4),
                "pos_y_ft": 0.0,
                "pos_z_ft": round(float(raw.fdm["position/h-agl-ft"]), 4),
                "roll_rad": round(float(raw.fdm["attitude/phi-rad"]), 6),
                "pitch_rad": round(float(raw.fdm["attitude/theta-rad"]), 6),
                "yaw_rad": round(yaw_val, 6),
                "alt_err": round(abs(raw.fdm["position/h-agl-ft"] - raw.target_altitude), 4)
            }
            all_telemetry.append(data)

            lon_deg = safe_fdm(raw.fdm, "position/long-gc-deg")
            lat_deg = safe_fdm(raw.fdm, "position/lat-gc-deg")
            alt_ft = safe_fdm(raw.fdm, "position/h-sl-ft", default=safe_fdm(raw.fdm, "position/h-agl-ft"))
            alt_m = alt_ft * 0.3048
            roll_deg = rad2deg(safe_fdm(raw.fdm, "attitude/phi-rad"))
            pitch_deg = rad2deg(safe_fdm(raw.fdm, "attitude/theta-rad"))
            yaw_deg = rad2deg(yaw_val)

            acmi.new_time_frame(t)
            acmi.update_object(obj_id, lon_deg, lat_deg, alt_m, roll_deg, pitch_deg, yaw_deg, 
                               name=f"F450-ep{ep}", obj_type="Air+Rotorcraft")
            
            t += dt
            if done[0]:
                acmi.remove_object(obj_id)
                break

    acmi.close()
    df = pd.DataFrame(all_telemetry)
    df.to_csv(args.output, index=False, header=True)
    print(f"CSV telemetry kaydedildi: {args.output}")
    print(f"ACMI (Tacview) kaydedildi: {args.acmi_output}")

if __name__ == '__main__':
    main()
