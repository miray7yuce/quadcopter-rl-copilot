"""Egitilmis politikayi calistir ve ucus izini olc."""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drone_rl.envs.f450_env import F450HoverEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="/content/runs/hover_v1")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    run = Path(args.run)

    venv = DummyVecEnv([lambda: F450HoverEnv()])
    venv = VecNormalize.load(str(run / "vecnormalize.pkl"), venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(str(run / "model"), device="cpu")

    raw = venv.envs[0]
    hedef = raw.target_altitude
    iz = []

    for ep in range(args.episodes):
        obs = venv.reset()
        h_kayit, tilt_kayit = [], []
        adim = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = venv.step(action)
            h = raw.fdm["position/h-agl-ft"]
            tilt = abs(raw.fdm["attitude/phi-rad"]) + abs(raw.fdm["attitude/theta-rad"])
            h_kayit.append(h)
            tilt_kayit.append(tilt)
            iz.append((ep, adim, h, tilt))
            adim += 1
            if done[0]:
                break

        son = h_kayit[len(h_kayit) // 2:]
        print(
            f"ep {ep}: adim={adim:3d}  "
            f"son yari irtifa ort={np.mean(son):6.2f} ft (hedef {hedef})  "
            f"sapma={np.mean(np.abs(np.array(son) - hedef)):5.2f} ft  "
            f"ort tilt={np.mean(tilt_kayit):.3f} rad"
        )

    if args.csv:
        np.savetxt(args.csv, np.array(iz), delimiter=",",
                   header="episode,adim,irtifa_ft,tilt_rad", comments="")
        print("kaydedildi:", args.csv)


if __name__ == "__main__":
    main()
