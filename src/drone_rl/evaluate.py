"""Egitilmis politikayi calistir; CSV telemetri ve/veya gercek ACMI
(Tacview) dosyasi olarak kaydet."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drone_rl.envs.f450_env import F450HoverEnv
from drone_rl.utils.units import ft_to_m
from drone_rl.acmi_writer import ACMIWriter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="/content/repo/runs/hover_v1")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--output", type=str, default="/content/telemetry.csv",
                     help="CSV telemetri ciktisi")
    ap.add_argument("--acmi-output", type=str, default=None,
                     help="Verilirse, gercek Tacview .acmi dosyasi da bu yola yazilir")
    ap.add_argument(
        "--use-best", action="store_true",
        help="model_final yerine best_model + vecnormalize_best kullan"
    )
    args = ap.parse_args()

    run = Path(args.run)

    if args.use_best:
        model_path = run / "best_model" / "best_model"
        vecnorm_path = run / "best_model" / "vecnormalize_best.pkl"
    else:
        model_path = run / "model_final"
        vecnorm_path = run / "vecnormalize.pkl"

    if not vecnorm_path.exists():
        raise FileNotFoundError(
            f"VecNormalize dosyasi bulunamadi: {vecnorm_path}\n"
            "train.py'nin ciktisi ile bu betigin bekledigi dosya adlari "
            "arasinda bir uyumsuzluk olabilir; --run yolunu kontrol edin."
        )

    venv = DummyVecEnv([lambda: F450HoverEnv()])
    venv = VecNormalize.load(str(vecnorm_path), venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(str(model_path), device="cpu")
    raw = venv.envs[0]
    all_telemetry = []

    acmi = ACMIWriter(name="F450", obj_type="Air+Rotorcraft+UAV", color="Blue") \
        if args.acmi_output else None

    global_t = 0.0  # ACMI icin episode'lar arasi kesintisiz artan zaman

    for ep in range(args.episodes):
        obs = venv.reset()
        t = 0.0
        dt = 1.0 / 20.0  # control_hz

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = venv.step(action)

            fdm = raw.fdm
            alt_ft = float(fdm["position/h-agl-ft"])
            roll_rad = float(fdm["attitude/phi-rad"])
            pitch_rad = float(fdm["attitude/theta-rad"])
            yaw_rad = float(fdm["attitude/psi-rad"])

            data = {
                "timestamp": round(t, 4),
                "episode": ep,
                "alt_ft": round(alt_ft, 4),
                "roll_rad": round(roll_rad, 6),
                "pitch_rad": round(pitch_rad, 6),
                "yaw_rad": round(yaw_rad, 6),
                "alt_err_ft": round(abs(alt_ft - raw.target_altitude), 4),
            }
            all_telemetry.append(data)

            if acmi is not None:
                lat_deg = float(fdm["position/lat-geod-deg"])
                lon_deg = float(fdm["position/long-gc-deg"])
                alt_m = ft_to_m(float(fdm["position/h-sl-ft"]))
                acmi.add_frame(
                    t=global_t,
                    lon_deg=lon_deg,
                    lat_deg=lat_deg,
                    alt_m=alt_m,
                    roll_deg=np.degrees(roll_rad),
                    pitch_deg=np.degrees(pitch_rad),
                    yaw_deg=np.degrees(yaw_rad),
                )

            t += dt
            global_t += dt
            if done[0]:
                break  # episode length

    df = pd.DataFrame(all_telemetry)
    df.to_csv(args.output, index=False, header=True)
    print(f"CSV telemetri kaydedildi: {args.output}")
    print(df.head())

    if acmi is not None:
        saved_path = acmi.save(args.acmi_output)
        print(f"ACMI (Tacview) dosyasi kaydedildi: {saved_path}")
        print(
            "Not: episode'lar arasi zaman kesintisiz devam ediyor "
            "(reset aninda arac konumda 'atlar' - bu normaldir, "
            "cunku her episode farkli baslangic irtifasindan basliyor)."
        )


if __name__ == '__main__':
    main()
