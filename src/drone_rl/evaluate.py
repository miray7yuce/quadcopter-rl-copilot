"""Egitilmis politikayi (PPO veya SAC) calistir; CSV telemetri ve/veya
gercek ACMI (Tacview) dosyasi olarak kaydet."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize

from drone_rl.config import load_config
from drone_rl.env_factory import make_eval_vec_env
from drone_rl.utils.units import ft_to_m
from drone_rl.acmi_writer import ACMIWriter


ALGO_CLASSES = {"ppo": PPO, "sac": SAC}


def resolve_model_paths(run: Path, use_best: bool):
    """--use-best bayragina gore yuklenecek model + VecNormalize dosya
    yollarini dondurur. train.py'nin ciktisiyla birebir eslesmesi gereken
    tek yer burasi - isimler degisirse sadece burasi guncellenir."""
    if use_best:
        return run / "best_model" / "best_model", run / "best_model" / "vecnormalize_best.pkl"
    return run / "model_final", run / "vecnormalize.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", type=str, choices=["ppo", "sac"], default="ppo",
                     help="Modelin egitildigi algoritma - egitimde --algo sac "
                          "kullandiysan burada da --algo sac vermelisin, aksi "
                          "halde model yuklenirken hata alirsin.")
    ap.add_argument("--config", type=str, default=None,
                     help="Egitimde kullanilan configs/ppo_hover.yaml. "
                          "Ortam parametrelerinin (hover_throttle, control_hz vb.) "
                          "egitimle AYNI olmasi icin, egitimde --config kullandiysan "
                          "burada da ayni dosyayi vermelisin.")
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

    cfg = load_config(args.config)
    run = Path(args.run)
    model_path, vecnorm_path = resolve_model_paths(run, args.use_best)

    if not vecnorm_path.exists():
        raise FileNotFoundError(
            f"VecNormalize dosyasi bulunamadi: {vecnorm_path}\n"
            "train.py'nin ciktisi ile bu betigin bekledigi dosya adlari "
            "arasinda bir uyumsuzluk olabilir; --run yolunu kontrol edin."
        )

    venv = make_eval_vec_env(cfg.env)
    venv = VecNormalize.load(str(vecnorm_path), venv)
    venv.training = False
    venv.norm_reward = False

    algo_cls = ALGO_CLASSES[args.algo]
    model = algo_cls.load(str(model_path), device="cpu")
    raw = venv.envs[0]
    control_dt = raw.control_dt  # f450_env.py'den; control_hz'i elle tekrarlamiyoruz

    all_telemetry = []
    acmi = ACMIWriter(name="F450", obj_type="Air+Rotorcraft+UAV", color="Blue") \
        if args.acmi_output else None

    global_t = 0.0  # ACMI icin episode'lar arasi kesintisiz artan zaman

    for ep in range(args.episodes):
        obs = venv.reset()
        t = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = venv.step(action)

            # infos[0], f450_env.step()'in VecEnv auto-reset'inden ONCE
            # topladigi ham telemetriyi tasir. raw.fdm'i DOGRUDAN OKUMUYORUZ:
            # done=True oldugunda DummyVecEnv, step() icinde ortami otomatik
            # resetler; bu noktada raw.fdm zaten YENI episode'un baslangic
            # durumunu gosterir, biten episode'un gercek son durumunu degil.
            # infos[0] bu sorunu yasamaz, cunku reset'ten once toplanmistir.
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
