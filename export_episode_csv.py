"""Iki drone'un (TRAINING = egitim modeli, BEST = havuzdaki en guncel
rakip) uçus telemetrisini bir veya daha fazla bolum boyunca ADIM-ADIM
CSV'ye kaydeder.

Egitim (Stage B) ARKA PLANDA calisirken de guvenle calistirilabilir:
bu script SADECE diskteki mevcut snapshot/havuz dosyalarini OKUR ve
KENDI AYRI bir degerlendirme ortaminda calistirir - egitim surecine
hicbir sekilde mudahale etmez.

Kullanim (repo kokunden):
    python tools/export_episode_csv.py --episodes 3

Ciktilar (exports/ klasorunde):
    training_drone.csv   - sadece TRAINING dronunun telemetrisi
    best_drone.csv        - sadece BEST dronunun telemetrisi
    both_drones_long.csv  - ikisi bir arada, 'drone' sutunuyla ayrilmis
                             (excel'de pivot/filtre yapmak icin pratik)
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from drone_rl.dogfight.config import load_dogfight_config
from drone_rl.dogfight.dogfight_env import DogfightEnv, NormalizerStats
from drone_rl.dogfight.env_factory import load_opponent_controller, make_dummy_vecnorm_env
from drone_rl.dogfight.checkpoint_pool import CheckpointPool


def _find_repo_root() -> Path:
    """DUZELTME: eskiden REPO_ROOT = Path(__file__).resolve().parent.parent
    idi - script tools/ altinda DEGIL de baska bir yere (orn. dogrudan
    /content/repo koku) yazilirsa bu hesap bir seviye kayip YANLIS
    dizine (orn. /content/configs/...) isaret ediyordu.

    Simdi once GUNCEL CALISMA DIZININ (cwd) kendisi, sonra __file__'in
    bulundugu ve ust dizinleri denenir; 'configs' VE 'runs' klasorlerini
    BIRLIKTE iceren ilk aday repo koku olarak kabul edilir. Boylece
    script'in nereye yazildigindan bagimsiz, dogru sekilde calisir."""
    candidates = [Path.cwd()]
    here = Path(__file__).resolve()
    candidates += [here.parent, here.parent.parent, here.parent.parent.parent]

    for c in candidates:
        if (c / "configs").is_dir() and (c / "src" / "drone_rl").is_dir():
            return c

    tried = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "Repo koku otomatik bulunamadi (icinde hem 'configs/' hem "
        "'src/drone_rl/' olan bir dizin aranıyor). Denenen yerler:\n"
        f"{tried}\n"
        "Bu betigi 'cd /content/repo && python tools/export_episode_csv.py' "
        "seklinde, repo kokunden calistirdiginizdan emin olun; ya da "
        "--config, --live-snapshot-dir, --pool argumanlarini elle verin."
    )


REPO_ROOT = _find_repo_root()
RUNS_DIR = REPO_ROOT / "runs"
EXPORT_DIR = REPO_ROOT / "exports"

FIELDS = [
    "episode", "step", "time_s", "drone",
    "pos_n_ft", "pos_e_ft", "pos_alt_ft",
    "roll_deg", "pitch_deg", "yaw_deg",
    "speed_fps", "hdot_fps",
    "range_ft", "closing_fps",
    "ata_deg", "aa_deg",
    "in_cone", "hp",
    "action_roll", "action_pitch", "action_yaw", "action_throttle",
    "step_reward",
    "reset_reason",
]


def _load_training_controller(model_path, vecnorm_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    vecnorm = VecNormalize.load(str(vecnorm_path), make_dummy_vecnorm_env())
    stats = NormalizerStats(vecnorm)
    model = PPO.load(str(model_path), device="cpu")

    class _Ctrl:
        def compute_action(self, env):
            obs = env._get_obs_for(env.fdm_self, env.fdm_opp, env.prev_action_self)
            norm_obs = stats.normalize(obs).reshape(1, -1)
            action, _ = model.predict(norm_obs, deterministic=True)
            return action[0]

    return _Ctrl()


def run_episodes(env: DogfightEnv, training_ctrl, n_episodes: int, max_steps: int):
    rows = []
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        step = 0
        while True:
            action = training_ctrl.compute_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            t = step * env.control_dt

            self_roll, self_pitch, self_yaw = info["self_attitude"]
            opp_roll, opp_pitch, opp_yaw = info["opp_attitude"]
            self_n, self_e, self_alt = info["self_pos"]
            opp_n, opp_e, opp_alt = info["opp_pos"]

            rows.append({
                "episode": ep, "step": step, "time_s": round(t, 3), "drone": "training",
                "pos_n_ft": self_n, "pos_e_ft": self_e, "pos_alt_ft": self_alt,
                "roll_deg": np.degrees(self_roll), "pitch_deg": np.degrees(self_pitch),
                "yaw_deg": np.degrees(self_yaw),
                "speed_fps": info["self_speed_fps"], "hdot_fps": info["self_hdot_fps"],
                "range_ft": info["range_ft"], "closing_fps": info["closing_fps"],
                "ata_deg": info["ata_deg"], "aa_deg": info["aa_deg"],
                "in_cone": info["opp_in_my_cone"], "hp": info["hp_self"],
                "action_roll": float(action[0]), "action_pitch": float(action[1]),
                "action_yaw": float(action[2]), "action_throttle": float(action[3]),
                "step_reward": info["self_reward"],
                "reset_reason": info["reset_reason"] or "",
            })
            rows.append({
                "episode": ep, "step": step, "time_s": round(t, 3), "drone": "best",
                "pos_n_ft": opp_n, "pos_e_ft": opp_e, "pos_alt_ft": opp_alt,
                "roll_deg": np.degrees(opp_roll), "pitch_deg": np.degrees(opp_pitch),
                "yaw_deg": np.degrees(opp_yaw),
                "speed_fps": info["opp_speed_fps"], "hdot_fps": info["opp_hdot_fps"],
                "range_ft": info["range_ft"], "closing_fps": info["closing_fps"],
                "ata_deg": info["opp_ata_deg"], "aa_deg": info["ata_deg"],
                "in_cone": info["me_in_opp_cone"], "hp": info["hp_opp"],
                "action_roll": "", "action_pitch": "", "action_yaw": "", "action_throttle": "",
                "step_reward": info["opp_reward"],
                "reset_reason": info["reset_reason"] or "",
            })

            if terminated or truncated or step >= max_steps:
                print(f"  bolum {ep}: {step} adim, bitis={info['reset_reason']}, "
                      f"HP training={info['hp_self']:.2f} best={info['hp_opp']:.2f}")
                break
    return rows


def write_csv(rows, path: Path, drone_filter: str = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            if drone_filter is None or r["drone"] == drone_filter:
                writer.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "dogfight_stage_b.yaml"))
    ap.add_argument("--live-snapshot-dir", type=str,
                    default=str(RUNS_DIR / "dogfight_stage_b" / "live_snapshot"))
    ap.add_argument("--pool", type=str, default=str(RUNS_DIR / "dogfight_pool"))
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    cfg = load_dogfight_config(args.config)

    print(f"repo koku      : {REPO_ROOT}")
    print(f"config         : {args.config}")
    print(f"live-snapshot  : {args.live_snapshot_dir}")
    print(f"pool           : {args.pool}")

    live_dir = Path(args.live_snapshot_dir)
    model_path = live_dir / "model.zip"
    vecnorm_path = live_dir / "vecnormalize.pkl"
    if not (model_path.exists() and vecnorm_path.exists()):
        raise FileNotFoundError(
            f"Egitim anlik goruntusu bulunamadi: {live_dir}\n"
            f"Stage B en az bir snapshot yazana kadar bekleyin "
            f"(--snapshot-freq varsayilani ile birkac dakika icinde olusur)."
        )

    pool = CheckpointPool(args.pool)
    if len(pool) == 0:
        raise FileNotFoundError(f"Havuz bos: {args.pool}. Once seed-pool calistirin.")

    opp_controller = load_opponent_controller(*pool.latest(), deterministic=True)
    env = DogfightEnv(cfg.env, opponent_controller=opp_controller)
    env.set_shaped_weight(cfg.env.shaped_weight_end)
    env.set_curriculum_progress(1.0)

    training_ctrl = _load_training_controller(model_path, vecnorm_path)

    max_steps = int(cfg.env.episode_seconds * cfg.env.control_hz)
    print(f"TRAINING snapshot: {model_path}")
    print(f"BEST (havuz)      : v{pool.latest_version()}")
    print(f"{args.episodes} bolum kaydediliyor (bolum basina en fazla {max_steps} adim)...")

    rows = run_episodes(env, training_ctrl, args.episodes, max_steps)

    write_csv(rows, EXPORT_DIR / "both_drones_long.csv")
    write_csv(rows, EXPORT_DIR / "training_drone.csv", drone_filter="training")
    write_csv(rows, EXPORT_DIR / "best_drone.csv", drone_filter="best")

    print("\nTamamlandi:")
    for name in ["both_drones_long.csv", "training_drone.csv", "best_drone.csv"]:
        print(" -", EXPORT_DIR / name)


if __name__ == "__main__":
    main()
