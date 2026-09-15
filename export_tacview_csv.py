"""Iki drone'un (TRAINING, BEST) ucusunu Tacview'un RESMI "Real-life CSV"
formatinda kaydeder - onceki export_acmi.py'den FARKLI: o Tacview'un
KENDI native ACMI formatini uretiyordu, bu ise Tacview'un DUZ CSV
IMPORT ozelligine uygun, dokumantasyona birebir uyan bir cikti uretir:

    https://raia-software-inc.gitbook.io/tacview/real-life-data/real-life-csv-data

O sayfadaki kurallar:
  * Sutunlar: Time, Longitude, Latitude, Altitude, Roll, Pitch, Yaw
    (Time=kayit basindan saniye; Longitude/Latitude=derece;
    Altitude=metre; Roll/Pitch/Yaw=derece)
  * BIR CSV DOSYASI = BIR UCAK. Iki drone icin IKI AYRI dosya uretilir;
    Tacview'da once birini File->Open ile acip sonra digerini
    File->Merge ile eklersiniz - boylece ikisi ayni sahnede,
    senkronize gorunur.
  * Ucak adi/pilot/renk METADATASI DOSYA ADINDAN okunur:
    "<NATO adi> (<pilot>) [<renk>].csv"

DUZELTME (surekli kayit): Eskiden TEK bolum kaydediliyordu ve bir
carpisma/crash olunca kayit o anda kesiliyordu - bolumler bazen
~10 saniyede bitince 'daha ne oldugunu anlamadan' kayit sona eriyordu.
Simdi varsayilan olarak EN AZ --min-duration-s (varsayilan 60s)
uzunlugunda KESINTISIZ bir kayit uretiliyor: bir bolum biterse ortam
otomatik resetlenip Time sutunu AYNI kesintisiz eksende artmaya
devam ediyor - hedef sureye ulasana kadar.

DIKKAT: Onceki export_episode_csv.py (analiz/rapor icin duz tablo CSV)
DOKUNULMADI, degistirilmedi - bu TAMAMEN AYRI, Tacview'a OZEL bir
script'tir.

Egitim (Stage B) ARKA PLANDA calisirken de guvenle calistirilabilir:
sadece diskteki mevcut snapshot/havuz dosyalarini OKUR, kendi ayri bir
degerlendirme ortaminda calisir.

Kullanim (repo kokunden):
    cd /content/repo && python tools/export_tacview_csv.py --min-duration-s 60

Cikti (exports/ klasorunde), her kayit icin:
    Quadrotor (TRAINING) [Blue] - rec1.csv
    Quadrotor (BEST) [Red] - rec1.csv

Tacview'da acma:
    1) "Quadrotor (TRAINING) [Blue] - rec1.csv" dosyasini File->Open ile ac
    2) "Quadrotor (BEST) [Red] - rec1.csv" dosyasini File->Merge ile ekle
"""

import argparse
import csv
import math
from pathlib import Path

from drone_rl.dogfight.config import load_dogfight_config
from drone_rl.dogfight.dogfight_env import (
    DogfightEnv, NormalizerStats, LAT0_DEG, LON0_DEG, FT_PER_DEG_LAT,
)
from drone_rl.dogfight.env_factory import load_opponent_controller, make_dummy_vecnorm_env
from drone_rl.dogfight.checkpoint_pool import CheckpointPool

FT_TO_M = 0.3048

# Tacview'un resmi CSV sutun basliklari (dokumantasyona birebir uygun)
CSV_HEADER = ["Time", "Longitude", "Latitude", "Altitude", "Roll", "Pitch", "Yaw"]


def _find_repo_root() -> Path:
    candidates = [Path.cwd()]
    here = Path(__file__).resolve()
    candidates += [here.parent, here.parent.parent, here.parent.parent.parent]
    for c in candidates:
        if (c / "configs").is_dir() and (c / "src" / "drone_rl").is_dir():
            return c
    tried = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "Repo koku otomatik bulunamadi. Denenen yerler:\n" + tried +
        "\n'cd /content/repo && python tools/export_tacview_csv.py' seklinde "
        "calistirin ya da --config/--live-snapshot-dir/--pool verin."
    )


REPO_ROOT = _find_repo_root()
RUNS_DIR = REPO_ROOT / "runs"
EXPORT_DIR = REPO_ROOT / "exports"


def _north_east_to_lonlat(north_ft: float, east_ft: float):
    """DogfightEnv._position()'un TAM TERSI - ayni sabitlerle, boylece
    CSV'deki konum ortamin kendi ic tutarliligiyla BIREBIR eslesir."""
    ft_per_deg_lon = FT_PER_DEG_LAT * math.cos(math.radians(LAT0_DEG))
    lat = LAT0_DEG + north_ft / FT_PER_DEG_LAT
    lon = LON0_DEG + east_ft / ft_per_deg_lon
    return lon, lat


def _heading_deg(yaw_rad: float) -> float:
    """Tacview 'Yaw' alani: gercek kuzeye (true north) gore, derece.
    JSBSim'in psi'si zaten kuzeyden saat yonunde olcup ayni
    konvansiyonu kullandigi icin sadece radyandan dereceye cevirip
    0-360 araligina sariyoruz."""
    return math.degrees(yaw_rad) % 360.0


class _TrainingController:
    def __init__(self, model_path, vecnorm_path):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize

        vecnorm = VecNormalize.load(str(vecnorm_path), make_dummy_vecnorm_env())
        self.stats = NormalizerStats(vecnorm)
        self.model = PPO.load(str(model_path), device="cpu")

    def compute_action(self, env):
        obs = env._get_obs_for(env.fdm_self, env.fdm_opp, env.prev_action_self)
        norm_obs = self.stats.normalize(obs).reshape(1, -1)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        return action[0]


def _build_eval_env(config_path, live_snapshot_dir, pool_dir):
    cfg = load_dogfight_config(config_path)

    live_dir = Path(live_snapshot_dir)
    model_path = live_dir / "model.zip"
    vecnorm_path = live_dir / "vecnormalize.pkl"
    if not (model_path.exists() and vecnorm_path.exists()):
        raise FileNotFoundError(
            f"Egitim anlik goruntusu bulunamadi: {live_dir}\n"
            f"Stage B en az bir snapshot yazana kadar bekleyin."
        )

    pool = CheckpointPool(pool_dir)
    if len(pool) == 0:
        raise FileNotFoundError(f"Havuz bos: {pool_dir}. Once seed-pool calistirin.")

    opp_controller = load_opponent_controller(*pool.latest(), deterministic=True)
    env = DogfightEnv(cfg.env, opponent_controller=opp_controller)
    env.set_shaped_weight(cfg.env.shaped_weight_end)
    env.set_curriculum_progress(1.0)
    # YENI: sadece bu kayit ortami icin - egitim config'i etkilenmez.
    env.set_terminate_on_fault(False)

    training_ctrl = _TrainingController(model_path, vecnorm_path)
    return env, training_ctrl, pool.latest_version()


def run_continuous_recording(env, training_ctrl, min_duration_s: float, max_episodes: int):
    """Ortami, TOPLAM sure en az min_duration_s olana kadar art arda
    bolumler halinde kosturur; Time sutunu KESINTISIZ (bolumler arasi
    sifirlanmadan) artmaya devam eder. Bir bolum crash/timeout ile
    biterse ortam otomatik resetlenip kayit devam eder.

    Doner: (training_rows, best_rows, toplam_sure_s, kosturulan_bolum_sayisi)
    """
    training_rows, best_rows = [], []
    total_time = 0.0
    episodes_run = 0

    while total_time < min_duration_s and episodes_run < max_episodes:
        env.reset()
        episodes_run += 1
        info = None
        while True:
            action = training_ctrl.compute_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_time += env.control_dt
            t = round(total_time, 3)

            self_n, self_e, self_alt = info["self_pos"]
            opp_n, opp_e, opp_alt = info["opp_pos"]
            s_roll, s_pitch, s_yaw = info["self_attitude"]
            o_roll, o_pitch, o_yaw = info["opp_attitude"]

            s_lon, s_lat = _north_east_to_lonlat(self_n, self_e)
            o_lon, o_lat = _north_east_to_lonlat(opp_n, opp_e)

            training_rows.append([
                f"{t:.2f}", f"{s_lon:.9f}", f"{s_lat:.9f}", f"{self_alt * FT_TO_M:.2f}",
                f"{math.degrees(s_roll):.3f}", f"{math.degrees(s_pitch):.3f}",
                f"{_heading_deg(s_yaw):.3f}",
            ])
            best_rows.append([
                f"{t:.2f}", f"{o_lon:.9f}", f"{o_lat:.9f}", f"{opp_alt * FT_TO_M:.2f}",
                f"{math.degrees(o_roll):.3f}", f"{math.degrees(o_pitch):.3f}",
                f"{_heading_deg(o_yaw):.3f}",
            ])

            ended = terminated or truncated
            if ended:
                print(f"    bolum {episodes_run} bitti (t={total_time:.1f}s, "
                     f"sebep={info['reset_reason']}) - hedefe ulasilmadiysa devam ediliyor")
                break
            if total_time >= min_duration_s:
                break

    if total_time < min_duration_s:
        print(f"  UYARI: {max_episodes} bolume ragmen {min_duration_s}s hedefine "
             f"ulasilamadi (toplam {total_time:.1f}s). Bolumler beklenenden cok "
             f"kisa suruyor olabilir - env/egitim durumunu kontrol edin.")

    return training_rows, best_rows, total_time, episodes_run


def write_tacview_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--live-snapshot-dir", type=str, default=None)
    ap.add_argument("--pool", type=str, default=None)
    ap.add_argument("--recordings", type=int, default=1,
                    help="Kac ayri kayit (dosya cifti) uretilecek, her biri en az --min-duration-s")
    ap.add_argument("--min-duration-s", type=float, default=60.0,
                    help="Her kaydin en az kac saniye surecegi (varsayilan 60)")
    ap.add_argument("--max-episodes", type=int, default=40,
                    help="Bir kayit icinde art arda kosturulacak en fazla bolum sayisi")
    args = ap.parse_args()

    config_path = args.config or str(REPO_ROOT / "configs" / "dogfight_stage_b.yaml")
    live_snapshot_dir = args.live_snapshot_dir or str(RUNS_DIR / "dogfight_stage_b" / "live_snapshot")
    pool_dir = args.pool or str(RUNS_DIR / "dogfight_pool")

    print(f"repo koku      : {REPO_ROOT}")
    print(f"config         : {config_path}")
    print(f"live-snapshot  : {live_snapshot_dir}")
    print(f"pool           : {pool_dir}")

    env, training_ctrl, pool_version = _build_eval_env(config_path, live_snapshot_dir, pool_dir)

    print(f"BEST (havuz)   : v{pool_version}")
    print(f"{args.recordings} kayit Tacview CSV olarak uretiliyor, her biri en az "
         f"{args.min_duration_s:.0f}s (gerekirse bolumler otomatik birlestirilecek)...")

    for rec in range(1, args.recordings + 1):
        training_rows, best_rows, total_time, episodes_run = run_continuous_recording(
            env, training_ctrl, min_duration_s=args.min_duration_s,
            max_episodes=args.max_episodes)

        training_path = EXPORT_DIR / f"Quadrotor (TRAINING) [Blue] - rec{rec}.csv"
        best_path = EXPORT_DIR / f"Quadrotor (BEST) [Red] - rec{rec}.csv"

        write_tacview_csv(training_path, training_rows)
        write_tacview_csv(best_path, best_rows)

        print(f"  kayit {rec}: {episodes_run} bolum birlestirildi, toplam {total_time:.1f}s")
        print(f"    -> {training_path.name}")
        print(f"    -> {best_path.name}")

    print("\nTamamlandi. Tacview'da:")
    print("  1) 'Quadrotor (TRAINING) ...csv' dosyasini File->Open ile acin")
    print("  2) 'Quadrotor (BEST) ...csv' dosyasini File->Merge ile ekleyin")


if __name__ == "__main__":
    main()

