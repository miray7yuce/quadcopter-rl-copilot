"""Iki drone'un (TRAINING = egitim modeli, BEST = havuzdaki en guncel
dondurulmus rakip) ucusunu Tacview'un KENDI formatinda (.acmi, metin
tabanli ACMI 2.1) kaydeder - dogrudan Tacview'a surukleyip 3B tekrar
izleyebilirsiniz.

DIKKAT: CSV dosyalari Tacview'da ACILMAZ (farkli format). Bu script
TAMAMEN BAGIMSIZ, baska hicbir export script'ine dokunmaz/bagli
degildir - CSV export script'ine hic dokunulmadi.

DUZELTME (surekli kayit): Eskiden TEK bolum kaydediliyordu ve bir
carpisma/crash olunca kayit o anda kesiliyordu - bolumler bazen
~10 saniyede bitince 'daha ne oldugunu anlamadan' kayit sona eriyordu.
Simdi varsayilan olarak EN AZ --min-duration-s (varsayilan 60s) uzunlugunda
KESINTISIZ tek bir dosya uretiliyor: bir bolum biterse (crash/timeout)
ortam otomatik resetlenip kayit AYNI dosyada, AYNI zaman ekseninde
devam ediyor - hedef sureye ulasana kadar. Boylece kisa crash'ler
kaydi erken kesmiyor, en az 1 dakikalik anlamli bir sahne garantileniyor.

Egitim (Stage B) ARKA PLANDA calisirken de guvenle calistirilabilir:
sadece diskteki mevcut snapshot/havuz dosyalarini OKUR, kendi ayri bir
degerlendirme ortaminda calisir - egitim surecine hic mudahale etmez.

Kullanim (repo kokunden):
    cd /content/repo && python tools/export_acmi.py --min-duration-s 60

Cikti (exports/ klasorunde):
    dogfight_recording_1.acmi
    dogfight_recording_2.acmi   (--recordings > 1 verilirse)
    ...
(her 'recording' AYRI bir dosya, ama HER BIRI ICINDE birden fazla
bolum sureklilik icinde birlesik olabilir)

Koordinat notu: DogfightEnv, kuzey/dogu-feet ofsetlerini LAT0_DEG=0.0,
LON0_DEG=0.0 etrafinda enlem/boylama cevirip JSBSim'e veriyor. Bu
script AYNI sabitleri kullanarak north/east ft degerlerini enlem/
boylama geri ceviriyor - yani ACMI'deki konum, ortamin kendi ic
tutarliligiyla BIREBIR eslesir (round-trip test edildi).
"""

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

from drone_rl.dogfight.config import load_dogfight_config
from drone_rl.dogfight.dogfight_env import (
    DogfightEnv, NormalizerStats, LAT0_DEG, LON0_DEG, FT_PER_DEG_LAT,
)
from drone_rl.dogfight.env_factory import load_opponent_controller, make_dummy_vecnorm_env
from drone_rl.dogfight.checkpoint_pool import CheckpointPool

FT_TO_M = 0.3048


def _find_repo_root() -> Path:
    """Repo koku calisma dizinine (cwd) gore bulunur - script'in
    nereye yazildigina bagimli degildir."""
    candidates = [Path.cwd()]
    here = Path(__file__).resolve()
    candidates += [here.parent, here.parent.parent, here.parent.parent.parent]
    for c in candidates:
        if (c / "configs").is_dir() and (c / "src" / "drone_rl").is_dir():
            return c
    tried = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "Repo koku otomatik bulunamadi (icinde hem 'configs/' hem "
        "'src/drone_rl/' olan bir dizin araniyor). Denenen yerler:\n"
        f"{tried}\n"
        "Bu betigi 'cd /content/repo && python tools/export_acmi.py' "
        "seklinde, repo kokunden calistirdiginizdan emin olun; ya da "
        "--config, --live-snapshot-dir, --pool argumanlarini elle verin."
    )


REPO_ROOT = _find_repo_root()
RUNS_DIR = REPO_ROOT / "runs"
EXPORT_DIR = REPO_ROOT / "exports"


def _north_east_to_lonlat(north_ft: float, east_ft: float):
    """DogfightEnv._position()'un TAM TERSI - ayni sabitlerle."""
    ft_per_deg_lon = FT_PER_DEG_LAT * math.cos(math.radians(LAT0_DEG))
    lat = LAT0_DEG + north_ft / FT_PER_DEG_LAT
    lon = LON0_DEG + east_ft / ft_per_deg_lon
    return lon, lat


def _acmi_object_line(obj_id, lon, lat, alt_m, roll_deg, pitch_deg, yaw_deg,
                      name, color):
    t = f"{lon:.9f}|{lat:.9f}|{alt_m:.2f}|{roll_deg:.2f}|{pitch_deg:.2f}|{yaw_deg:.2f}"
    return (f"{obj_id},T={t},Name={name},Color={color},"
           f"Type=Air+Rotorcraft,CallSign={name}")


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


def _build_eval_env(config_path, live_snapshot_dir, pool_dir, min_duration_s: float):
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
    # YENI: EGITIM config'ine (yaml) DOKUNMADAN, sadece BU kayit ortami
    # icin dengesizlik/egim/irtifa/sinir ihlalleri artik bolumu
    # sonlandirmiyor - mentorun istedigi 'kesintisiz uzun ucus kaydi'
    # tam olarak bu. Egitim guvenli/standart sinirlariyla (yaml'daki
    # terminate_on_fault: true) devam ediyor, TAMAMEN ETKILENMEDI.
    # Collision/HP=0/sayisal-sapma/felaket esikleri HALA aktif - yani
    # JSBSim gercekten kirilirsa kayit o bolumu bitirip devam eder,
    # cokme olmaz.
    env.set_terminate_on_fault(False)
    # YENI (KRITIK): bolum suresi, kayit hedefinden (min_duration_s)
    # DAHA UZUN yapiliyor - boylece 60s'lik bir kayit, egitimdeki 45s'lik
    # bolum sinirindan dolayi 2 parcaya BOLUNUP art arda 'birlestirilmis'
    # (reset'li/isinlanmali) olmuyor; TEK, KESINTISIZ bir bolumun tamami
    # kaydediliyor. +10s pay, hedefe TAM ulasilirken bolumun tam o anda
    # bitmemesini garanti eder.
    env.set_max_episode_seconds(min_duration_s + 10.0)

    training_ctrl = _TrainingController(model_path, vecnorm_path)
    return env, training_ctrl, pool.latest_version()


def write_continuous_acmi(path: Path, env, training_ctrl,
                          min_duration_s: float, max_episodes: int) -> tuple:
    """Ortami, TOPLAM sure en az min_duration_s olana kadar art arda
    bolumler halinde kosturup HEPSINI AYNI dosyaya, AYNI (kesintisiz)
    zaman eksenine yazar. Bir bolum crash/timeout ile biterse ortam
    otomatik resetlenip kayit devam eder - boylece kisa bir crash
    kaydi erken kesmez.

    max_episodes: guvenlik siniri - her bolum beklenenden cok kisa
    surerse (orn. surekli erken crash) sonsuz donguye girmemek icin.

    Doner: (toplam_sure_s, kosturulan_bolum_sayisi)
    """
    ref_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "FileType=text/acmi/tacview",
        "FileVersion=2.2",
        f"0,ReferenceTime={ref_time}",
    ]

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

            lines.append(f"#{t}")
            lines.append(_acmi_object_line(
                "1", s_lon, s_lat, self_alt * FT_TO_M,
                math.degrees(s_roll), math.degrees(s_pitch), math.degrees(s_yaw),
                name="TRAINING", color="Blue",
            ))
            lines.append(_acmi_object_line(
                "2", o_lon, o_lat, opp_alt * FT_TO_M,
                math.degrees(o_roll), math.degrees(o_pitch), math.degrees(o_yaw),
                name="BEST", color="Red",
            ))

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

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {episodes_run} bolum birlestirildi, toplam {total_time:.1f}s -> {path}")
    return total_time, episodes_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--live-snapshot-dir", type=str, default=None)
    ap.add_argument("--pool", type=str, default=None)
    ap.add_argument("--recordings", type=int, default=1,
                    help="Kac ayri .acmi dosyasi uretilecek (her biri en az --min-duration-s)")
    ap.add_argument("--min-duration-s", type=float, default=60.0,
                    help="Her kayit dosyasinin en az kac saniye surecegi (varsayilan 60)")
    ap.add_argument("--max-episodes", type=int, default=40,
                    help="Bir kayit icinde art arda kosturulacak en fazla bolum sayisi "
                         "(sonsuz donguyu onleyen guvenlik siniri)")
    args = ap.parse_args()

    config_path = args.config or str(REPO_ROOT / "configs" / "dogfight_stage_b.yaml")
    live_snapshot_dir = args.live_snapshot_dir or str(RUNS_DIR / "dogfight_stage_b" / "live_snapshot")
    pool_dir = args.pool or str(RUNS_DIR / "dogfight_pool")

    print(f"repo koku      : {REPO_ROOT}")
    print(f"config         : {config_path}")
    print(f"live-snapshot  : {live_snapshot_dir}")
    print(f"pool           : {pool_dir}")

    env, training_ctrl, pool_version = _build_eval_env(
        config_path, live_snapshot_dir, pool_dir, min_duration_s=args.min_duration_s)

    print(f"BEST (havuz)   : v{pool_version}")
    print(f"{args.recordings} kayit uretiliyor, her biri EN AZ {args.min_duration_s:.0f}s "
         f"SUREKLI/KESINTISIZ tek bir bolum olarak (bolum suresi bu kayit icin "
         f"{args.min_duration_s + 10:.0f}s'ye ayarlandi, egitim etkilenmedi)...")

    for rec in range(1, args.recordings + 1):
        out_path = EXPORT_DIR / f"dogfight_recording_{rec}.acmi"
        write_continuous_acmi(out_path, env, training_ctrl,
                              min_duration_s=args.min_duration_s,
                              max_episodes=args.max_episodes)

    print("\nTamamlandi. Uretilen .acmi dosyalarini dogrudan Tacview'a "
         "surukleyip birakabilirsiniz (TRAINING=Mavi, BEST=Kirmizi).")


if __name__ == "__main__":
    main()



