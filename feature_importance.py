"""Egitilmis TRAINING modelinin 30 boyutlu gozlem uzayindaki her
ozelligin ne kadar 'onemli' oldugunu OLCER ve paperdeki Figure 3'e
benzer bir radar (orumcek agi) grafik uretir.

YONTEM: Permutation Importance (permutasyon onem analizi).
  1. Egitilmis politika ile birkac bolum kosturulup gercek gozlem
     vektorleri (X, N x 30) toplanir.
  2. Bu gozlemler uzerinde politikanin verdigi TABAN aksiyonlar
     hesaplanir.
  3. Her boyut (sutun) icin: o boyutun degerleri satirlar arasinda
     RASTGELE KARISTIRILIR (permute edilir), digerleri sabit tutulur,
     politika tekrar calistirilir.
  4. Aksiyon ciktisinin ORTALAMA DEGISIM buyuklugu (L2 norm farki)
     o boyutun 'onemi' olarak kaydedilir - boyut onemliyse
     karistirilinca aksiyon ciddi degisir, onemsizse degisim kucuk
     kalir.
  5. Kararlilik icin islem birkac kez tekrarlanip ortalanir.

Bu, paperdeki Figure 3'ten farkli olarak GERCEK, egitilmis modelden
olculmus bir sonuctur - varsayimsal/gosterim amacli degildir.

Egitim (Stage B) ARKA PLANDA calisirken de guvenle calistirilabilir:
sadece diskteki mevcut snapshot/havuz dosyalarini OKUR, kendi ayri bir
degerlendirme ortaminda calisir.

Kullanim (repo kokunden):
    cd /content/repo && python tools/feature_importance.py --episodes 5

Cikti (exports/ klasorunde):
    feature_importance_radar.png   - grup bazli radar grafik (paper stili)
    feature_importance_raw.csv     - her 30 boyut icin ham/normalize onem
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from drone_rl.dogfight.config import load_dogfight_config
from drone_rl.dogfight.dogfight_env import DogfightEnv, NormalizerStats, OBS_DIM
from drone_rl.dogfight.env_factory import load_opponent_controller, make_dummy_vecnorm_env
from drone_rl.dogfight.checkpoint_pool import CheckpointPool


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
        "\n'cd /content/repo && python tools/feature_importance.py' "
        "seklinde calistirin ya da --config/--live-snapshot-dir/--pool verin."
    )


REPO_ROOT = _find_repo_root()
RUNS_DIR = REPO_ROOT / "runs"
EXPORT_DIR = REPO_ROOT / "exports"

# --------------------------------------------------------------------
# 30 boyutlu gozlem vektorunun anlamli gruplari (bkz. dogfight_env.py
# _get_obs_for() icindeki index yorumlari).
# --------------------------------------------------------------------
FEATURE_NAMES = [
    "roll", "pitch",                                   # 0-1
    "rate_p", "rate_q", "rate_r",                       # 2-4
    "body_u", "body_v", "body_w",                        # 5-7
    "speed",                                             # 8
    "los_x", "los_y", "los_z",                          # 9-11
    "range", "closing", "dz",                            # 12-14
    "cos_ata", "cos_aa",                                 # 15-16
    "opp_vel_x", "opp_vel_y", "opp_vel_z",               # 17-19
    "in_my_cone", "in_other_cone",                       # 20-21
    "altitude", "boundary_dist",                         # 22-23
    "hp_own", "hp_other",                                # 24-25
    "prev_roll", "prev_pitch", "prev_yaw", "prev_thr",   # 26-29
]
assert len(FEATURE_NAMES) == OBS_DIM

GROUPS = {
    "Egocentric Kinematics": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "Engagement Geometry": [9, 10, 11, 12, 13, 14, 15, 16],
    "Opponent State": [17, 18, 19, 20, 21, 24, 25],
    "Safety / Environment": [22, 23],
    "Action Memory": [26, 27, 28, 29],
}


class _TrainingController:
    def __init__(self, model_path, vecnorm_path):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize

        vecnorm = VecNormalize.load(str(vecnorm_path), make_dummy_vecnorm_env())
        self.stats = NormalizerStats(vecnorm)
        self.model = PPO.load(str(model_path), device="cpu")

    def raw_to_action(self, raw_obs_batch: np.ndarray) -> np.ndarray:
        """Ham (normalize edilmemis) gozlem batch'ini (N,30) alip
        deterministik aksiyon batch'i (N,4) doner."""
        norm = self.stats.normalize(raw_obs_batch.astype(np.float32))
        actions, _ = self.model.predict(norm, deterministic=True)
        return actions

    def compute_action(self, env):
        obs = env._get_obs_for(env.fdm_self, env.fdm_opp, env.prev_action_self)
        return self.raw_to_action(obs.reshape(1, -1))[0]


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

    training_ctrl = _TrainingController(model_path, vecnorm_path)
    return env, training_ctrl, pool.latest_version()


def collect_observations(env, training_ctrl, n_episodes: int, max_steps: int) -> np.ndarray:
    """N bolum boyunca TRAINING dronunun GERCEKTEN gordugu ham gozlem
    vektorlerini toplar - boylece permutasyon analizinde kullanilan
    veri dagilimi gercek ucus sirasinda karsilasilan durumlari yansitir."""
    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        step = 0
        while True:
            action = training_ctrl.compute_action(env)
            raw_obs = env._get_obs_for(env.fdm_self, env.fdm_opp, env.prev_action_self)
            rows.append(raw_obs.copy())
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            if terminated or truncated or step >= max_steps:
                break
    return np.stack(rows, axis=0)


def permutation_importance(training_ctrl, X: np.ndarray, n_repeats: int = 5,
                          seed: int = 0) -> np.ndarray:
    """Her boyut icin ortalama aksiyon-degisim buyuklugunu doner
    (OBS_DIM,) seklinde, HENUZ normalize edilmemis (ham) buyukluk."""
    rng = np.random.default_rng(seed)
    n_dims = X.shape[1]

    baseline_actions = training_ctrl.raw_to_action(X)

    importance = np.zeros(n_dims, dtype=np.float64)
    for d in range(n_dims):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, d] = rng.permutation(X_perm[:, d])
            perm_actions = training_ctrl.raw_to_action(X_perm)
            diff = np.linalg.norm(perm_actions - baseline_actions, axis=1)
            deltas.append(diff.mean())
        importance[d] = float(np.mean(deltas))
    return importance


def group_scores(importance: np.ndarray) -> dict:
    """Boyut bazli onemi gruplara gore ortalayip [0,1] araliginda
    normalize eder (paperdeki 'Normalized Feature Importance' ile
    ayni olcekte, dogrudan karsilastirilabilir olsun diye)."""
    raw_group = {g: float(np.mean(importance[idx])) for g, idx in GROUPS.items()}
    max_val = max(raw_group.values()) or 1.0
    return {g: v / max_val for g, v in raw_group.items()}


def plot_radar(group_norm: dict, out_path: Path, subtitle: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(group_norm.keys())
    values = list(group_norm.values())
    n = len(labels)

    angles = [i / n * 2 * math.pi for i in range(n)]
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="gray")

    ax.plot(angles, values, linewidth=2, color="#1f6feb")
    ax.fill(angles, values, color="#1f6feb", alpha=0.25)

    title = "Normalized Feature Importance\nof the Observation Space"
    if subtitle:
        title += f"\n({subtitle})"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=28)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_raw_csv(importance: np.ndarray, out_path: Path):
    max_val = importance.max() or 1.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "group", "raw_importance", "normalized_importance"])
        name_to_group = {}
        for g, idxs in GROUPS.items():
            for i in idxs:
                name_to_group[i] = g
        for i, name in enumerate(FEATURE_NAMES):
            w.writerow([name, name_to_group[i], importance[i], importance[i] / max_val])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--live-snapshot-dir", type=str, default=None)
    ap.add_argument("--pool", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=5,
                    help="Gozlem toplamak icin kosturulacak bolum sayisi")
    ap.add_argument("--repeats", type=int, default=5,
                    help="Her boyut icin permutasyon tekrar sayisi (kararlilik)")
    args = ap.parse_args()

    config_path = args.config or str(REPO_ROOT / "configs" / "dogfight_stage_b.yaml")
    live_snapshot_dir = args.live_snapshot_dir or str(RUNS_DIR / "dogfight_stage_b" / "live_snapshot")
    pool_dir = args.pool or str(RUNS_DIR / "dogfight_pool")

    print(f"repo koku      : {REPO_ROOT}")
    print(f"config         : {config_path}")
    print(f"live-snapshot  : {live_snapshot_dir}")
    print(f"pool           : {pool_dir}")

    env, training_ctrl, pool_version = _build_eval_env(config_path, live_snapshot_dir, pool_dir)
    max_steps = int(env.cfg.episode_seconds * env.cfg.control_hz)

    print(f"BEST (havuz)   : v{pool_version}")
    print(f"{args.episodes} bolum kosturulup gozlemler toplaniyor...")
    X = collect_observations(env, training_ctrl, args.episodes, max_steps)
    print(f"Toplam {X.shape[0]} adim gozlem toplandi. Permutasyon analizi basliyor "
         f"({args.repeats} tekrar/boyut, {OBS_DIM} boyut)...")

    importance = permutation_importance(training_ctrl, X, n_repeats=args.repeats)
    group_norm = group_scores(importance)

    print("\nGrup bazli normalize onem:")
    for g, v in group_norm.items():
        print(f"  {g:24s} {v:.3f}")

    png_path = EXPORT_DIR / "feature_importance_radar.png"
    csv_path = EXPORT_DIR / "feature_importance_raw.csv"
    plot_radar(group_norm, png_path, subtitle=f"pool v{pool_version}, {X.shape[0]} steps")
    write_raw_csv(importance, csv_path)

    print(f"\nTamamlandi:\n  {png_path}\n  {csv_path}")


if __name__ == "__main__":
    main()



