"""F450 hover gorevi icin PPO veya SAC egitimi + EvalCallback.

--algo ppo (varsayilan) veya --algo sac ile hangi algoritmanin
kullanilacagi secilir. Ortam, callback'ler ve kayit mantigi her iki
algoritma icin de ayni; sadece model olusturma satiri degisir.
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

from drone_rl.config import load_config
from drone_rl.env_factory import make_training_vec_env


class SaveVecNormalizeCallback(BaseCallback):
    """EvalCallback her yeni en iyi modeli bulduğunda, o anki
    VecNormalize istatistiklerini de best_model klasörüne kaydeder.
    Bunsuz best_model.zip yüklendiğinde yanlış olceklenmis
    gozlemlerle calisir (best_model aninda kaydedilen istatistikler,
    egitim sonunda kaydedilenle ayni olmayabilir)."""

    def __init__(self, save_path: Path):
        super().__init__()
        self.save_path = Path(save_path)

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None:
            self.save_path.mkdir(parents=True, exist_ok=True)
            vec_normalize.save(str(self.save_path / "vecnormalize_best.pkl"))
        return True


def build_model(algo: str, cfg, venv, tensorboard_log: str):
    """Secilen algoritmaya gore PPO veya SAC modeli olusturur.
    Ortam (venv) ve kayit/callback mantigi her iki durumda da aynidir;
    sadece burasi algoritmaya ozel hiperparametreleri kullanir."""
    if algo == "ppo":
        return PPO(
            cfg.ppo.policy, venv,
            n_steps=cfg.ppo.n_steps, batch_size=cfg.ppo.batch_size, n_epochs=cfg.ppo.n_epochs,
            gamma=cfg.ppo.gamma, gae_lambda=cfg.ppo.gae_lambda, clip_range=cfg.ppo.clip_range,
            learning_rate=cfg.ppo.learning_rate, ent_coef=cfg.ppo.ent_coef,
            verbose=1, device="cpu",
            tensorboard_log=tensorboard_log,
        )
    elif algo == "sac":
        return SAC(
            cfg.sac.policy, venv,
            learning_rate=cfg.sac.learning_rate,
            buffer_size=cfg.sac.buffer_size,
            learning_starts=cfg.sac.learning_starts,
            batch_size=cfg.sac.batch_size,
            tau=cfg.sac.tau,
            gamma=cfg.sac.gamma,
            train_freq=cfg.sac.train_freq,
            gradient_steps=cfg.sac.gradient_steps,
            ent_coef=cfg.sac.ent_coef,
            verbose=1, device="cpu",
            tensorboard_log=tensorboard_log,
        )
    else:
        raise ValueError(f"Bilinmeyen algoritma: {algo!r} (ppo veya sac olmali)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", type=str, choices=["ppo", "sac"], default="ppo",
                     help="Egitim algoritmasi (varsayilan: ppo)")
    ap.add_argument("--config", type=str, default=None,
                     help="configs/ppo_hover.yaml gibi bir config dosyasi; "
                          "verilmezse kod-ici varsayilanlar kullanilir "
                          "(refactor oncesi hardcoded degerlerle ayni)")
    ap.add_argument("--timesteps", type=int, default=None,
                     help="Verilirse config'teki train.timesteps'i gecersiz kilar")
    ap.add_argument("--n-envs", type=int, default=None,
                     help="Verilirse config'teki train.n_envs'i gecersiz kilar")
    ap.add_argument("--out", type=str, default="/content/runs/hover",
                     help="Cikti klasoru. PPO ve SAC AYNI --out ile calistirilirsa "
                          "birbirinin dosyalarinin uzerine yazar; farkli algoritmalar "
                          "icin farkli klasor kullan (ornek: hover_ppo_v1, hover_sac_v1).")
    ap.add_argument("--eval-freq", type=int, default=10000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    timesteps = args.timesteps if args.timesteps is not None else cfg.train.timesteps
    n_envs = args.n_envs if args.n_envs is not None else cfg.train.n_envs

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Egitim Ortami
    venv = make_training_vec_env(cfg.env, n_envs=n_envs, training=True, norm_reward=True)

    # Degerlendirme Ortami (Callback icin) - normalizasyon istatistikleri
    # sync_envs_normalization ile EvalCallback tarafindan egitim
    # ortamindan otomatik kopyalanir.
    eval_env = make_training_vec_env(cfg.env, n_envs=1, training=False, norm_reward=False)

    model = build_model(args.algo, cfg, venv, tensorboard_log=str(out / "tb"))

    # 1. Checkpoint Callback: Periyodik kayit
    ckpt_cb = CheckpointCallback(
        save_freq=max(20_000 // n_envs, 1),
        save_path=str(out / "ckpt"),
        name_prefix=args.algo,
    )

    # 2. Eval Callback: En iyi modeli bulma ve test
    best_model_path = out / "best_model"
    save_vecnorm_cb = SaveVecNormalizeCallback(best_model_path)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_path),
        callback_on_new_best=save_vecnorm_cb,
        log_path=str(out / "logs"),
        eval_freq=max(args.eval_freq // n_envs, 1),
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=timesteps, callback=[ckpt_cb, eval_cb])

    model.save(out / "model_final")
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"Egitim tamamlandi ve kaydedildi ({args.algo}):", out)
    print("  Son model      :", out / "model_final.zip", "+", out / "vecnormalize.pkl")
    print("  En iyi model   :", best_model_path / "best_model.zip", "+", best_model_path / "vecnormalize_best.pkl")


if __name__ == "__main__":
    main()
