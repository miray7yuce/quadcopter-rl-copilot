"""F450 hover gorevi icin PPO egitimi + EvalCallback."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

from drone_rl.envs.f450_env import F450HoverEnv


def make_env():
    return Monitor(F450HoverEnv())


class SaveVecNormalizeCallback(BaseCallback):
    """EvalCallback her yeni en iyi modeli bulduğunda, o anki
    VecNormalize istatistiklerini de best_model klasörüne kaydeder.
    Bunsuz best_model.zip yüklendiğinde yanlış olceklenmis
    gozlemlerle calisir (bkz. train_final ile kaydedilen istatistikler
    en iyi model anindakiyle ayni olmayabilir)."""

    def __init__(self, save_path: Path):
        super().__init__()
        self.save_path = Path(save_path)

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None:
            self.save_path.mkdir(parents=True, exist_ok=True)
            vec_normalize.save(str(self.save_path / "vecnormalize_best.pkl"))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--out", type=str, default="/content/runs/hover")
    ap.add_argument("--eval-freq", type=int, default=10000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Egitim Ortami
    venv = DummyVecEnv([make_env for _ in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Degerlendirme Ortami (Callback icin)
    eval_env = DummyVecEnv([make_env])
    # Egitim ortaminin normalizasyon istatistiklerini kullanmasi icin wrap ediyoruz
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    model = PPO(
        "MlpPolicy", venv,
        n_steps=1024, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        learning_rate=3e-4, ent_coef=0.0,
        verbose=1, device="cpu",
        tensorboard_log=str(out / "tb"),
    )

    # 1. Checkpoint Callback: Periyodik kayit
    ckpt_cb = CheckpointCallback(
        save_freq=max(20_000 // args.n_envs, 1),
        save_path=str(out / "ckpt"),
        name_prefix="ppo",
    )

    # 2. Eval Callback: En iyi modeli bulma ve test
    #    callback_on_new_best sayesinde en iyi model bulundugunda
    #    o anki VecNormalize istatistikleri de kaydedilir.
    best_model_path = out / "best_model"
    save_vecnorm_cb = SaveVecNormalizeCallback(best_model_path)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_path),
        callback_on_new_best=save_vecnorm_cb,
        log_path=str(out / "logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.timesteps, callback=[ckpt_cb, eval_cb])

    # Egitim sonundaki (son) model ve normalizasyon istatistikleri.
    # Not: best_model.zip + vecnormalize_best.pkl, egitim boyunca
    # bulunan EN IYI modeldir; model_final.zip + vecnormalize.pkl ise
    # egitimin SONUNDAKI modeldir. Ikisi ayni olmayabilir.
    model.save(out / "model_final")
    venv.save(str(out / "vecnormalize.pkl"))
    print("Egitim tamamlandi ve kaydedildi:", out)
    print("  Son model      :", out / "model_final.zip", "+", out / "vecnormalize.pkl")
    print("  En iyi model   :", best_model_path / "best_model.zip", "+", best_model_path / "vecnormalize_best.pkl")


if __name__ == "__main__":
    main()
