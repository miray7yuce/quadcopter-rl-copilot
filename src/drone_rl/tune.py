"""Optuna ile PPO ve SAC icin hiperparametre optimizasyonu.

Bu, train.py'nin bir 'modu' degil, ayri bir arac: her calisma (trial)
kisa bir egitim yapip sonucu (ortalama reward) Optuna'ya bildiriyor,
Optuna da bir sonraki denemede hangi hiperparametreleri deneyecegini
bu geri bildirime gore seciyor.

Kullanim:
    python -m drone_rl.tune --algo ppo --n-trials 30 --timesteps-per-trial 60000 --out /content/tuning/ppo
    python -m drone_rl.tune --algo sac --n-trials 30 --timesteps-per-trial 60000 --out /content/tuning/sac
"""

import argparse
import json
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback

from drone_rl.config import load_config
from drone_rl.env_factory import make_training_vec_env


class TrialEvalCallback(EvalCallback):
    """Normal EvalCallback gibi periyodik degerlendirme yapar, ama
    her degerlendirme sonucunu Optuna'ya da raporlar (Optuna kotu
    giden denemeleri erken kesebilsin diye) VE ekrana bir ilerleme
    satiri yazdirir (uzun trial'lar sirasinda 'calisiyor mu' belirsizligini
    onlemek icin)."""

    def __init__(self, eval_env, trial, total_timesteps, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.trial = trial
        self.total_timesteps = total_timesteps
        self.eval_idx = 0

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            print(
                f"  [trial {self.trial.number}] "
                f"{self.num_timesteps}/{self.total_timesteps} adim - "
                f"ortalama reward: {self.last_mean_reward:.2f}",
                flush=True,
            )
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                print(f"  [trial {self.trial.number}] erken kesildi (prune)", flush=True)
                raise optuna.TrialPruned()
        return continue_training


def suggest_ppo_params(trial: optuna.Trial) -> dict:
    """PPO icin arama uzayi. Yaygin/etkili PPO hiperparametreleri."""
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": n_steps,
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "n_epochs": trial.suggest_int("n_epochs", 3, 30),
        "gamma": trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
    }


def suggest_sac_params(trial: optuna.Trial) -> dict:
    """SAC icin arama uzayi. train_freq ve gradient_steps'i esit
    tutuyoruz (yaygin pratik: her N adimda bir, N kadar guncelleme)."""
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32])
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [10_000, 100_000, 1_000_000]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "tau": trial.suggest_float("tau", 0.001, 0.02, log=True),
        "gamma": trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]),
        "train_freq": train_freq,
        "gradient_steps": train_freq,
        "learning_starts": 1000,
        "ent_coef": "auto",
    }


def build_trial_model(algo: str, params: dict, venv):
    if algo == "ppo":
        return PPO(
            "MlpPolicy", venv,
            n_steps=params["n_steps"], batch_size=params["batch_size"],
            n_epochs=params["n_epochs"], gamma=params["gamma"],
            gae_lambda=params["gae_lambda"], clip_range=params["clip_range"],
            learning_rate=params["learning_rate"], ent_coef=params["ent_coef"],
            verbose=0, device="cpu",
        )
    elif algo == "sac":
        return SAC(
            "MlpPolicy", venv,
            learning_rate=params["learning_rate"], buffer_size=params["buffer_size"],
            learning_starts=params["learning_starts"], batch_size=params["batch_size"],
            tau=params["tau"], gamma=params["gamma"],
            train_freq=params["train_freq"], gradient_steps=params["gradient_steps"],
            ent_coef=params["ent_coef"],
            verbose=0, device="cpu",
        )
    else:
        raise ValueError(f"Bilinmeyen algoritma: {algo!r}")


def make_objective(algo: str, cfg, args):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_ppo_params(trial) if algo == "ppo" else suggest_sac_params(trial)

        print(f"\n=== Trial {trial.number} basladi ===", flush=True)
        print(f"  Parametreler: {params}", flush=True)

        venv = make_training_vec_env(cfg.env, n_envs=args.n_envs, training=True, norm_reward=True)
        eval_env = make_training_vec_env(cfg.env, n_envs=1, training=False, norm_reward=False)

        model = build_trial_model(algo, params, venv)

        eval_cb = TrialEvalCallback(
            eval_env,
            trial=trial,
            total_timesteps=args.timesteps_per_trial,
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            deterministic=True,
            render=False,
            verbose=0,
        )

        try:
            model.learn(total_timesteps=args.timesteps_per_trial, callback=eval_cb)
        except optuna.TrialPruned:
            venv.close()
            eval_env.close()
            raise

        reward = eval_cb.last_mean_reward
        venv.close()
        eval_env.close()

        print(f"=== Trial {trial.number} bitti - sonuc: {reward:.2f} ===", flush=True)

        # Optuna NaN/None ile calismaz; cok kotu bir deneme oldugunu belirtmek
        # icin cok dusuk bir sayi donduruyoruz.
        if reward is None or reward != reward:  # reward != reward -> NaN kontrolu
            return -1e6
        return float(reward)

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", type=str, choices=["ppo", "sac"], required=True,
                     help="Hangi algoritma icin hiperparametre aranacak")
    ap.add_argument("--config", type=str, default=None,
                     help="Ortam ayarlari icin configs/ppo_hover.yaml gibi bir dosya "
                          "(sadece env: bolumu kullanilir, ppo:/sac: bolumleri "
                          "bu aramada gecersiz kilinir)")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--timesteps-per-trial", type=int, default=60_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--eval-freq", type=int, default=10_000)
    ap.add_argument("--out", type=str, default="/content/tuning/result")
    ap.add_argument("--study-name", type=str, default=None)
    ap.add_argument("--storage", type=str, default=None,
                     help="Optuna icin kalici depolama (ornek: sqlite:////content/tuning/study.db). "
                          "Verilmezse calisma bittiginde sonuclar kaybolur, sadece --out'a "
                          "yazilan json/csv kalir.")
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.INFO)

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name=args.study_name or f"{args.algo}_hover",
        storage=args.storage,
        load_if_exists=True,
    )

    print(f"Arama basliyor: algo={args.algo}, n_trials={args.n_trials}, "
          f"timesteps_per_trial={args.timesteps_per_trial}", flush=True)

    study.optimize(make_objective(args.algo, cfg, args), n_trials=args.n_trials)

    print("\n=== Arama tamamlandi ===")
    print("En iyi deger (ortalama reward):", study.best_value)
    print("En iyi parametreler:", study.best_params)

    best_path = out / f"best_params_{args.algo}.json"
    with open(best_path, "w") as f:
        json.dump({"algo": args.algo, "best_value": study.best_value,
                    "best_params": study.best_params}, f, indent=2)
    print("Kaydedildi:", best_path)

    trials_csv = out / f"trials_{args.algo}.csv"
    study.trials_dataframe().to_csv(trials_csv, index=False)
    print("Tum denemeler:", trials_csv)


if __name__ == "__main__":
    main()
