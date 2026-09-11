import argparse
import json
from pathlib import Path

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from drone_rl.dogfight.config import load_dogfight_config
from drone_rl.dogfight.env_factory import make_dogfight_training_vec_env, make_dogfight_env
from drone_rl.dogfight.checkpoint_pool import CheckpointPool
from drone_rl.dogfight.dogfight_env import NormalizerStats


ACTIVATION_MAP = {"tanh": nn.Tanh, "relu": nn.ReLU}


def build_policy_kwargs(cfg_ppo):
    kwargs = {}
    pi_arch = cfg_ppo.net_arch_pi or [128, 128]
    vf_arch = cfg_ppo.net_arch_vf or [128, 128]
    kwargs["net_arch"] = dict(pi=pi_arch, vf=vf_arch)
    if cfg_ppo.activation_fn:
        kwargs["activation_fn"] = ACTIVATION_MAP[cfg_ppo.activation_fn.lower()]
    return kwargs


class StandoffCurriculumCallback(BaseCallback):
    def __init__(self, w_start, w_end, ramp_steps):
        super().__init__()
        self.w_start = w_start
        self.w_end = w_end
        self.ramp_steps = max(ramp_steps, 1)

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / self.ramp_steps)
        w = self.w_start + (self.w_end - self.w_start) * frac
        self.training_env.env_method("set_standoff_weight", w)
        return True


class PeriodicSnapshotCallback(BaseCallback):
    def __init__(self, out_dir: Path, save_freq: int):
        super().__init__()
        self.snapshot_dir = Path(out_dir) / "live_snapshot"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = max(save_freq, 1)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            vecnorm = self.model.get_vec_normalize_env()
            self.model.save(str(self.snapshot_dir / "model"))
            if vecnorm is not None:
                vecnorm.save(str(self.snapshot_dir / "vecnormalize.pkl"))
            # YENI: gercek zamanli arayuzde egitim timestep'ini
            # gostermek icin kucuk bir meta dosyasi da yaziliyor.
            meta = {"num_timesteps": int(self.num_timesteps)}
            (self.snapshot_dir / "meta.json").write_text(json.dumps(meta))
        return True


def _evaluate_vs_fixed(model, vecnorm_stats: NormalizerStats, env_cfg,
                        fixed_opponent, n_episodes: int):
    env = make_dogfight_env(env_cfg, fixed_opponent=fixed_opponent)
    rewards, wins = [], 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            norm_obs = vecnorm_stats.normalize(obs).reshape(1, -1)
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        if info["my_score"] > info["opp_score"]:
            wins += 1
    return float(np.mean(rewards)), wins / n_episodes


class PromotionCallback(BaseCallback):
    def __init__(self, pool: CheckpointPool, env_cfg, promo_cfg, out_dir: Path):
        super().__init__()
        self.pool = pool
        self.env_cfg = env_cfg
        self.promo_cfg = promo_cfg
        self.out_dir = out_dir
        self._consecutive_pass = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.promo_cfg.eval_freq != 0:
            return True

        latest = self.pool.latest()
        if latest is None:
            print("[promotion] havuz bos, atlaniyor")
            return True

        vecnorm_train = self.model.get_vec_normalize_env()
        stats = NormalizerStats(vecnorm_train)

        mean_reward, win_rate = _evaluate_vs_fixed(
            self.model, stats, self.env_cfg, latest, self.promo_cfg.n_eval_episodes
        )
        baseline = self.pool.latest_mean_reward() or 1e-6
        improve_pct = (mean_reward - baseline) / abs(baseline) * 100.0

        passed = (win_rate >= self.promo_cfg.win_rate_threshold
                  and improve_pct >= self.promo_cfg.mean_reward_improve_pct)

        print(f"[promotion] step={self.num_timesteps} win_rate={win_rate:.2f} "
              f"mean_reward={mean_reward:.2f} (baseline={baseline:.2f}, "
              f"improve={improve_pct:+.1f}%) pass={passed} "
              f"({self._consecutive_pass + (1 if passed else 0)}/"
              f"{self.promo_cfg.consecutive_passes_required})")

        if passed:
            self._consecutive_pass += 1
        else:
            self._consecutive_pass = 0

        if self._consecutive_pass >= self.promo_cfg.consecutive_passes_required:
            tmp_model = self.out_dir / f"_promo_candidate_{self.num_timesteps}.zip"
            tmp_vecnorm = self.out_dir / f"_promo_candidate_{self.num_timesteps}_vecnorm.pkl"
            self.model.save(str(tmp_model))
            vecnorm_train.save(str(tmp_vecnorm))

            version = self.pool.add(str(tmp_model), str(tmp_vecnorm), mean_reward, win_rate,
                                     note=f"step={self.num_timesteps}")
            print(f"[promotion] YENI VERSIYON: v{version} havuza eklendi!")

            tmp_model.unlink(missing_ok=True)
            tmp_vecnorm.unlink(missing_ok=True)

            new_latest = self.pool.latest()
            self.training_env.env_method("set_opponent_controller_from_pool", new_latest)
            self._consecutive_pass = 0

        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, choices=["a", "b"])
    ap.add_argument("--config", type=str)
    ap.add_argument("--out", type=str, default="/content/runs/dogfight_run")
    ap.add_argument("--pool", type=str, default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--snapshot-freq", type=int, default=10000)

    ap.add_argument("--seed-pool", action="store_true")
    ap.add_argument("--from", dest="from_run", type=str)

    args = ap.parse_args()

    if args.seed_pool:
        if not args.from_run or not args.pool:
            raise ValueError("--seed-pool icin --from ve --pool gerekli")
        pool = CheckpointPool(args.pool)
        run_dir = Path(args.from_run)
        model_path = run_dir / "model_final.zip"
        vecnorm_path = run_dir / "vecnormalize.pkl"

        cfg = load_dogfight_config(None)
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from drone_rl.dogfight.dogfight_env import DogfightEnv
        dummy = DummyVecEnv([lambda: Monitor(DogfightEnv(cfg.env))])
        vecnorm = VecNormalize.load(str(vecnorm_path), dummy)
        stats = NormalizerStats(vecnorm)
        model = PPO.load(str(model_path), device="cpu")

        env = DogfightEnv(cfg.env)
        rewards, wins = [], 0
        for _ in range(20):
            obs, _ = env.reset()
            done = False
            ep_r = 0.0
            while not done:
                norm_obs = stats.normalize(obs).reshape(1, -1)
                action, _ = model.predict(norm_obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action[0])
                ep_r += r
                done = term or trunc
            rewards.append(ep_r)
            if info["my_score"] > info["opp_score"]:
                wins += 1
        mean_reward = float(np.mean(rewards))
        win_rate = wins / 20

        version = pool.add(str(model_path), str(vecnorm_path), mean_reward, win_rate,
                            note="stage-a seed")
        print(f"Havuz tohumlandi: v{version}, mean_reward={mean_reward:.2f}, win_rate={win_rate:.2f}")
        return

    cfg = load_dogfight_config(args.config)
    timesteps = args.timesteps or cfg.train.timesteps
    n_envs = args.n_envs or cfg.train.n_envs
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    venv = make_dogfight_training_vec_env(
        cfg.env, n_envs=n_envs, stage=args.stage, pool_dir=args.pool,
        training=True, norm_reward=True,
    )

    policy_kwargs = build_policy_kwargs(cfg.ppo)
    model = PPO(
        cfg.ppo.policy, venv,
        n_steps=cfg.ppo.n_steps, batch_size=cfg.ppo.batch_size, n_epochs=cfg.ppo.n_epochs,
        gamma=cfg.ppo.gamma, gae_lambda=cfg.ppo.gae_lambda, clip_range=cfg.ppo.clip_range,
        learning_rate=cfg.ppo.learning_rate, ent_coef=cfg.ppo.ent_coef,
        policy_kwargs=policy_kwargs, verbose=1, device="cpu",
        tensorboard_log=str(out / "tb"),
    )

    ckpt_cb = CheckpointCallback(save_freq=max(20_000 // n_envs, 1),
                                  save_path=str(out / "ckpt"), name_prefix="ppo")
    standoff_cb = StandoffCurriculumCallback(
        cfg.env.standoff_weight_start, cfg.env.standoff_weight_end, cfg.env.standoff_ramp_steps
    )
    snapshot_cb = PeriodicSnapshotCallback(out, save_freq=max(args.snapshot_freq // n_envs, 1))
    callbacks = [ckpt_cb, standoff_cb, snapshot_cb]

    if args.stage == "b":
        if not args.pool:
            raise ValueError("--stage b icin --pool gerekli")
        pool = CheckpointPool(args.pool)
        if len(pool) == 0:
            raise ValueError("Havuz bos - once --seed-pool calistirin")
        promo_cb = PromotionCallback(pool, cfg.env, cfg.promotion, out)
        callbacks.append(promo_cb)

    model.learn(total_timesteps=timesteps, callback=callbacks)

    model.save(out / "model_final")
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"Egitim tamamlandi: {out}")


if __name__ == "__main__":
    main()
