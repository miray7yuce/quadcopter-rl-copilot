
"""Dogfight egitim betigi - v8.

Degisiklikler:
  * StandoffCurriculumCallback -> ProgressCallback: hem shaped odul
    agirligini sonumler (AOS makalesindeki lambda_r decay) hem de rakip
    mufredatinin ilerlemesini ortamlara bildirir.
  * DiagnosticsCallback: TensorBoard'a ATA, menzil, closing, koni
    yakalama orani, HP ve reset sebebi dagilimi yazar. Bu metrikler
    olmadan "neden takip etmiyor" sorusunu olcerek cevaplayamiyorduk.
  * Lineer ogrenme hizi sonumlemesi, target_kl, max_grad_norm, vf_coef.
  * Kazanma olcutu artik koni adim sayisi degil, HP farki/dusurme.
  * SubprocVecEnv secenegi (--vec).
  * seed-pool artik --config'i dikkate aliyor (eskiden sessizce
    varsayilan config ile degerlendiriyordu).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from drone_rl.dogfight.config import load_dogfight_config
from drone_rl.dogfight.env_factory import (
    make_dogfight_training_vec_env, make_dogfight_env, make_dummy_vecnorm_env,
)
from drone_rl.dogfight.checkpoint_pool import CheckpointPool
from drone_rl.dogfight.dogfight_env import NormalizerStats, OBS_DIM


ACTIVATION_MAP = {"tanh": nn.Tanh, "relu": nn.ReLU}


def build_policy_kwargs(cfg_ppo):
    kwargs = {}
    pi_arch = cfg_ppo.net_arch_pi or [256, 256]
    vf_arch = cfg_ppo.net_arch_vf or [256, 256]
    kwargs["net_arch"] = dict(pi=pi_arch, vf=vf_arch)
    if cfg_ppo.activation_fn:
        kwargs["activation_fn"] = ACTIVATION_MAP[cfg_ppo.activation_fn.lower()]
    return kwargs


def linear_schedule(initial: float, final_frac: float):
    """SB3 progress_remaining 1 -> 0 gider."""
    final = initial * float(final_frac)

    def _f(progress_remaining: float) -> float:
        return final + (initial - final) * progress_remaining

    return _f


class ProgressCallback(BaseCallback):
    """Shaped odul agirligi + rakip mufredati ilerlemesi.

    Shaped (yogun) odul basta guclu olmali ki ajan hic sinyal almadan
    kaybolmasin; ilerledikce sonumlenmeli ki terminal (kazanma) sinyali
    baskin hale gelsin ve odul bicimlendirmesi politikayi carpitmasin."""

    def __init__(self, env_cfg, update_every: int = 200):
        super().__init__()
        self.env_cfg = env_cfg
        self.update_every = max(update_every, 1)
        self._last_w = None
        self._last_p = None

    def _on_step(self) -> bool:
        if self.n_calls % self.update_every != 0:
            return True
        cfg = self.env_cfg

        frac_s = min(1.0, self.num_timesteps / max(cfg.shaped_ramp_steps, 1))
        w = cfg.shaped_weight_start + (cfg.shaped_weight_end - cfg.shaped_weight_start) * frac_s
        self.training_env.env_method("set_shaped_weight", w)

        p = min(1.0, self.num_timesteps / max(cfg.curriculum_ramp_steps, 1))
        self.training_env.env_method("set_curriculum_progress", p)

        self.logger.record("curriculum/shaped_weight", w)
        self.logger.record("curriculum/progress", p)
        self._last_w, self._last_p = w, p
        return True


class DiagnosticsCallback(BaseCallback):
    """Rollout boyunca info dict'lerini toplayip TensorBoard'a yazar."""

    def __init__(self):
        super().__init__()
        self._reset()

    def _reset(self):
        self.ata = []
        self.rng = []
        self.closing = []
        self.lock = []
        self.exposed = []
        self.hp_self = []
        self.hp_opp = []
        self.reasons = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        dones = self.locals.get("dones")

        for i, info in enumerate(infos):
            if "ata_deg" not in info:
                continue
            self.ata.append(info["ata_deg"])
            self.rng.append(info["range_ft"])
            self.closing.append(info["closing_fps"])
            self.lock.append(1.0 if info["opp_in_my_cone"] else 0.0)
            self.exposed.append(1.0 if info["me_in_opp_cone"] else 0.0)

            # Episode bittiyse: bitis sebebi ve kalan HP'ler
            if dones is not None and i < len(dones) and dones[i]:
                r = info.get("reset_reason") or "unknown"
                self.reasons[r] = self.reasons.get(r, 0) + 1
                self.hp_self.append(info.get("hp_self", 0.0))
                self.hp_opp.append(info.get("hp_opp", 0.0))
        return True

    def _on_rollout_end(self) -> None:
        if self.ata:
            self.logger.record("dogfight/ata_deg_mean", float(np.mean(self.ata)))
            self.logger.record("dogfight/range_ft_mean", float(np.mean(self.rng)))
            self.logger.record("dogfight/closing_fps_mean", float(np.mean(self.closing)))
            self.logger.record("dogfight/lock_rate", float(np.mean(self.lock)))
            self.logger.record("dogfight/exposed_rate", float(np.mean(self.exposed)))
        if self.hp_self:
            self.logger.record("dogfight/hp_self_end", float(np.mean(self.hp_self)))
            self.logger.record("dogfight/hp_opp_end", float(np.mean(self.hp_opp)))
        total = sum(self.reasons.values())
        if total:
            for r, c in self.reasons.items():
                self.logger.record(f"reset_reason/{r}", c / total)
        self._reset()


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
            meta = {"num_timesteps": int(self.num_timesteps), "obs_dim": int(OBS_DIM)}
            (self.snapshot_dir / "meta.json").write_text(json.dumps(meta))
        return True


def _episode_won(info) -> bool:
    """v8: kazanma artik koni adim sayisiyla degil, HP ile belirlenir."""
    if info.get("reset_reason") == "opponent_down":
        return True
    if info.get("reset_reason") in ("self_down", "self_ground", "self_ceiling",
                                    "self_tumble", "self_boundary"):
        return False
    return float(info.get("hp_opp", 0.0)) < float(info.get("hp_self", 0.0))


def _evaluate_vs_fixed(model, vecnorm_stats: NormalizerStats, env_cfg,
                       fixed_opponent, n_episodes: int, shaped_weight=None):
    env = make_dogfight_env(env_cfg, fixed_opponent=fixed_opponent,
                            deterministic_opponent=True)
    if shaped_weight is not None:
        env.set_shaped_weight(shaped_weight)
    env.set_curriculum_progress(1.0)

    rewards, wins = [], 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        info = {}
        while not done:
            norm_obs = vecnorm_stats.normalize(obs).reshape(1, -1)
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        if _episode_won(info):
            wins += 1
    return float(np.mean(rewards)), wins / max(n_episodes, 1)


class PromotionCallback(BaseCallback):
    def __init__(self, pool: CheckpointPool, env_cfg, promo_cfg, out_dir: Path):
        super().__init__()
        self.pool = pool
        self.env_cfg = env_cfg
        self.promo_cfg = promo_cfg
        self.out_dir = out_dir
        self._consecutive_pass = 0
        self._next_eval = promo_cfg.eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval += self.promo_cfg.eval_freq

        latest = self.pool.latest()
        if latest is None:
            print("[promotion] havuz bos, atlaniyor")
            return True

        vecnorm_train = self.model.get_vec_normalize_env()
        stats = NormalizerStats(vecnorm_train)

        mean_reward, win_rate = _evaluate_vs_fixed(
            self.model, stats, self.env_cfg, latest, self.promo_cfg.n_eval_episodes
        )
        baseline = self.pool.latest_mean_reward()
        if baseline is None or abs(baseline) < 1e-6:
            improve_pct = 100.0
        else:
            improve_pct = (mean_reward - baseline) / abs(baseline) * 100.0

        passed = (win_rate >= self.promo_cfg.win_rate_threshold
                  and improve_pct >= self.promo_cfg.mean_reward_improve_pct)

        self.logger.record("promotion/win_rate", win_rate)
        self.logger.record("promotion/mean_reward", mean_reward)

        print(f"[promotion] step={self.num_timesteps} win_rate={win_rate:.2f} "
              f"mean_reward={mean_reward:.2f} (baseline={baseline}, "
              f"improve={improve_pct:+.1f}%) pass={passed} "
              f"({self._consecutive_pass + (1 if passed else 0)}/"
              f"{self.promo_cfg.consecutive_passes_required})")

        self._consecutive_pass = self._consecutive_pass + 1 if passed else 0

        if self._consecutive_pass >= self.promo_cfg.consecutive_passes_required:
            tmp_model = self.out_dir / f"_promo_candidate_{self.num_timesteps}.zip"
            tmp_vecnorm = self.out_dir / f"_promo_candidate_{self.num_timesteps}_vecnorm.pkl"
            self.model.save(str(tmp_model))
            vecnorm_train.save(str(tmp_vecnorm))

            version = self.pool.add(str(tmp_model), str(tmp_vecnorm), mean_reward,
                                    win_rate, note=f"step={self.num_timesteps}",
                                    obs_dim=OBS_DIM)
            print(f"[promotion] YENI VERSIYON: v{version} havuza eklendi!")

            tmp_model.unlink(missing_ok=True)
            tmp_vecnorm.unlink(missing_ok=True)
            self._consecutive_pass = 0

        return True


def _seed_pool(args):
    from drone_rl.dogfight.dogfight_env import DogfightEnv, OpponentCurriculum

    pool = CheckpointPool(args.pool)
    run_dir = Path(args.from_run)
    model_path = run_dir / "model_final.zip"
    vecnorm_path = run_dir / "vecnormalize.pkl"

    # DUZELTME: eskiden burada load_dogfight_config(None) cagriliyordu,
    # yani --config yok sayilip VARSAYILAN ayarlarla degerlendirme
    # yapiliyordu. Artik verilen config kullaniliyor.
    cfg = load_dogfight_config(args.config)

    vecnorm = VecNormalize.load(str(vecnorm_path), make_dummy_vecnorm_env())
    stats = NormalizerStats(vecnorm)
    if stats.obs_dim != OBS_DIM:
        raise ValueError(
            f"Stage A checkpoint'i {stats.obs_dim} boyutlu, bu surum {OBS_DIM} "
            f"bekliyor. Eski run'i kullanamazsiniz - Stage A'yi bu surumle "
            f"yeniden calistirin."
        )
    model = PPO.load(str(model_path), device="cpu")

    env = DogfightEnv(cfg.env, opponent_curriculum=OpponentCurriculum(cfg.env))
    env.set_curriculum_progress(1.0)
    env.set_shaped_weight(cfg.env.shaped_weight_end)

    rewards, wins = [], 0
    n = 20
    for _ in range(n):
        obs, _ = env.reset()
        done, ep_r, info = False, 0.0, {}
        while not done:
            norm_obs = stats.normalize(obs).reshape(1, -1)
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action[0])
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
        if _episode_won(info):
            wins += 1

    mean_reward = float(np.mean(rewards))
    win_rate = wins / n
    version = pool.add(str(model_path), str(vecnorm_path), mean_reward, win_rate,
                       note="stage-a seed", obs_dim=OBS_DIM)
    print(f"Havuz tohumlandi: v{version}, mean_reward={mean_reward:.2f}, "
          f"win_rate={win_rate:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, choices=["a", "b"])
    ap.add_argument("--config", type=str)
    ap.add_argument("--out", type=str, default="/content/runs/dogfight_run")
    ap.add_argument("--pool", type=str, default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--vec", type=str, default=None, choices=["auto", "dummy", "subproc"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--snapshot-freq", type=int, default=10000)
    ap.add_argument("--seed-pool", action="store_true")
    ap.add_argument("--from", dest="from_run", type=str)
    args = ap.parse_args()

    if args.seed_pool:
        if not args.from_run or not args.pool:
            raise ValueError("--seed-pool icin --from ve --pool gerekli")
        _seed_pool(args)
        return

    cfg = load_dogfight_config(args.config)
    timesteps = args.timesteps or cfg.train.timesteps
    n_envs = args.n_envs or cfg.train.n_envs
    vec = args.vec or cfg.train.vec
    seed = args.seed if args.seed is not None else cfg.train.seed
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    venv = make_dogfight_training_vec_env(
        cfg.env, n_envs=n_envs, stage=args.stage, pool_dir=args.pool,
        training=True, norm_reward=True, clip_reward=cfg.ppo.clip_reward,
        vec=vec, seed=seed,
    )

    policy_kwargs = build_policy_kwargs(cfg.ppo)
    model = PPO(
        cfg.ppo.policy, venv,
        n_steps=cfg.ppo.n_steps, batch_size=cfg.ppo.batch_size, n_epochs=cfg.ppo.n_epochs,
        gamma=cfg.ppo.gamma, gae_lambda=cfg.ppo.gae_lambda, clip_range=cfg.ppo.clip_range,
        learning_rate=linear_schedule(cfg.ppo.learning_rate, cfg.ppo.lr_final_frac),
        ent_coef=cfg.ppo.ent_coef, vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm, target_kl=cfg.ppo.target_kl,
        policy_kwargs=policy_kwargs, verbose=1, device="cpu", seed=seed,
        tensorboard_log=str(out / "tb"),
    )

    ckpt_cb = CheckpointCallback(save_freq=max(20_000 // n_envs, 1),
                                 save_path=str(out / "ckpt"), name_prefix="ppo")
    progress_cb = ProgressCallback(cfg.env)
    diag_cb = DiagnosticsCallback()
    snapshot_cb = PeriodicSnapshotCallback(out, save_freq=max(args.snapshot_freq // n_envs, 1))
    callbacks = [ckpt_cb, progress_cb, diag_cb, snapshot_cb]

    if args.stage == "b":
        if not args.pool:
            raise ValueError("--stage b icin --pool gerekli")
        pool = CheckpointPool(args.pool)
        if len(pool) == 0:
            raise ValueError("Havuz bos - once --seed-pool calistirin")
        callbacks.append(PromotionCallback(pool, cfg.env, cfg.promotion, out))

    model.learn(total_timesteps=timesteps, callback=callbacks)

    model.save(out / "model_final")
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"Egitim tamamlandi: {out}")


if __name__ == "__main__":
    main()
