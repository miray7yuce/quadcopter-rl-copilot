"""F450 ucus gorevleri icin PPO egitimi + EvalCallback.

--task hover  -> F450HoverEnv (sabit irtifa)
--task flight -> F450FlightEnv (rastgele irtifa + rastgele heading)
"""

import argparse
from pathlib import Path
import numpy as np
import torch.nn as nn
from drone_rl.acmi_writer import ACMIWriter
from drone_rl.utils.units import ft_to_m

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

from drone_rl.config import load_config
from drone_rl.env_factory import (
    make_training_vec_env, make_flight_training_vec_env,
)


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path: Path):
        super().__init__()
        self.save_path = Path(save_path)

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None:
            self.save_path.mkdir(parents=True, exist_ok=True)
            vec_normalize.save(str(self.save_path / "vecnormalize_best.pkl"))
        return True


class ACMISnapshotCallback(BaseCallback):
    def __init__(self, eval_env, out_dir: Path, eval_freq: int, control_dt: float):
        super().__init__()
        self.eval_env = eval_env
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.eval_freq = eval_freq
        self.control_dt = control_dt

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._record_snapshot()
        return True

    def _record_snapshot(self):
        acmi = ACMIWriter(name="F450", obj_type="Air+Rotorcraft+UAV", color="Blue")
        obs = self.eval_env.reset()
        t = 0.0

        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, infos = self.eval_env.step(action)
            telem = infos[0]

            acmi.add_frame(
                t=t,
                lon_deg=telem["lon_deg"],
                lat_deg=telem["lat_deg"],
                alt_m=ft_to_m(telem["alt_sl_ft"]),
                roll_deg=np.degrees(telem["roll_rad"]),
                pitch_deg=np.degrees(telem["pitch_rad"]),
                yaw_deg=np.degrees(telem["yaw_rad"]),
            )
            t += self.control_dt

            if done[0]:
                break

        path = self.out_dir / f"snapshot_{self.num_timesteps}.acmi"
        acmi.save(path)
        print(f"  [ACMI] snapshot kaydedildi: {path}", flush=True)


ACTIVATION_MAP = {"tanh": nn.Tanh, "relu": nn.ReLU}


def build_policy_kwargs(cfg_ppo):
    has_custom = (
        cfg_ppo.net_arch_pi is not None
        or cfg_ppo.net_arch_vf is not None
        or cfg_ppo.activation_fn is not None
    )
    if not has_custom:
        return None

    kwargs = {}
    pi_arch = cfg_ppo.net_arch_pi if cfg_ppo.net_arch_pi is not None else [64, 64]
    vf_arch = cfg_ppo.net_arch_vf if cfg_ppo.net_arch_vf is not None else [64, 64]
    kwargs["net_arch"] = dict(pi=pi_arch, vf=vf_arch)

    if cfg_ppo.activation_fn is not None:
        act_key = cfg_ppo.activation_fn.lower()
        if act_key not in ACTIVATION_MAP:
            raise ValueError(f"Bilinmeyen activation_fn: {cfg_ppo.activation_fn!r}")
        kwargs["activation_fn"] = ACTIVATION_MAP[act_key]

    return kwargs


def build_model(algo: str, cfg, venv, tensorboard_log: str):
    if algo != "ppo":
        raise ValueError(f"Bilinmeyen algoritma: {algo!r} (sadece ppo destekleniyor)")

    policy_kwargs = build_policy_kwargs(cfg.ppo)
    return PPO(
        cfg.ppo.policy, venv,
        n_steps=cfg.ppo.n_steps, batch_size=cfg.ppo.batch_size, n_epochs=cfg.ppo.n_epochs,
        gamma=cfg.ppo.gamma, gae_lambda=cfg.ppo.gae_lambda, clip_range=cfg.ppo.clip_range,
        learning_rate=cfg.ppo.learning_rate, ent_coef=cfg.ppo.ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1, device="cpu",
        tensorboard_log=tensorboard_log,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", type=str, choices=["ppo"], default="ppo")
    ap.add_argument("--task", type=str, choices=["hover", "flight"], default="hover",
                     help="hover: sabit irtifa | flight: rastgele irtifa+heading")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--out", type=str, default="/content/runs/run")
    ap.add_argument("--eval-freq", type=int, default=10000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    timesteps = args.timesteps if args.timesteps is not None else cfg.train.timesteps
    n_envs = args.n_envs if args.n_envs is not None else cfg.train.n_envs

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.task == "hover":
        venv = make_training_vec_env(cfg.env, n_envs=n_envs, training=True, norm_reward=True)
        eval_env = make_training_vec_env(cfg.env, n_envs=1, training=False, norm_reward=False)
    else:
        venv = make_flight_training_vec_env(cfg.flight_env, n_envs=n_envs, training=True, norm_reward=True)
        eval_env = make_flight_training_vec_env(cfg.flight_env, n_envs=1, training=False, norm_reward=False)

    model = build_model(args.algo, cfg, venv, tensorboard_log=str(out / "tb"))

    ckpt_cb = CheckpointCallback(
        save_freq=max(20_000 // n_envs, 1),
        save_path=str(out / "ckpt"),
        name_prefix=args.algo,
    )

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

    raw_eval_env = eval_env.venv.envs[0].unwrapped
    acmi_snapshot_cb = ACMISnapshotCallback(
        eval_env=eval_env,
        out_dir=out / "acmi_snapshots",
        eval_freq=max(args.eval_freq // n_envs, 1),
        control_dt=raw_eval_env.control_dt,
    )

    model.learn(total_timesteps=timesteps, callback=[ckpt_cb, eval_cb, acmi_snapshot_cb])

    model.save(out / "model_final")
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"Egitim tamamlandi ve kaydedildi ({args.algo}, task={args.task}):", out)
    print("  Son model      :", out / "model_final.zip", "+", out / "vecnormalize.pkl")
    print("  En iyi model   :", best_model_path / "best_model.zip", "+", best_model_path / "vecnormalize_best.pkl")


if __name__ == "__main__":
    main()

