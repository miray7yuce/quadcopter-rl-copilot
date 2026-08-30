"""F450 hover gorevi icin PPO egitimi."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from drone_rl.envs.f450_env import F450HoverEnv


def make_env():
    return Monitor(F450HoverEnv())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--out", type=str, default="/content/runs/hover")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    venv = DummyVecEnv([make_env for _ in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", venv,
        n_steps=1024, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        learning_rate=3e-4, ent_coef=0.0,
        verbose=1, device="cpu",
        tensorboard_log=str(out / "tb"),
    )

    ckpt = CheckpointCallback(
        save_freq=max(20_000 // args.n_envs, 1),
        save_path=str(out / "ckpt"),
        name_prefix="ppo",
    )

    model.learn(total_timesteps=args.timesteps, callback=ckpt)
    model.save(out / "model")
    venv.save(str(out / "vecnormalize.pkl"))
    print("kaydedildi:", out)


if __name__ == "__main__":
    main()
