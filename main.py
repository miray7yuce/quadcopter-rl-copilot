#!/usr/bin/env python3
import argparse
import sys
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

CONFIGS_DIR = REPO_ROOT / "configs"
RUNS_DIR = REPO_ROOT / "runs"
HTML_PATH = REPO_ROOT / "dogfightSim_realtime.html"


def _common_train_argv(args, base):
    argv = list(base)
    if args.timesteps:
        argv += ["--timesteps", str(args.timesteps)]
    if args.n_envs:
        argv += ["--n-envs", str(args.n_envs)]
    if getattr(args, "vec", None):
        argv += ["--vec", args.vec]
    if getattr(args, "seed", None) is not None:
        argv += ["--seed", str(args.seed)]
    if getattr(args, "resume", False):
        argv += ["--resume"]
    argv += ["--snapshot-freq", str(args.snapshot_freq)]
    return argv


def cmd_train_a(args):
    from drone_rl.dogfight import train as train_mod
    out = args.out or str(RUNS_DIR / "dogfight_stage_a")
    config = args.config or str(CONFIGS_DIR / "dogfight_stage_a.yaml")
    sys.argv = _common_train_argv(
        args, ["train.py", "--stage", "a", "--config", config, "--out", out])
    train_mod.main()


def cmd_seed_pool(args):
    from drone_rl.dogfight import train as train_mod
    from_run = args.from_run or str(RUNS_DIR / "dogfight_stage_a")
    pool = args.pool or str(RUNS_DIR / "dogfight_pool")
    config = args.config or str(CONFIGS_DIR / "dogfight_stage_a.yaml")
    sys.argv = ["train.py", "--seed-pool", "--from", from_run,
                "--pool", pool, "--config", config]
    train_mod.main()


def cmd_train_b(args):
    from drone_rl.dogfight import train as train_mod
    out = args.out or str(RUNS_DIR / "dogfight_stage_b")
    config = args.config or str(CONFIGS_DIR / "dogfight_stage_b.yaml")
    pool = args.pool or str(RUNS_DIR / "dogfight_pool")
    sys.argv = _common_train_argv(
        args, ["train.py", "--stage", "b", "--config", config,
               "--out", out, "--pool", pool])
    train_mod.main()


def cmd_calibrate(args):
    from drone_rl.dogfight import calibrate as cal_mod
    config = args.config or str(CONFIGS_DIR / "dogfight_stage_a.yaml")
    cal_mod.main(config)


def cmd_pool_info(args):
    from drone_rl.dogfight.checkpoint_pool import CheckpointPool
    pool = CheckpointPool(args.pool or str(RUNS_DIR / "dogfight_pool"))
    print(pool.summary())


def cmd_demo(args):
    from drone_rl.dogfight.realtime_dogfight_server import start_server
    live_snapshot_dir = args.live_snapshot_dir or str(RUNS_DIR / "dogfight_stage_b" / "live_snapshot")
    pool_dir = args.pool or str(RUNS_DIR / "dogfight_pool")
    config = args.config or str(CONFIGS_DIR / "dogfight_stage_b.yaml")
    html_path = args.html or str(HTML_PATH)

    start_server(
        live_snapshot_dir=live_snapshot_dir, pool_dir=pool_dir,
        config_path=config, html_path=html_path,
        demo_max_steps=args.max_steps, reload_interval_s=args.reload_interval,
        port=args.port,
    )

    url = f"http://localhost:{args.port}"
    print(f"\nSunucu hazir: {url}")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    print("Durdurmak icin Ctrl+C.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKapatiliyor.")


def build_parser():
    ap = argparse.ArgumentParser(prog="main.py")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_train_args(p):
        p.add_argument("--config", type=str, default=None)
        p.add_argument("--out", type=str, default=None)
        p.add_argument("--timesteps", type=int, default=None)
        p.add_argument("--n-envs", type=int, default=None)
        p.add_argument("--vec", type=str, default=None,
                       choices=["auto", "dummy", "subproc"])
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--resume", action="store_true",
                       help="live_snapshot'ta kayitli model/vecnormalize varsa oradan devam et")
        p.add_argument("--snapshot-freq", type=int, default=10000)

    p_a = sub.add_parser("train-a")
    add_train_args(p_a)
    p_a.set_defaults(func=cmd_train_a)

    p_seed = sub.add_parser("seed-pool")
    p_seed.add_argument("--from", dest="from_run", type=str, default=None)
    p_seed.add_argument("--pool", type=str, default=None)
    p_seed.add_argument("--config", type=str, default=None)
    p_seed.set_defaults(func=cmd_seed_pool)

    p_b = sub.add_parser("train-b")
    add_train_args(p_b)
    p_b.add_argument("--pool", type=str, default=None)
    p_b.set_defaults(func=cmd_train_b)

    p_cal = sub.add_parser("calibrate")
    p_cal.add_argument("--config", type=str, default=None)
    p_cal.set_defaults(func=cmd_calibrate)

    p_pool = sub.add_parser("pool-info")
    p_pool.add_argument("--pool", type=str, default=None)
    p_pool.set_defaults(func=cmd_pool_info)

    p_demo = sub.add_parser("demo")
    p_demo.add_argument("--live-snapshot-dir", dest="live_snapshot_dir", type=str, default=None)
    p_demo.add_argument("--pool", type=str, default=None)
    p_demo.add_argument("--config", type=str, default=None)
    p_demo.add_argument("--html", type=str, default=None)
    p_demo.add_argument("--port", type=int, default=8020)
    p_demo.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    p_demo.add_argument("--reload-interval", dest="reload_interval", type=float, default=2.0)
    p_demo.add_argument("--no-browser", action="store_true")
    p_demo.set_defaults(func=cmd_demo)

    return ap


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
