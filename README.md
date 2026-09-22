# F450 Self-Play Dogfight

Reinforcement-learning-based autonomous flight control and aerial pursuit for an F450
quadrotor, simulated in full 6-DOF physics (JSBSim) and trained with Proximal Policy
Optimization (PPO / Stable-Baselines3).

The project has two stages:

- **Stage A — Single-agent flight control.** A single F450 learns to climb from a random
  spawn altitude to a target altitude along a specific heading, using a curriculum of
  scripted opponents to bootstrap basic stability and pursuit behaviour.
- **Stage B — Self-play dogfight.** Two F450s are placed in a shared arena. The active
  learner ("TRAINING") is trained against a growing pool of its own frozen past
  checkpoints ("BEST"), promoted only when it proves measurably superior in held-out
  evaluation.

---

## Highlights

- **30-dimensional observation space**, signed body-axis line-of-sight vector so the
  agent can infer *which way* to turn from a single frame, not just how far off-target
  it is.
- **Multi-component reward** (Gaussian pursuit-geometry shaping, distance/closing terms,
  WEZ/HP damage model, graduated safety penalties, terminal win/loss bonus).
- **Self-play with evidence-gated promotion** — a checkpoint is only promoted to the
  opponent pool after beating the current best in ≥55% of evaluation episodes *and*
  improving mean reward by ≥5%, sustained over two consecutive evaluations.
- **PFSP-lite opponent sampling** — win-rate-weighted softmax sampling from the
  checkpoint pool, instead of naive uniform sampling over stale opponents.
- **Defensive numerical safety net** — observation/reward clipping plus an independent
  physics-breakdown threshold (extreme tilt / speed / altitude), engineered after a real
  NaN-cascade failure during training.
- **Live 3D web demo** (FastAPI + WebSocket + Three.js) showing TRAINING and BEST
  simultaneously, auto-reloading as training progresses.
- **Tacview (ACMI 2.2) export** with continuous-recording mode — a clip's episode length
  is extended past the recording target so it completes inside a single uninterrupted
  episode.
- **Permutation-based feature importance** for interpreting what the trained policy
  actually attends to.

---


## Installation

```bash
pip install -r requirements.txt
```

Requires a working JSBSim installation with the F450 flight dynamics model available on
its aircraft search path.

---

## Usage

All commands are run through `main.py`.

**1. Calibrate control signs** (run once per new JSBSim model/build):
```bash
python main.py calibrate
```

**2. Train Stage A** (single-agent, scripted curriculum):
```bash
python main.py train-a
# resume after an interruption:
python main.py train-a --resume
```

**3. Seed the self-play pool** with the Stage A result:
```bash
python main.py seed-pool
```

**4. Train Stage B** (self-play dogfight):
```bash
python main.py train-b
python main.py train-b --resume
```

**5. Inspect the checkpoint pool:**
```bash
python main.py pool-info
```

**6. Launch the live 3D demo:**
```bash
python main.py demo
```

**7. Export a Tacview recording:**
```bash
python tools/export_acmi.py --min-duration-s 60
```

**8. Run feature-importance analysis:**
```bash
python tools/feature_importance.py --episodes 5
```

---

## Reward Function (Stage B)

```
R = R_track − R_threat − R_dist + R_close − R_tooclose
    + R_lock − R_exposed − R_control − R_safety + R_terminal
```

| Term | Meaning |
|---|---|
| `R_track` / `R_threat` | Gaussian pursuit-geometry reward/penalty (ATA/AA alignment) |
| `R_dist` / `R_close` | Distance penalty / closing-rate reward |
| `R_tooclose` | Penalty for approaching the collision boundary |
| `R_lock` / `R_exposed` | Reward/penalty for being inside the opponent's weapon-engagement cone |
| `R_control` | Penalty for abrupt or aggressive control input |
| `R_safety` | Graduated penalty for approaching altitude/boundary/descent-rate limits |
| `R_terminal` | One-time terminal reward: ±50 for win/loss, −25 for collision |

Full derivation and per-term weights are in `src/drone_rl/dogfight/config.py`
(`DogfightEnvConfig`).

---

## Self-Play Mechanism

```
TRAINING  ──trains against──►  opponent sampled from Checkpoint Pool
    │                                     ▲
    └── periodic evaluation ──────────────┘
        (win-rate ≥ 55%, mean-reward +5%,
         2 consecutive passes required)
                │
                ▼
        promoted as new pool version (v_N+1)
```

The opponent pool is sampled with a win-rate-weighted softmax (PFSP-lite): mostly the
latest version, occasionally an older one, biased toward stronger past checkpoints so
training time isn't wasted against already-beaten opponents.
