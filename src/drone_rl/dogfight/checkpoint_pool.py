import json
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class CheckpointPool:
    def __init__(self, pool_dir: str):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.pool_dir / "manifest.json"
        self._load()

    def _load(self):
        if self.manifest_path.exists():
            self.entries = json.loads(self.manifest_path.read_text())
        else:
            self.entries = []

    def _save(self):
        self.manifest_path.write_text(json.dumps(self.entries, indent=2))

    def __len__(self):
        return len(self.entries)

    def add(self, model_src: str, vecnorm_src: str, mean_reward: float,
            win_rate: float, note: str = "") -> int:
        version = len(self.entries) + 1
        dst_dir = self.pool_dir / f"v{version}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        model_dst = dst_dir / "model.zip"
        vecnorm_dst = dst_dir / "vecnormalize.pkl"
        shutil.copy(model_src, model_dst)
        shutil.copy(vecnorm_src, vecnorm_dst)
        entry = {
            "version": version,
            "model": str(model_dst),
            "vecnorm": str(vecnorm_dst),
            "mean_reward": float(mean_reward),
            "win_rate": float(win_rate),
            "note": note,
        }
        self.entries.append(entry)
        self._save()
        return version

    def latest(self) -> Optional[Tuple[str, str]]:
        if not self.entries:
            return None
        e = self.entries[-1]
        return e["model"], e["vecnorm"]

    def latest_mean_reward(self) -> Optional[float]:
        if not self.entries:
            return None
        return self.entries[-1]["mean_reward"]

    def sample(self, latest_prob: float = 0.7, rng: Optional[np.random.Generator] = None
               ) -> Optional[Tuple[str, str]]:
        if not self.entries:
            return None
        rng = rng or np.random.default_rng()
        if len(self.entries) == 1 or rng.random() < latest_prob:
            e = self.entries[-1]
        else:
            idx = int(rng.integers(0, len(self.entries) - 1))
            e = self.entries[idx]
        return e["model"], e["vecnorm"]

    def summary(self) -> str:
        lines = [f"Pool: {self.pool_dir} ({len(self.entries)} versiyon)"]
        for e in self.entries:
            lines.append(f"  v{e['version']}: mean_reward={e['mean_reward']:.2f} "
                          f"win_rate={e['win_rate']:.2f} note={e['note']}")
        return "\n".join(lines)
