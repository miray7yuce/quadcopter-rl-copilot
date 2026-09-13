"""Self-play icin checkpoint havuzu - PFSP-lite.

Korunan davranislar:
  * Havuz KENDI KENDINI DOGRULAR: manifest'te olup diskte dosyasi
    eksik olan entry'ler otomatik temizlenir (yarida kalmis bir
    promotion egitimi cokertmesin diye).
  * Versiyon numarasi "son entry + 1" (prune sonrasi cakismayi onler).
  * sample()/latest()/add() manifest'i DISKTEN TAZE okur - egitim
    surecinin ayri bir CheckpointPool nesnesiyle yaptigi promotion'lari
    ortamlar hemen gorur (stale opponent problemi).

v8'de eklenen (T2 - sadelestirilmis):
  * PFSP-lite ornekleme. Eskiden: %70 en son, %30 eskiler arasinda
    UNIFORM. Uniform ornekleme, cok eski/zayif politikalarin surekli
    secilmesine ve egitimin bosa harcanmasina yol aciyordu (bkz.
    aerospace-12-00265, Bolum 3.1 - "obsolete strategies are more
    likely to be sampled, potentially degrading RL performance").
    Simdi eskiler arasinda secim, kayitli win_rate uzerinden softmax
    agirligiyla yapiliyor: guclu checkpoint'ler daha sik secilir.
    Tam Elo + SA-Boltzmann meta-solver yerine, ayni etkiyi veren
    birkac satirlik sade bir surum.
"""

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
        self._prune_missing()

    # ------------------------------------------------------------------
    def _load(self):
        if self.manifest_path.exists():
            try:
                self.entries = json.loads(self.manifest_path.read_text())
            except json.JSONDecodeError:
                # baska bir surec tam o anda yaziyor olabilir; eldekini koru
                self.entries = getattr(self, "entries", [])
        else:
            self.entries = []

    def _save(self):
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.entries, indent=2))
        tmp.replace(self.manifest_path)

    def _prune_missing(self):
        valid, removed = [], []
        for e in self.entries:
            if Path(e["model"]).exists() and Path(e["vecnorm"]).exists():
                valid.append(e)
            else:
                removed.append(e["version"])
        if removed:
            print(f"[CheckpointPool] UYARI: diskte dosyasi eksik oldugu icin "
                  f"manifest'ten cikarilan versiyonlar: {removed}")
            self.entries = valid
            self._save()

    def __len__(self):
        return len(self.entries)

    # ------------------------------------------------------------------
    def add(self, model_src: str, vecnorm_src: str, mean_reward: float,
            win_rate: float, note: str = "", obs_dim: Optional[int] = None) -> int:
        self._load()
        next_version = (self.entries[-1]["version"] + 1) if self.entries else 1
        dst_dir = self.pool_dir / f"v{next_version}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        model_dst = dst_dir / "model.zip"
        vecnorm_dst = dst_dir / "vecnormalize.pkl"

        shutil.copy(model_src, model_dst)
        shutil.copy(vecnorm_src, vecnorm_dst)

        entry = {
            "version": next_version,
            "model": str(model_dst),
            "vecnorm": str(vecnorm_dst),
            "mean_reward": float(mean_reward),
            "win_rate": float(win_rate),
            "note": note,
        }
        if obs_dim is not None:
            entry["obs_dim"] = int(obs_dim)
        self.entries.append(entry)
        self._save()
        return next_version

    def latest(self) -> Optional[Tuple[str, str]]:
        self._load()
        if not self.entries:
            return None
        e = self.entries[-1]
        return e["model"], e["vecnorm"]

    def latest_mean_reward(self) -> Optional[float]:
        self._load()
        if not self.entries:
            return None
        return self.entries[-1]["mean_reward"]

    def latest_version(self) -> Optional[int]:
        self._load()
        if not self.entries:
            return None
        return self.entries[-1]["version"]

    # ------------------------------------------------------------------
    def _pfsp_weights(self, entries, temperature: float) -> np.ndarray:
        """win_rate uzerinden softmax. Dusuk sicaklik = guclu
        checkpoint'lere daha cok agirlik; yuksek sicaklik = uniform."""
        wr = np.array([float(e.get("win_rate", 0.5)) for e in entries], dtype=np.float64)
        t = max(float(temperature), 1e-3)
        logits = (wr - wr.max()) / t
        w = np.exp(logits)
        total = w.sum()
        if not np.isfinite(total) or total <= 0:
            return np.full(len(entries), 1.0 / len(entries))
        return w / total

    def sample(self, latest_prob: float = 0.6,
               rng: Optional[np.random.Generator] = None,
               temperature: float = 0.25) -> Optional[Tuple[str, str]]:
        self._load()
        if not self.entries:
            return None
        rng = rng or np.random.default_rng()

        if len(self.entries) == 1 or rng.random() < latest_prob:
            e = self.entries[-1]
        else:
            older = self.entries[:-1]
            probs = self._pfsp_weights(older, temperature)
            idx = int(rng.choice(len(older), p=probs))
            e = older[idx]

        if not (Path(e["model"]).exists() and Path(e["vecnorm"]).exists()):
            print(f"[CheckpointPool] UYARI: v{e['version']} diskte bulunamadi, "
                  f"havuz yeniden dogrulaniyor.")
            self._load()
            self._prune_missing()
            if not self.entries:
                return None
            e = self.entries[-1]

        return e["model"], e["vecnorm"]

    def summary(self) -> str:
        self._load()
        lines = [f"Pool: {self.pool_dir} ({len(self.entries)} versiyon)"]
        for e in self.entries:
            lines.append(f"  v{e['version']}: mean_reward={e['mean_reward']:.2f} "
                         f"win_rate={e['win_rate']:.2f} obs_dim={e.get('obs_dim', '?')} "
                         f"note={e['note']}")
        return "\n".join(lines)
