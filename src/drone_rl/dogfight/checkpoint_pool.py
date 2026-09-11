"""Self-play icin checkpoint havuzu - PFSP-lite.

DUZELTME: Havuz artik KENDI KENDINI DOGRULUYOR. Onceki calisma
sirasinda (kernel kesintisi, bilgisayar degisimi vb.) bir promotion'in
dosya kopyalama adimi yarida kalmis olabilir - manifest.json bir
versiyonu (orn. v2) listeler ama gercek model.zip/vecnormalize.pkl
dosyalari diskte YOK. Bu, sample()/latest() cagrildiginda
FileNotFoundError ile egitimi cokertiyordu.

Simdi: yukleme sirasinda HER entry'nin dosyalarinin GERCEKTEN var olup
olmadigi kontrol ediliyor - eksik olanlar manifest'ten SESSIZCE
(sadece bir uyari basarak) CIKARILIYOR ve manifest.json GUNCELLENIYOR.
Boylece bir daha ayni bozuk versiyona rastlanmiyor.
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

    def _load(self):
        if self.manifest_path.exists():
            self.entries = json.loads(self.manifest_path.read_text())
        else:
            self.entries = []

    def _save(self):
        self.manifest_path.write_text(json.dumps(self.entries, indent=2))

    def _prune_missing(self):
        """YENI: manifest'te olup diskte dosyalari EKSIK olan entry'leri
        temizler - kesintiye ugramis bir promotion'dan kalma bozuk
        kayitlarin egitimi cokertmesini onler."""
        valid = []
        removed = []
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

    def add(self, model_src: str, vecnorm_src: str, mean_reward: float,
            win_rate: float, note: str = "") -> int:
        # YENI: versiyon numarasi artik "son entry'nin numarasi + 1" -
        # eskiden len(entries)+1 idi, ama bir versiyon prune edilmisse
        # (silinmisse) bu iki numaranin CAKISMASINA yol acabiliyordu.
        next_version = (self.entries[-1]["version"] + 1) if self.entries else 1
        dst_dir = self.pool_dir / f"v{next_version}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        model_dst = dst_dir / "model.zip"
        vecnorm_dst = dst_dir / "vecnormalize.pkl"

        # YENI: once GECICI bir isimle kopyala, ikisi de TAM bitince
        # manifest'e ekle - kesinti olursa yarim kalan dosya asla
        # manifest'e girmez (bir sonraki _prune_missing zaten temizler
        # ama bunu bastan onlemek daha saglam).
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
        self.entries.append(entry)
        self._save()
        return next_version

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

        # YENI: sample ANINDA da guvenlik kontrolu - manifest.json disinda
        # (orn. baska bir surecin ayni anda yazdigi) bir bozulma olursa
        # bile egitim COKMEZ, en guncel/saglam entry'e duser.
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
        lines = [f"Pool: {self.pool_dir} ({len(self.entries)} versiyon)"]
        for e in self.entries:
            lines.append(f"  v{e['version']}: mean_reward={e['mean_reward']:.2f} "
                          f"win_rate={e['win_rate']:.2f} note={e['note']}")
        return "\n".join(lines)



