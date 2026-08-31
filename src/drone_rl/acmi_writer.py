"""Tacview ACMI (.acmi) format yazici - basit tek-obje coklu-episode kaydi.

ACMI format referansi: https://www.tacview.net/documentation/acmi/en/

Bu writer sadece bu proje icin gerekli minimum alt kumeyi destekler:
tek bir hava araci objesi, zaman serisi konum/aci guncellemeleri.
Coklu obje, olay (event) kayitlari, veya ek property'ler (hiz, RPM vb.)
desteklenmiyor - ihtiyac olursa genisletilebilir.

Beklenen birimler (ACMI standardi):
- lon_deg, lat_deg : derece (WGS84)
- alt_m            : metre (deniz seviyesinden veya yerden - tutarli olmasi yeterli)
- roll_deg, pitch_deg, yaw_deg : derece

JSBSim ft ve radyan kullandigi icin cagiran kod bu donusumu
(units.ft_to_m, np.degrees) yapmali; bu siniftan once cagirilmalidir.
"""

from pathlib import Path
from datetime import datetime, timezone


class ACMIWriter:
    def __init__(self, object_id=1, name="F450", obj_type="Air+Rotorcraft+UAV", color="Blue"):
        self.object_id = object_id
        self.name = name
        self.obj_type = obj_type
        self.color = color
        self._lines = []
        self._header_written = False
        self._object_declared = False
        self._last_frame_time = None

    def _write_header(self):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._lines.append("FileType=text/acmi/tacview")
        self._lines.append("FileVersion=2.2")
        self._lines.append(f"0,ReferenceTime={now}")
        self._header_written = True

    def add_frame(self, t, lon_deg, lat_deg, alt_m, roll_deg, pitch_deg, yaw_deg):
        """Bir zaman anindaki obje durumunu ekler.

        t: saniye cinsinden, dosya boyunca MONOTONIK ARTAN olmali
           (birden fazla episode varsa t'yi sifirlama, devam ettir).
        """
        if not self._header_written:
            self._write_header()

        # Ayni t icin tekrar frame acmayalim (float hassasiyeti icin yuvarla)
        t_rounded = round(t, 2)
        if self._last_frame_time != t_rounded:
            self._lines.append(f"#{t_rounded:.2f}")
            self._last_frame_time = t_rounded

        obj_line = (
            f"{self.object_id:x},T={lon_deg:.7f}|{lat_deg:.7f}|{alt_m:.2f}|"
            f"{roll_deg:.2f}|{pitch_deg:.2f}|{yaw_deg:.2f}"
        )
        if not self._object_declared:
            obj_line += f",Name={self.name},Type={self.obj_type},Color={self.color}"
            self._object_declared = True

        self._lines.append(obj_line)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._lines) + "\n")
        return path
