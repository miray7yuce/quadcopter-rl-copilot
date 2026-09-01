# quadcopter-rl-copilot

JSBSim F450 quadcopter modeli uzerinde PPO ile hover kontrolu.

## Sonuc (hover_v1)

300.000 adim egitim sonrasi, 5 degerlendirme episode'unda:

- Episode uzunlugu: 400/400 (hic dusme yok)
- Hedef irtifadan ortalama sapma: 0.02 - 0.06 ft
- Ortalama egilme: 0.02 - 0.08 rad

## Kurulum (Colab)

    !pip install -q stable-baselines3 gymnasium
    !pip uninstall -y -q jsbsim
    !pip install -q jsbsim==1.2.4

Repoyu klonladiktan sonra src dizinini Python yoluna ekle:

    import os, sys
    sys.path.insert(0, "/content/repo/src")
    os.environ["PYTHONPATH"] = "/content/repo/src"

## Kullanim

Egitim:

    cd src && python -m drone_rl.train --timesteps 300000 --n-envs 4 --out ../runs/hover_v2

Degerlendirme:

    cd src && python -m drone_rl.evaluate --run ../runs/hover_v1 --csv /content/iz.csv

## Yapi

- src/drone_rl/envs/f450_env.py - Gymnasium ortami
- src/drone_rl/train.py - PPO egitimi
- src/drone_rl/evaluate.py - egitilmis politikanin olculmesi
- configs/ppo_hover.yaml - kullanilan ayarlar
- runs/hover_v1/ - egitilmis model ve normalizasyon istatistikleri
- notebooks/quadcopter_rl.ipynb - Colab calisma defteri

## Notlar

- JSBSim emperyal birim kullanir (ft, lbs, fps).
- Hover gazi 0.410 olarak olculdu. Aksiyon bu deger etrafinda +-0.25
  araliginda olceklenir, boylece sifir aksiyon "asili kal" anlamina gelir.
- vecnormalize.pkl model ile birlikte yuklenmelidir, aksi halde politika
  yanlis olcekli gozlem alir ve calismaz.
- F450 XML'i yuklenirken "version 3.0" uyarisi verir; zararsizdir.
