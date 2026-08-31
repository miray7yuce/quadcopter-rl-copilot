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

Degerlendirme (egitim sonundaki modelle, model_final.zip + vecnormalize.pkl):

    cd src && python -m drone_rl.evaluate --run ../runs/hover_v2 --output /content/telemetry.csv

Degerlendirme (en iyi modelle, best_model.zip + vecnormalize_best.pkl):

    cd src && python -m drone_rl.evaluate --run ../runs/hover_v2 --output /content/telemetry.csv --use-best

## Yapi

- src/drone_rl/envs/f450_env.py - Gymnasium ortami
- src/drone_rl/train.py - PPO egitimi (EvalCallback ile en iyi model + VecNormalize kaydi)
- src/drone_rl/evaluate.py - egitilmis politikanin olculmesi ve CSV telemetri ciktisi
- configs/ppo_hover.yaml - kullanilan ayarlar (NOT: su an kod tarafindan okunmuyor, sadece referans)
- runs/hover_v1/ - egitilmis model ve normalizasyon istatistikleri
- notebooks/quadcopter_rl.ipynb - Colab calisma defteri

## Notlar

- JSBSim emperyal birim kullanir (ft, lbs, fps).
- Hover gazi f450_env.py icinde HOVER_THROTTLE olarak sabitlenmistir.
  configs/ppo_hover.yaml'daki hover_throttle degeri su an sadece
  dokumantasyon amaclidir, kod tarafindan okunmaz.
- vecnormalize.pkl (veya vecnormalize_best.pkl) model ile birlikte
  yuklenmelidir, aksi halde politika yanlis olcekli gozlem alir ve
  calismaz.
- F450 XML'i yuklenirken "version 3.0" uyarisi verir; zararsizdir.
- evaluate.py su an gercek bir ACMI (.acmi / Tacview) dosyasi degil,
  duz bir CSV telemetri dosyasi uretir.
