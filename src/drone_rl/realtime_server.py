"""F450 flight - gercek zamanli (WASD + ok tuslariyla oynanabilir) backend.

Colab icinde calisir: FastAPI + WebSocket ile her karede (frame) ortami
bir adim ilerletir, ya PPO'nun urettigi aksiyonu ya da WASD/ok
tuslarindan gelen aksiyonu kullanir, sonucu tarayiciya gonderir.

Kontroller (tarayicida):
  W / S       -> ileri / geri (pitch)
  A / D       -> sola / saga (roll)
  Yukari Ok   -> yukselme (throttle+)
  Asagi Ok    -> alcalma (throttle-)
  Hicbir tus basili degilse -> PPO otomatik ucusa devam eder

Retraining GEREKMIYOR - ayni egitilmis model (model_final.zip +
vecnormalize.pkl) burada da kullaniliyor, sadece nerede/nasil
calistirdigimiz degisiyor.
"""

import asyncio
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from drone_rl.config import load_config
from drone_rl.env_factory import make_flight_env, make_flight_eval_vec_env
from drone_rl.evaluate import resolve_model_paths


# --- Manuel kontrol hissi icin ayarlanabilir sabitler ---
# Bunlar fiziksel dogruluk degil, "oyun hissi" sabitleri - istedigin gibi
# degistirebilirsin. Onceki degerlere (0.30/0.30/0.40) gore buyutuldu,
# cunku kucuk degerler quadin gercek kutlesi/ataleti yuzunden cok yavas
# hizlanip yavasliyordu - daha buyuk degerler daha "gercek zamanli" hissettirir.
PITCH_MAG = 0.55   # W/S -> ileri/geri
ROLL_MAG = 0.55    # A/D -> sola/saga
ALT_MAG = 0.65     # E/F -> yukselme/alcalma

# Cevirirken (W/A/S/D) irtifayi sabit tutan geri besleme (feedback)
# katsayisi. Drone yana/one yatinca toplam itki artik tam dikey olmadigi
# icin dogal olarak alcalir (gercek quadcopterlarda da boyledir) - bu
# katsayi o kaybi telafi ediyor.
ALT_HOLD_KP = 0.08
ALT_HOLD_MAX = 0.5


class ManualController:
    """W/A/S/D = ileri/geri/sola/saga (pitch/roll ile), E/F = yukselme/alcalma.

    W/A/S/D basiliyken E/F basili DEGILSE, egilme (tilt) kaynakli dogal
    irtifa kaybini otomatik telafi ederek irtifayi kilitli tutar - yoksa
    "A/D irtifayi da degistiriyor" gibi kafa karistirici bir yan etki
    olur (bu FIZIKSEL bir etki, motor mixing hatasi degil: drone
    yatinca toplam itkinin dikey bilesimi azalir).

    E veya F basilinca kilit birakilir, dogrudan tam guclu yukselme/
    alcalma komutu verilir. Tum tuslar birakilinca (PPO otomatik pilota
    donulunce) kilit sifirlanir - manuel kontrole tekrar girildiginde
    O ANKI irtifadan yeniden kilitlenir, eski/bayat bir degerden degil.
    """

    def __init__(self):
        self.alt_lock = None

    def compute_action(self, keys: dict, current_alt_ft: float) -> np.ndarray:
        pitch = PITCH_MAG if keys.get("w") else (-PITCH_MAG if keys.get("s") else 0.0)
        roll = ROLL_MAG if keys.get("d") else (-ROLL_MAG if keys.get("a") else 0.0)

        if keys.get("e") or keys.get("f"):
            self.alt_lock = None
            throttle = ALT_MAG if keys.get("e") else -ALT_MAG
        else:
            if self.alt_lock is None:
                self.alt_lock = current_alt_ft
            throttle = float(np.clip(
                ALT_HOLD_KP * (self.alt_lock - current_alt_ft), -ALT_HOLD_MAX, ALT_HOLD_MAX
            ))

        motor_fl = throttle + pitch + roll
        motor_fr = throttle + pitch - roll
        motor_rl = throttle - pitch + roll
        motor_rr = throttle - pitch - roll

        action = np.array([motor_fl, motor_fr, motor_rl, motor_rr], dtype=np.float32)
        return np.clip(action, -1.0, 1.0)

    def reset_lock(self):
        self.alt_lock = None


def any_key_pressed(keys: dict) -> bool:
    # DIKKAT: ok tuslari artik burada YOK - onlar sadece kamera icin,
    # drone kontrolune dahil degiller (E/F irtifa icin kullaniliyor).
    return any(keys.get(k) for k in ("w", "a", "s", "d", "e", "f"))


class NormalizerStats:
    """VecNormalize'in mean/var degerlerini tasiyan hafif bir tasiyici.

    Egitimde kullanilan normalize etme formulunu (obs -> normalized obs),
    tek bir canli gozlem uzerinde MANUEL olarak uygulayabilmek icin -
    canli dongude gercek bir VecEnv/VecNormalize wrapper'i kullanmiyoruz
    (tek ortam, tek adim, sürekli acik kalan bir dongu oldugu icin daha
    basit). Bu formul VecNormalize.normalize_obs() ile birebir ayni.
    """

    def __init__(self, vecnorm: VecNormalize):
        self.mean = vecnorm.obs_rms.mean.astype(np.float32)
        self.var = vecnorm.obs_rms.var.astype(np.float32)
        self.epsilon = vecnorm.epsilon
        self.clip_obs = vecnorm.clip_obs

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        normed = (obs - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normed, -self.clip_obs, self.clip_obs).astype(np.float32)


def load_policy(run: str, config: str, use_best: bool = False):
    cfg = load_config(config)
    run_path = Path(run)
    model_path, vecnorm_path = resolve_model_paths(run_path, use_best)

    if not vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize dosyasi bulunamadi: {vecnorm_path}")

    # Gercek bir egitim/eval dongusu kurmuyoruz - sadece VecNormalize'in
    # ogrendigi mean/var istatistiklerini disari cikarmak icin gecici
    # (dummy) bir vec-env uzerinden yukluyoruz.
    dummy_venv = make_flight_eval_vec_env(cfg.flight_env)
    vecnorm = VecNormalize.load(str(vecnorm_path), dummy_venv)
    stats = NormalizerStats(vecnorm)

    model = PPO.load(str(model_path), device="cpu")
    return model, stats, cfg


app = FastAPI()

# start_server() cagrildiginda doldurulur; /ws endpoint'i buradan okur.
STATE = {"model": None, "stats": None, "cfg": None, "html_path": None}


@app.get("/")
def index():
    return FileResponse(STATE["html_path"])


# Basariyla hedefe ulasildiktan sonra, hemen resetlemeden once ekranda
# ne kadar sure daha (saniye) izleyelim - saf gorsellestirme amacli,
# env'in kendi terminated=True mantigini (egitimde kullanilan) DEGISTIRMIYORUZ,
# sadece sunucu tarafinda "reset()'i cagirmayi geciktiriyoruz".
SUCCESS_LINGER_SECONDS = 3.0


@app.websocket("/ws")
async def flight_loop(websocket: WebSocket):
    await websocket.accept()

    model = STATE["model"]
    stats = STATE["stats"]
    cfg = STATE["cfg"]

    env = make_flight_env(cfg.flight_env)
    obs, _ = env.reset()

    # None: normal ucus. Sayi: "basariyla ulasti, su kadar adim sonra
    # resetlenecek" geri sayimi. Bu sayede terminated=True dondugu anda
    # DEGIL, SUCCESS_LINGER_SECONDS kadar sonra reset() cagriliyor -
    # boylece "TARGET REACHED" durumunu ekranda birkac saniye gorebiliyoruz.
    linger_steps_total = max(1, int(SUCCESS_LINGER_SECONDS / env.control_dt))
    linger_remaining = None
    manual_ctrl = ManualController()

    # WASD durumu ayri bir "receiver" task'inde tutuluyor, boylece ana
    # fizik dongusu tus mesaji beklemek zorunda kalmadan kendi hizinda
    # (control_dt) akmaya devam edebiliyor. Poll+timeout yontemi yerine
    # bu, hem daha az CPU harcar hem tus olaylarini kacirmaz.
    current_keys = {}

    async def receiver():
        nonlocal current_keys
        try:
            while True:
                raw = await websocket.receive_text()
                current_keys = json.loads(raw) if raw else {}
        except WebSocketDisconnect:
            pass

    receiver_task = asyncio.create_task(receiver())

    try:
        while True:
            keys = current_keys

            if any_key_pressed(keys):
                current_alt_ft = env.fdm["position/h-agl-ft"]
                action = manual_ctrl.compute_action(keys, current_alt_ft)
                mode = "manual"
            else:
                manual_ctrl.reset_lock()
                norm_obs = stats.normalize(obs).reshape(1, -1)
                action, _ = model.predict(norm_obs, deterministic=True)
                action = action[0]
                mode = "auto"

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reset = False

            if linger_remaining is not None:
                # Basari sonrasi "bekleme" penceresindeyiz. Cakisma olursa
                # beklemeden hemen resetle; yoksa geri sayimi azalt.
                linger_remaining -= 1
                if info.get("crashed") or linger_remaining <= 0:
                    obs, _ = env.reset()
                    episode_reset = True
                    linger_remaining = None
                # yoksa: reset ETME, env'in terminated=True demesine ragmen
                # step() atmaya devam ediyoruz - JSBSim'in fizigi bunu
                # umursamiyor, sadece bizim reset() cagirip cagirmamamiz
                # onemli.
            elif terminated or truncated:
                if info.get("reached_target") and not info.get("crashed"):
                    # Basariyla ulasti - hemen resetlemek yerine bekleme
                    # geri sayimini baslat.
                    linger_remaining = linger_steps_total
                else:
                    obs, _ = env.reset()
                    episode_reset = True

            payload = dict(info)
            payload["mode"] = mode
            payload["episode_reset"] = episode_reset
            payload["target_altitude_ft"] = env.target_altitude
            payload["control_dt"] = env.control_dt
            payload["motor_throttle"] = np.clip(
                env.hover_throttle + action * env.throttle_range, 0.0, 1.0
            ).tolist()

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(env.control_dt)

    except WebSocketDisconnect:
        pass
    finally:
        receiver_task.cancel()


def start_server(run: str, config: str, html_path: str, use_best: bool = False, port: int = 8000):
    """Colab hucresinden cagrilacak baslatma fonksiyonu.

    Ornek kullanim (Colab hucresi):
        from drone_rl.realtime_server import start_server
        start_server(
            run="/content/runs/flight_ppo_v4",
            config="/content/repo/configs/ppo_flight.yaml",
            html_path="/content/droneSim_realtime.html",
        )
        from google.colab.output import serve_kernel_port_as_window
        serve_kernel_port_as_window(8000)
    """
    import threading
    import uvicorn

    model, stats, cfg = load_policy(run, config, use_best)
    STATE["model"] = model
    STATE["stats"] = stats
    STATE["cfg"] = cfg
    STATE["html_path"] = html_path

    thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning"),
        daemon=True,
    )
    thread.start()
    print(f"Sunucu baslatildi (arka planda, port {port}).")
    print("Simdi asagidaki hucreyi calistirip acilan pencereye/linke tikla.")
