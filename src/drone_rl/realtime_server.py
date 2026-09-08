"""F450 flight - gercek zamanli (WASDEF ile oynanabilir) backend.

Colab icinde calisir: FastAPI + WebSocket ile her karede (frame) ortami
bir adim ilerletir, ya PPO'nun urettigi aksiyonu ya da WASDEF tuslarindan
gelen aksiyonu kullanir, sonucu tarayiciya gonderir.

Kontroller (tarayicida):
  W / S       -> ileri / geri (pitch_cmd)
  A / D       -> sola / saga (roll_cmd)
  E / F       -> yukselme / alcalma (throttle_cmd)
  Ok tuslari  -> SADECE kamera kadrajini oynatir, drone'a hicbir etkisi yok
  Hicbir tus basili degilse -> PPO otomatik ucusa devam eder

DUZELTME (v4): action artik motor-bazli bir mix DEGIL, dogrudan
F450FlightEnv.step()'in bekledigi [roll_cmd, pitch_cmd, yaw_cmd,
throttle_cmd] 4-vektoru (env dosyasindaki v4 aciklamasina bakin - bu
aircraft'in gercek kontrol arayuzu aileron/elevator/rudder/throttle
skalerleridir, motor-bazli degil). Onceki motor-mix yaklasimi (mix_motors,
MOTOR_MIX_MODE) TAMAMEN KALDIRILDI cunku o arayuz zaten fiziksel olarak
hicbir etki uretmiyordu.

DUZELTME (v4): irtifada "runtime PD-hold" hack'i KALDIRILDI. Salinim
sorunu artik PPO'nun kendisinin (yeni reward + DUZGUN CALISAN kontrol
arayuzuyle YENIDEN egitilerek) ogrenmesi gereken bir sey - sunucu
tarafinda ek bir zorlama/duzeltme YOK. PPO ne ogrendiyse oynatilan tam
olarak odur.

ONEMLI: Bu dosya, env'deki v4 kontrol-arayuzu duzeltmesiyle BIRLIKTE
kullanilmalidir. Eski (model_final.zip) model bu YENI arayuzle egitilmedi
- yeniden egitim yapmadan bu sunucuyu calistirirsaniz PPO'nun ciktilari
(artik gercekten calisan roll/pitch/throttle komutlarina donusecegi icin)
muhtemelen kontrolsuz/cakisma seklinde davranir. Once train.py ile
--task flight yeniden egitin.
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
# Bunlar fiziksel dogruluk degil, "oyun hissi" sabitleri. Env'in step()
# fonksiyonundaki action semantigiyle (roll_cmd, pitch_cmd, yaw_cmd,
# throttle_cmd, hepsi -1..1) DOGRUDAN uyumlu.
PITCH_MAG = 0.45   # W/S -> ileri/geri (pitch_cmd)
ROLL_MAG = 0.45    # A/D -> sola/saga (roll_cmd)
ALT_MAG = 0.65     # E/F -> yukselme/alcalma (throttle_cmd)

# Cevirirken (W/A/S/D) irtifayi sabit tutan geri besleme (feedback)
# katsayisi. Bu SADECE manuel ucus kullanilabilirligi icindir (drone
# yatinca toplam itkinin dikey bileseni dogal olarak azalir, bu da onu
# telafi eder) - PPO'nun otomatik ucusuna HICBIR sekilde karismaz, o
# tamamen ayri bir kod yolunda (asagida "auto" mode) calisir.
ALT_HOLD_KP = 0.08
ALT_HOLD_MAX = 0.5


class ManualController:
    """W/A/S/D = ileri/geri/sola/saga (pitch_cmd/roll_cmd), E/F =
    yukselme/alcalma (throttle_cmd). yaw_cmd her zaman 0 (klavyede
    kontrolu yok - kullanici istegi geregi).

    W/A/S/D basiliyken E/F basili DEGILSE, egilme (tilt) kaynakli dogal
    irtifa kaybini otomatik telafi ederek irtifayi kilitli tutar. E veya
    F basilinca kilit birakilir. Tum tuslar birakilinca (PPO'ya
    donulunce) kilit sifirlanir.
    """

    def __init__(self):
        self.alt_lock = None

    def compute_action(self, keys: dict, current_alt_ft: float) -> np.ndarray:
        pitch_cmd = PITCH_MAG if keys.get("w") else (-PITCH_MAG if keys.get("s") else 0.0)
        roll_cmd = ROLL_MAG if keys.get("d") else (-ROLL_MAG if keys.get("a") else 0.0)
        yaw_cmd = 0.0

        if keys.get("e") or keys.get("f"):
            self.alt_lock = None
            throttle_cmd = ALT_MAG if keys.get("e") else -ALT_MAG
        else:
            if self.alt_lock is None:
                self.alt_lock = current_alt_ft
            throttle_cmd = float(np.clip(
                ALT_HOLD_KP * (self.alt_lock - current_alt_ft), -ALT_HOLD_MAX, ALT_HOLD_MAX
            ))

        return np.array([roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd], dtype=np.float32)

    def reset_lock(self):
        self.alt_lock = None


def any_key_pressed(keys: dict) -> bool:
    # Ok tuslari burada YOK - onlar sadece kamera icin, drone kontrolune
    # hicbir sekilde dahil degiller.
    return any(keys.get(k) for k in ("w", "a", "s", "d", "e", "f"))


class NormalizerStats:
    """VecNormalize'in mean/var degerlerini tasiyan hafif bir tasiyici."""

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

    dummy_venv = make_flight_eval_vec_env(cfg.flight_env)
    vecnorm = VecNormalize.load(str(vecnorm_path), dummy_venv)
    stats = NormalizerStats(vecnorm)

    model = PPO.load(str(model_path), device="cpu")
    return model, stats, cfg


app = FastAPI()

STATE = {"model": None, "stats": None, "cfg": None, "html_path": None}


@app.get("/")
def index():
    return FileResponse(STATE["html_path"])


SUCCESS_LINGER_SECONDS = 3.0


@app.websocket("/ws")
async def flight_loop(websocket: WebSocket):
    await websocket.accept()

    model = STATE["model"]
    stats = STATE["stats"]
    cfg = STATE["cfg"]

    env = make_flight_env(cfg.flight_env)
    obs, _ = env.reset()

    linger_steps_total = max(1, int(SUCCESS_LINGER_SECONDS / env.control_dt))
    linger_remaining = None
    manual_ctrl = ManualController()

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
                # Tamamen PPO. Salinim/oturma davranisi TAMAMEN modelin
                # kendi ogrendigi seydir - sunucu tarafinda hicbir
                # duzeltme/zorlama uygulanmiyor.
                norm_obs = stats.normalize(obs).reshape(1, -1)
                action, _ = model.predict(norm_obs, deterministic=True)
                action = action[0]
                mode = "auto"

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reset = False

            if linger_remaining is not None:
                linger_remaining -= 1
                if info.get("crashed") or linger_remaining <= 0:
                    obs, _ = env.reset()
                    episode_reset = True
                    linger_remaining = None
            elif terminated or truncated:
                if info.get("reached_target") and not info.get("crashed"):
                    linger_remaining = linger_steps_total
                else:
                    obs, _ = env.reset()
                    episode_reset = True

            payload = dict(info)
            payload["mode"] = mode
            payload["episode_reset"] = episode_reset
            payload["target_altitude_ft"] = env.target_altitude
            payload["control_dt"] = env.control_dt
            # motor_throttle artik env._get_telemetry() icinde, fdm'den
            # OKUNAN gercek fcs/throttle-pos-norm[i] degerleri (bkz. v4
            # notu) - burada ayrica hesaplanmiyor.

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
