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
# degistirebilirsin (0.1-0.5 arasi makul bir aralik).
PITCH_MAG = 0.30
ROLL_MAG = 0.30
ALT_MAG = 0.40


def manual_action_from_keys(keys: dict) -> np.ndarray:
    """WASD + ok tuslarini 4 motorun aksiyonuna (-1..1) cevirir.

    UYARI - motor mixing varsayimi: JSBSim'in F450 modelinde 4 motorun
    (fcs/throttle-cmd-norm[0..3]) hangi fiziksel koseye (on-sol, on-sag,
    arka-sol, arka-sag) karsilik geldigini buradan goremiyoruz - bu
    bilgi JSBSim model dosyasinin icinde tanimli, disaridan erisilemiyor.

    Asagidaki mixing YAYGIN bir X-quad konvansiyonu varsayiyor (motor
    sirasi: [on-sol, on-sag, arka-sol, arka-sag]). Test ettiginde "ileri"
    tusu geriye gidiyor ya da "sola" tusu saga donduruyorsa, PITCH_MAG
    veya ROLL_MAG'in isaretini (+/-) ters cevirmen yeterli - fiziksel
    bir hata degil, sadece varsayim yanlis yone denk gelmis demektir.
    """
    pitch = PITCH_MAG if keys.get("w") else (-PITCH_MAG if keys.get("s") else 0.0)
    roll = ROLL_MAG if keys.get("d") else (-ROLL_MAG if keys.get("a") else 0.0)
    throttle = ALT_MAG if keys.get("ArrowUp") else (-ALT_MAG if keys.get("ArrowDown") else 0.0)

    motor_fl = throttle + pitch + roll
    motor_fr = throttle + pitch - roll
    motor_rl = throttle - pitch + roll
    motor_rr = throttle - pitch - roll

    action = np.array([motor_fl, motor_fr, motor_rl, motor_rr], dtype=np.float32)
    return np.clip(action, -1.0, 1.0)


def any_key_pressed(keys: dict) -> bool:
    return any(keys.get(k) for k in ("w", "a", "s", "d", "ArrowUp", "ArrowDown"))


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


@app.websocket("/ws")
async def flight_loop(websocket: WebSocket):
    await websocket.accept()

    model = STATE["model"]
    stats = STATE["stats"]
    cfg = STATE["cfg"]

    env = make_flight_env(cfg.flight_env)
    obs, _ = env.reset()

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
                action = manual_action_from_keys(keys)
                mode = "manual"
            else:
                norm_obs = stats.normalize(obs).reshape(1, -1)
                action, _ = model.predict(norm_obs, deterministic=True)
                action = action[0]
                mode = "auto"

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reset = False
            if terminated or truncated:
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
