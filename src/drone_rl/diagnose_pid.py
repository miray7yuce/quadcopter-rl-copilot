"""TANI SCRIPTI - RL'e hic gerek olmadan, env'in FIZIKSEL OLARAK
ogrenilebilir (flyable) olup olmadigini kontrol eder.

Amac: "Sorun RL'in ogrenememesi mi, yoksa env/fizikte hala bir sikinti
mi?" sorusunu KESIN olarak ayirt etmek. Basit bir PID otopilot ile
(sadece irtifa tutmaya calisiyor, heading/yon takibi YOK, sadece duz
tutup irtifaya cikip orada kalmaya calisiyor) ayni env.step() arayuzunu
kullanarak N episode ucuruyoruz.

Yorumlama:
- PID BASARILI (crash orani dusuk, irtifada duzgunce oturuyor)
  -> env/fizik SAGLIKLI. Sorun kesinlikle RL egitiminde (curriculum,
     adim sayisi, hiperparametre) - env'e DOKUNMAYA GEREK YOK.
- PID DE SIK CRASH OLUYOR
  -> env'de hala fiziksel bir sorun var (authority cok kisik, crash
     sinirlari cok siki, kontrol gecikmesi cok yuksek vs.) - RL'i
     suclamadan ONCE env'i duzeltmeliyiz.

Kullanim (Colab hucresi):
    import sys
    sys.path.insert(0, "/content/repo/src")
    from drone_rl.diagnose_pid import run_diagnosis
    run_diagnosis(config="/content/repo/configs/ppo_flight.yaml", n_episodes=30)
"""

import numpy as np

from drone_rl.config import load_config
from drone_rl.env_factory import make_flight_env


class AltitudeHoldPID:
    """SADECE irtifa tutmaya calisan, KASITLI OLARAK basit bir otopilot.
    Heading/yon takibi YOK (roll_cmd=0 sabit) - amac, gorevin EN TEMEL
    alt-katmanini (tirman + irtifada otur) izole ederek test etmek."""

    def __init__(self, kp=0.14, kd=0.35, ki=0.01):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def compute_action(self, alt_err_ft, hdot_fps, r_now):
        # alt_err_ft = target - current (pozitif ise TIRMANMALI)
        self.integral = np.clip(self.integral + alt_err_ft * 0.05, -20, 20)
        throttle_cmd = self.kp * alt_err_ft - self.kd * hdot_fps + self.ki * self.integral
        throttle_cmd = float(np.clip(throttle_cmd, -1.0, 1.0))

        # Yaw'da hicbir hedef yok - sadece kalinti spin'i sondur (server'daki
        # manuel kontrolcuyle AYNI mantik).
        yaw_cmd = float(np.clip(-0.9 * r_now, -1.0, 1.0))

        # Roll/pitch: HEDEF YOK, duz/level ucusu koru (0,0).
        return np.array([0.0, 0.0, yaw_cmd, throttle_cmd], dtype=np.float32)


def run_diagnosis(config: str, n_episodes: int = 30, verbose: bool = True):
    cfg = load_config(config)
    env = make_flight_env(cfg.flight_env)
    pid = AltitudeHoldPID()

    crashes = 0
    successes = 0
    timeouts = 0
    alt_err_at_end = []
    max_alt_err_after_settle = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        pid.reset()
        done = False
        settled = False
        post_settle_errors = []

        while not done:
            f = env.fdm
            alt_err = env.target_altitude - f["position/h-agl-ft"]
            hdot = f["velocities/h-dot-fps"]
            r_now = f["velocities/r-rad_sec"]

            action = pid.compute_action(alt_err, hdot, r_now)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if info["alt_err_ft"] < cfg.flight_env.success_alt_tol_ft * 1.5:
                settled = True
            if settled:
                post_settle_errors.append(info["alt_err_ft"])

            if info["crashed"]:
                crashes += 1
            elif info["reached_target"]:
                successes += 1
            elif truncated:
                timeouts += 1

        alt_err_at_end.append(info["alt_err_ft"])
        if post_settle_errors:
            max_alt_err_after_settle.append(max(post_settle_errors))

        if verbose:
            status = "CRASH" if info["crashed"] else ("SUCCESS" if info["reached_target"] else "TIMEOUT")
            print(f"  ep {ep+1:02d}/{n_episodes}: {status:8s} "
                  f"final_alt_err={info['alt_err_ft']:.2f}ft")

    print("\n" + "=" * 50)
    print(f"TANI SONUCU ({n_episodes} episode, sadece irtifa-tutma PID, heading YOK)")
    print("=" * 50)
    print(f"Crash orani     : {crashes}/{n_episodes} (%{100*crashes/n_episodes:.0f})")
    print(f"Basari orani    : {successes}/{n_episodes} (%{100*successes/n_episodes:.0f})")
    print(f"Timeout orani   : {timeouts}/{n_episodes} (%{100*timeouts/n_episodes:.0f})")
    if max_alt_err_after_settle:
        print(f"Oturduktan sonraki max sapma (ort): {np.mean(max_alt_err_after_settle):.2f} ft")
    print()
    if crashes / n_episodes < 0.15:
        print(">> ENV/FIZIK SAGLIKLI GORUNUYOR. Sorun RL egitiminde (curriculum/")
        print(">> adim sayisi/hiperparametre) - env'e dokunmadan RL stratejisini")
        print(">> degistirmeliyiz (asagidaki curriculum onerisine bakin).")
    else:
        print(">> BASIT PID BILE SIK CRASH OLUYOR - env'de hala fiziksel bir sorun")
        print(">> var (authority/crash siniri/kontrol gecikmesi). RL'i suclamadan")
        print(">> once env'i incelemeliyiz - bu sonucu bana gonderin.")

    return {
        "crash_rate": crashes / n_episodes,
        "success_rate": successes / n_episodes,
        "timeout_rate": timeouts / n_episodes,
    }
