
"""pitch_cmd isaret kalibrasyonu.

`_apply_action` icinde `elevator = -pitch_cmd * pitch_authority` esleme
var, ama JSBSim'de elevator komutunun BURNU YUKARI mi ASAGI mi cevirdigi
ucak modeline (ve F450 + ScasEngage kurulumuna) bagli. Bir multikopter
ILERLEMEK icin burnunu ASAGI egmek zorunda oldugundan, scripted rakip
kontrolculerin (HoverOpponent, ScriptedCircleOpponent,
KappaPursuitOpponent) dogru calismasi bu isarete baglidir.

Bu betik dronu havada sabitleyip SABIT pozitif bir pitch_cmd uygular ve
3 saniye sonra burun yonunde mi geriye mi gittigini olcer. Sonucu
`configs/*.yaml` icindeki `env.forward_pitch_sign` alanina yazin.

Kullanim:
    python main.py calibrate
"""

import math

import numpy as np
import jsbsim

from drone_rl.dogfight.config import DogfightEnvConfig, load_dogfight_config


def measure_forward_pitch_sign(cfg: DogfightEnvConfig, hold_s: float = 3.0,
                               verbose: bool = True) -> float:
    physics_dt = 1.0 / cfg.physics_hz
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    if not fdm.load_model("F450"):
        raise RuntimeError("F450 yuklenemedi")
    fdm.set_dt(physics_dt)

    fdm["ic/lat-gc-deg"] = 0.0
    fdm["ic/long-gc-deg"] = 0.0
    fdm["ic/h-agl-ft"] = cfg.base_altitude_ft
    fdm["ic/u-fps"] = 0.0
    fdm["ic/v-fps"] = 0.0
    fdm["ic/w-fps"] = 0.0
    fdm["ic/phi-rad"] = 0.0
    fdm["ic/theta-rad"] = 0.0
    fdm["ic/psi-true-rad"] = 0.0          # burun KUZEYE bakiyor
    fdm.run_ic()
    for i in range(4):
        fdm[f"propulsion/engine[{i}]/set-running"] = 1
    fdm["fcs/ScasEngage"] = 1
    fdm["fcs/aileron-cmd-norm"] = 0.0
    fdm["fcs/elevator-cmd-norm"] = 0.0
    fdm["fcs/rudder-cmd-norm"] = 0.0
    fdm["fcs/throttle-cmd-norm"] = cfg.hover_throttle

    surface = np.zeros(3, dtype=np.float64)
    alpha = physics_dt / (cfg.control_surface_tau_s + physics_dt)
    elevator_target = float(np.clip(-1.0 * cfg.pitch_authority, -1.0, 1.0))  # pitch_cmd = +1

    n_steps = int(hold_s * cfg.physics_hz)
    for _ in range(n_steps):
        surface += alpha * (np.array([0.0, elevator_target, 0.0]) - surface)
        fdm["fcs/aileron-cmd-norm"] = float(surface[0])
        fdm["fcs/elevator-cmd-norm"] = float(surface[1])
        fdm["fcs/rudder-cmd-norm"] = float(surface[2])
        fdm["fcs/throttle-cmd-norm"] = cfg.hover_throttle
        fdm.run()

    v_north = fdm["velocities/v-north-fps"]
    theta = fdm["attitude/theta-rad"]
    u_body = fdm["velocities/u-fps"]

    sign = 1.0 if v_north > 0.0 else -1.0

    if verbose:
        print("=" * 58)
        print("pitch_cmd = +1.0 uygulandiktan sonra (burun kuzeye bakiyordu):")
        print(f"  kuzey hizi      : {v_north:+.3f} ft/s")
        print(f"  govde ileri hizi: {u_body:+.3f} ft/s")
        print(f"  pitch acisi     : {math.degrees(theta):+.2f} deg")
        print("-" * 58)
        print(f"  ONERILEN DEGER  : env.forward_pitch_sign = {sign:+.1f}")
        if abs(v_north) < 0.5:
            print("  UYARI: hareket cok kucuk. pitch_authority/hover_throttle")
            print("         degerlerini kontrol edin veya hold_s'i artirin.")
        print("=" * 58)
    return sign


def main(config_path=None):
    cfg = load_dogfight_config(config_path).env
    return measure_forward_pitch_sign(cfg)


if __name__ == "__main__":
    main()
