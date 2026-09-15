"""Dogfight egitim betigi - v8.

Degisiklikler:
  * StandoffCurriculumCallback -> ProgressCallback: hem shaped odul
    agirligini sonumler (AOS makalesindeki lambda_r decay) hem de rakip
    mufredatinin ilerlemesini ortamlara bildirir.
  * DiagnosticsCallback: TensorBoard'a ATA, menzil, closing, koni
    yakalama orani, HP ve reset sebebi dagilimi yazar. Bu metrikler
    olmadan "neden takip etmiyor" sorusunu olcerek cevaplayamiyorduk.
  * Lineer ogrenme hizi sonumlemesi, target_kl, max_grad_norm, vf_coef.
  * Kazanma olcutu artik koni adim sayisi degil, HP farki/dusurme.
  * SubprocVecEnv secenegi (--vec).
  * seed-pool artik --config'i dikkate aliyor (eskiden sessizce
    varsayilan config ile degerlendiriyordu).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from drone_rl.dogfight.config import load_dogfight_config
from drone_rl.dogfight.env_factory import (
    make_dogfight_training_vec_env, make_dogfight_env, make_dummy_vecnorm_env,
)
from drone_rl.dogfight.checkpoint_pool import CheckpointPool
from drone_rl.dogfight.dogfight_env import NormalizerStats, OBS_DIM


ACTIVATION_MAP = {"tanh": nn.Tanh, "relu": nn.ReLU}


def build_policy_kwargs(cfg_ppo):
    kwargs = {}
    pi_arch = cfg_ppo.net_arch_pi or [256, 256]
    vf_arch = cfg_ppo.net_arch_vf or [256, 256]
    kwargs["net_arch"] = dict(pi=pi_arch, vf=vf_arch)
    if cfg_ppo.activation_fn:
        kwargs["activation_fn"] = ACTIVATION_MAP[cfg_ppo.activation_fn.lower()]
    return kwargs


def linear_schedule(initial: float, final_frac: float):
    """SB3 progress_remaining 1 -> 0 gider."""
    final = initial * float(final_frac)

    def _f(progress_remaining: float) -> float:
        return final + (initial - final) * progress_remaining

    return _f


def _model_has_nan(model) -> bool:
    """YENI: --resume ile yuklenen bir checkpoint'in agirliklarinda
    NaN/Inf olup olmadigini kontrol eder. Bir NaN cokusu SONRASI
    kaydedilmis (veya cokus TAM kaydetme anina denk gelmis) bir
    live_snapshot, temiz gozlemlerle bile SESSIZCE bozuk kalmaya
    devam eder - ag'in KENDISI zehirlenmis olur. Bu kontrol olmadan
    --resume, corken checkpoint'i sessizce yukleyip AYNI hatayla
    tekrar cokerdi (ilk cokus rollout toplarken, ikincisi egitim
    adimi sirasinda - farkli yerlerde ama AYNI kok neden)."""
    import torch
    for p in model.policy.parameters():
        if not torch.isfinite(p).all():
            return True
    return False


class ProgressCallback(BaseCallback):
    """Shaped odul agirligi + rakip mufredati ilerlemesi.

    Shaped (yogun) odul basta guclu olmali ki ajan hic sinyal almadan
    kaybolmasin; ilerledikce sonumlenmeli ki terminal (kazanma) sinyali
    baskin hale gelsin ve odul bicimlendirmesi politikayi carpitmasin."""

    def __init__(self, env_cfg, update_every: int = 200):
        super().__init__()
        self.env_cfg = env_cfg
        self.update_every = max(update_every, 1)
        self._last_w = None
        self._last_p = None

    def _on_step(self) -> bool:
        if self.n_calls % self.update_every != 0:
            return True
        cfg = self.env_cfg

        frac_s = min(1.0, self.num_timesteps / max(cfg.shaped_ramp_steps, 1))
        w = cfg.shaped_weight_start + (cfg.shaped_weight_end - cfg.shaped_weight_start) * frac_s
        self.training_env.env_method("set_shaped_weight", w)

        p = min(1.0, self.num_timesteps / max(cfg.curriculum_ramp_steps, 1))
        self.training_env.env_method("set_curriculum_progress", p)

        self.logger.record("curriculum/shaped_weight", w)
        self.logger.record("curriculum/progress", p)
        self._last_w, self._last_p = w, p
        return True


class DiagnosticsCallback(BaseCallback):
    """Rollout boyunca info dict'lerini toplayip TensorBoard'a yazar."""

    def __init__(self):
        super().__init__()
        self._reset()

    def _reset(self):
        self.ata = []
        self.rng = []
        self.closing = []
        self.lock = []
        self.exposed = []
        self.hp_self = []
        self.hp_opp = []
        self.reasons = {}
        # YENI: terminate_on_fault=False iken bolum bitmiyor, bu yuzden
        # 'reset_reason' istatistigi artik instabilite sikligini
        # YAKALAYAMIYOR. Bu iki liste, HER ADIMDA (bolum bitse de
        # bitmese de) dengesizlik olup olmadigini takip eder.
        self.self_fault_flags = []
        self.opp_fault_flags = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        dones = self.locals.get("dones")

        for i, info in enumerate(infos):
            if "ata_deg" not in info:
                continue
            self.ata.append(info["ata_deg"])
            self.rng.append(info["range_ft"])
            self.closing.append(info["closing_fps"])
            self.lock.append(1.0 if info["opp_in_my_cone"] else 0.0)
            self.exposed.append(1.0 if info["me_in_opp_cone"] else 0.0)
            self.self_fault_flags.append(1.0 if info.get("self_fault") else 0.0)
            self.opp_fault_flags.append(1.0 if info.get("opp_fault") else 0.0)

            # Episode bittiyse: bitis sebebi ve kalan HP'ler
            if dones is not None and i < len(dones) and dones[i]:
                r = info.get("reset_reason") or "unknown"
                self.reasons[r] = self.reasons.get(r, 0) + 1
                self.hp_self.append(info.get("hp_self", 0.0))
                self.hp_opp.append(info.get("hp_opp", 0.0))
        return True

    def _on_rollout_end(self) -> None:
        if self.ata:
            self.logger.record("dogfight/ata_deg_mean", float(np.mean(self.ata)))
            self.logger.record("dogfight/range_ft_mean", float(np.mean(self.rng)))
            self.logger.record("dogfight/closing_fps_mean", float(np.mean(self.closing)))
            self.logger.record("dogfight/lock_rate", float(np.mean(self.lock)))
            self.logger.record("dogfight/exposed_rate", float(np.mean(self.exposed)))
        if self.self_fault_flags:
            self.logger.record("dogfight/self_fault_rate", float(np.mean(self.self_fault_flags)))
            self.logger.record("dogfight/opp_fault_rate", float(np.mean(self.opp_fault_flags)))
        if self.hp_self:
            self.logger.record("dogfight/hp_self_end", float(np.mean(self.hp_self)))
            self.logger.record("dogfight/hp_opp_end", float(np.mean(self.hp_opp)))
        total = sum(self.reasons.values())
        if total:
            for r, c in self.reasons.items():
                self.logger.record(f"reset_reason/{r}", c / total)
        self._reset()


class PeriodicSnapshotCallback(BaseCallback):
    def __init__(self, out_dir: Path, save_freq: int):
        super().__init__()
        self.snapshot_dir = Path(out_dir) / "live_snapshot"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = max(save_freq, 1)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            vecnorm = self.model.get_vec_normalize_env()
            self.model.save(str(self.snapshot_dir / "model"))
            if vecnorm is not None:
                vecnorm.save(str(self.snapshot_dir / "vecnormalize.pkl"))
            meta = {"num_timesteps": int(self.num_timesteps), "obs_dim": int(OBS_DIM)}
            (self.snapshot_dir / "meta.json").write_text(json.dumps(meta))
        return True


_OPPONENT_FAULT_REASONS = (
    "opponent_down", "opponent_ground", "opponent_ceiling",
    "opponent_tumble", "opponent_boundary",
)
_SELF_FAULT_REASONS = (
    "self_down", "self_ground", "self_ceiling", "self_tumble", "self_boundary",
)


def _episode_won(info) -> bool:
    """v8.1 DUZELTME: eskiden SADECE reset_reason=='opponent_down' acik
    galibiyet sayiliyordu; rakip kendi hatasiyla (tumble/boundary/ground/
    ceiling - yani sana hic kilit/hasar vermeden) duserse HICBIR dala
    girmiyor, en sonda hp_opp < hp_self karsilastirmasina dusuyordu.
    Ama boyle bir bolumde IKI TARAFIN DA HP'si hala baslangic degerinde
    (3.0 == 3.0) olabilir - '<' False cikip bu acik ustunluk GALIBIYET
    SAYILMIYORDU. Oysa ortamin odul fonksiyonu bunu zaten
    opponent_fault_bonus ile SENIN lehine puanliyordu (bkz.
    dogfight_env.py step()) - yani egitim sinyali ile degerlendirme
    metrigi CELISIYORDU. Bu, mean_reward surekli iyilesirken win_rate'in
    gurultulu/dusuk kalmasinin (ve promotion'un tikanmasinin) ana
    nedenlerinden biriydi.

    Simdi: rakibin HERHANGI bir kendi-hatasi (opponent_* reset_reason)
    ile bitmesi DOGRUDAN galibiyet sayilir - carpisma (collision) ve
    karsilikli dusme (mutual_down) haric, HP karsilastirmasina hic
    gerek kalmadan."""
    reason = info.get("reset_reason")
    if reason in _OPPONENT_FAULT_REASONS:
        return True
    if reason in _SELF_FAULT_REASONS:
        return False
    if reason in ("collision", "mutual_down"):
        return False
    # timeout ya da beklenmeyen bir sebep: HP farkina bak
    return float(info.get("hp_opp", 0.0)) < float(info.get("hp_self", 0.0))


def _evaluate_vs_fixed(model, vecnorm_stats: NormalizerStats, env_cfg,
                       fixed_opponent, n_episodes: int, shaped_weight=None):
    env = make_dogfight_env(env_cfg, fixed_opponent=fixed_opponent,
                            deterministic_opponent=True)
    if shaped_weight is not None:
        env.set_shaped_weight(shaped_weight)
    env.set_curriculum_progress(1.0)

    rewards, wins = [], 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        info = {}
        while not done:
            norm_obs = vecnorm_stats.normalize(obs).reshape(1, -1)
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        if _episode_won(info):
            wins += 1
    return float(np.mean(rewards)), wins / max(n_episodes, 1)


class PromotionCallback(BaseCallback):
    def __init__(self, pool: CheckpointPool, env_cfg, promo_cfg, out_dir: Path):
        super().__init__()
        self.pool = pool
        self.env_cfg = env_cfg
        self.promo_cfg = promo_cfg
        self.out_dir = out_dir
        self._consecutive_pass = 0
        self._next_eval = promo_cfg.eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval += self.promo_cfg.eval_freq

        latest = self.pool.latest()
        if latest is None:
            print("[promotion] havuz bos, atlaniyor")
            return True

        vecnorm_train = self.model.get_vec_normalize_env()
        stats = NormalizerStats(vecnorm_train)

        mean_reward, win_rate = _evaluate_vs_fixed(
            self.model, stats, self.env_cfg, latest, self.promo_cfg.n_eval_episodes
        )
        baseline = self.pool.latest_mean_reward()
        if baseline is None or abs(baseline) < 1e-6:
            improve_pct = 100.0
        else:
            improve_pct = (mean_reward - baseline) / abs(baseline) * 100.0

        passed = (win_rate >= self.promo_cfg.win_rate_threshold
                  and improve_pct >= self.promo_cfg.mean_reward_improve_pct)

        self.logger.record("promotion/win_rate", win_rate)
        self.logger.record("promotion/mean_reward", mean_reward)

        print(f"[promotion] step={self.num_timesteps} win_rate={win_rate:.2f} "
              f"mean_reward={mean_reward:.2f} (baseline={baseline}, "
              f"improve={improve_pct:+.1f}%) pass={passed} "
              f"({self._consecutive_pass + (1 if passed else 0)}/"
              f"{self.promo_cfg.consecutive_passes_required})")

        self._consecutive_pass = self._consecutive_pass + 1 if passed else 0

        if self._consecutive_pass >= self.promo_cfg.consecutive_passes_required:
            tmp_model = self.out_dir / f"_promo_candidate_{self.num_timesteps}.zip"
            tmp_vecnorm = self.out_dir / f"_promo_candidate_{self.num_timesteps}_vecnorm.pkl"
            self.model.save(str(tmp_model))
            vecnorm_train.save(str(tmp_vecnorm))

            version = self.pool.add(str(tmp_model), str(tmp_vecnorm), mean_reward,
                                    win_rate, note=f"step={self.num_timesteps}",
                                    obs_dim=OBS_DIM)
            print(f"[promotion] YENI VERSIYON: v{version} havuza eklendi!")

            tmp_model.unlink(missing_ok=True)
            tmp_vecnorm.unlink(missing_ok=True)
            self._consecutive_pass = 0

        return True


def _seed_pool(args):
    from drone_rl.dogfight.dogfight_env import DogfightEnv, OpponentCurriculum

    pool = CheckpointPool(args.pool)
    run_dir = Path(args.from_run)
    model_path = run_dir / "model_final.zip"
    vecnorm_path = run_dir / "vecnormalize.pkl"

    # DUZELTME: eskiden burada load_dogfight_config(None) cagriliyordu,
    # yani --config yok sayilip VARSAYILAN ayarlarla degerlendirme
    # yapiliyordu. Artik verilen config kullaniliyor.
    cfg = load_dogfight_config(args.config)

    vecnorm = VecNormalize.load(str(vecnorm_path), make_dummy_vecnorm_env())
    stats = NormalizerStats(vecnorm)
    if stats.obs_dim != OBS_DIM:
        raise ValueError(
            f"Stage A checkpoint'i {stats.obs_dim} boyutlu, bu surum {OBS_DIM} "
            f"bekliyor. Eski run'i kullanamazsiniz - Stage A'yi bu surumle "
            f"yeniden calistirin."
        )
    model = PPO.load(str(model_path), device="cpu")

    env = DogfightEnv(cfg.env, opponent_curriculum=OpponentCurriculum(cfg.env))
    env.set_curriculum_progress(1.0)
    env.set_shaped_weight(cfg.env.shaped_weight_end)

    rewards, wins = [], 0
    n = 20
    for _ in range(n):
        obs, _ = env.reset()
        done, ep_r, info = False, 0.0, {}
        while not done:
            norm_obs = stats.normalize(obs).reshape(1, -1)
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action[0])
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
        if _episode_won(info):
            wins += 1

    mean_reward = float(np.mean(rewards))
    win_rate = wins / n
    version = pool.add(str(model_path), str(vecnorm_path), mean_reward, win_rate,
                       note="stage-a seed", obs_dim=OBS_DIM)
    print(f"Havuz tohumlandi: v{version}, mean_reward={mean_reward:.2f}, "
          f"win_rate={win_rate:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, choices=["a", "b"])
    ap.add_argument("--config", type=str)
    ap.add_argument("--out", type=str, default="/content/runs/dogfight_run")
    ap.add_argument("--pool", type=str, default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--vec", type=str, default=None, choices=["auto", "dummy", "subproc"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--snapshot-freq", type=int, default=10000)
    ap.add_argument("--seed-pool", action="store_true")
    ap.add_argument("--from", dest="from_run", type=str)
    # YENI: --resume - runs/<out>/live_snapshot/{model.zip,vecnormalize.pkl}
    # mevcutsa egitim SIFIRDAN degil, oradan devam eder. Colab runtime
    # kopmalarinda ilerlemenin kaybolmamasi icin eklendi.
    ap.add_argument("--resume", action="store_true",
                    help="live_snapshot'ta kayitli model/vecnormalize varsa oradan devam et")
    args = ap.parse_args()

    if args.seed_pool:
        if not args.from_run or not args.pool:
            raise ValueError("--seed-pool icin --from ve --pool gerekli")
        _seed_pool(args)
        return

    cfg = load_dogfight_config(args.config)
    timesteps = args.timesteps or cfg.train.timesteps
    n_envs = args.n_envs or cfg.train.n_envs
    vec = args.vec or cfg.train.vec
    seed = args.seed if args.seed is not None else cfg.train.seed
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    live_dir = out / "live_snapshot"
    resume_model_path = live_dir / "model.zip"
    resume_vecnorm_path = live_dir / "vecnormalize.pkl"
    do_resume = bool(args.resume and resume_model_path.exists() and resume_vecnorm_path.exists())
    if args.resume and not do_resume:
        print(f"[resume] --resume verildi ama '{live_dir}' altinda kayitli bir "
             f"snapshot bulunamadi (model.zip/vecnormalize.pkl) - SIFIRDAN baslaniyor.")

    venv = make_dogfight_training_vec_env(
        cfg.env, n_envs=n_envs, stage=args.stage, pool_dir=args.pool,
        training=True, norm_reward=True, clip_reward=cfg.ppo.clip_reward,
        vec=vec, seed=seed,
        vecnormalize_path=str(resume_vecnorm_path) if do_resume else None,
    )

    policy_kwargs = build_policy_kwargs(cfg.ppo)

    if do_resume:
        # DIKKAT: policy_kwargs / mimari buradan gecilmiyor - PPO.load
        # kaydedilmis modelin KENDI mimarisini ve agirliklarini geri
        # yukler. Sadece cevre (env) YENI (guncel kod/odul duzeltmeleri
        # ile) baglaniyor - boylece TRAIN.PY'DA yapilan bir duzeltme
        # (orn. _episode_won mantigi ya da odul fonksiyonu) agirliklari
        # SIFIRLAMADAN devreye girer.
        model = PPO.load(str(resume_model_path), env=venv, device="cpu")

        # YENI: cokmus/zehirlenmis bir checkpoint'i SESSIZCE yukleyip
        # AYNI hatayla tekrar cokmek yerine, burada erkenden ve ACIKCA
        # durur.
        if _model_has_nan(model):
            raise RuntimeError(
                f"\n[resume] KRITIK: '{resume_model_path}' agirliklarinda "
                f"NaN/Inf tespit edildi - bu checkpoint bir NaN cokusu "
                f"SIRASINDA ya da SONRASINDA kaydedilmis, artik KULLANILAMAZ "
                f"(gozlemler temiz olsa bile ag kendi kendine NaN uretmeye "
                f"devam eder).\n\n"
                f"Secenekler:\n"
                f"  1) '{out}/ckpt/' klasorunde daha ESKI bir 'ppo_*_steps.zip' "
                f"var mi kontrol edin (varsa NaN icermeyen birini secip "
                f"--resume YERINE bu betigi elle o dosyayi yukleyecek "
                f"sekilde calistirmak gerekir).\n"
                f"  2) Havuzdaki (CheckpointPool) en son PROMOTE EDILMIS "
                f"versiyon KESINLIKLE NaN icermez (promosyon degerlendirmesi "
                f"NaN aksiyonla asla basarili olamaz) - 'python main.py "
                f"pool-info' ile kontrol edip o versiyonu yeni bir egitimin "
                f"baslangici olarak kullanabilirsiniz.\n"
                f"  3) --resume KULLANMADAN sifirdan baslatin - gozlem "
                f"kirpma duzeltmesi sayesinde bu sorun BIR DAHA olusmayacak."
            )

        already_done = int(model.num_timesteps)
        remaining = max(timesteps - already_done, 0)
        print(f"[resume] {resume_model_path} yuklendi.")
        print(f"[resume] su ana kadar tamamlanan adim : {already_done}")
        print(f"[resume] yaml'daki hedef adim          : {timesteps}")
        print(f"[resume] bu calistirmada kalan adim    : {remaining}")
    else:
        model = PPO(
            cfg.ppo.policy, venv,
            n_steps=cfg.ppo.n_steps, batch_size=cfg.ppo.batch_size, n_epochs=cfg.ppo.n_epochs,
            gamma=cfg.ppo.gamma, gae_lambda=cfg.ppo.gae_lambda, clip_range=cfg.ppo.clip_range,
            learning_rate=linear_schedule(cfg.ppo.learning_rate, cfg.ppo.lr_final_frac),
            ent_coef=cfg.ppo.ent_coef, vf_coef=cfg.ppo.vf_coef,
            max_grad_norm=cfg.ppo.max_grad_norm, target_kl=cfg.ppo.target_kl,
            policy_kwargs=policy_kwargs, verbose=1, device="cpu", seed=seed,
            tensorboard_log=str(out / "tb"),
        )
        already_done = 0
        remaining = timesteps

    # DUZELTME: save_vecnormalize=True eklendi - eskiden ckpt/ klasoru
    # SADECE model agirliklarini kaydediyordu, eslesen vecnormalize.pkl'i
    # DEGIL. Bu yuzden bir NaN cokusu sonrasi geri donulebilecek, hem
    # model hem normalizasyon istatistikleri birlikte SAGLAM bir ara
    # nokta yoktu - sadece live_snapshot vardi, o da corken onunla
    # birlikte bozuluyordu.
    ckpt_cb = CheckpointCallback(save_freq=max(20_000 // n_envs, 1),
                                 save_path=str(out / "ckpt"), name_prefix="ppo",
                                 save_vecnormalize=True)
    progress_cb = ProgressCallback(cfg.env)
    diag_cb = DiagnosticsCallback()
    snapshot_cb = PeriodicSnapshotCallback(out, save_freq=max(args.snapshot_freq // n_envs, 1))
    callbacks = [ckpt_cb, progress_cb, diag_cb, snapshot_cb]

    if args.stage == "b":
        if not args.pool:
            raise ValueError("--stage b icin --pool gerekli")
        pool = CheckpointPool(args.pool)
        if len(pool) == 0:
            raise ValueError("Havuz bos - once --seed-pool calistirin")
        promo_cb = PromotionCallback(pool, cfg.env, cfg.promotion, out)
        if do_resume:
            # YENI: resume aninda _next_eval'i mevcut ilerlemeye gore
            # HIZALA - yoksa (varsayilan _next_eval=eval_freq oldugu
            # icin) num_timesteps zaten cok ilerideyken ilk birkac
            # _on_step cagrisinda ART ARDA, GEREKSIZ COK SAYIDA
            # degerlendirme (her biri n_eval_episodes bolum!) tetiklenir.
            promo_cb._next_eval = (
                (already_done // cfg.promotion.eval_freq) + 1
            ) * cfg.promotion.eval_freq
            print(f"[resume] promotion degerlendirmesi bir sonraki "
                 f"adim={promo_cb._next_eval}'de tetiklenecek.")
        callbacks.append(promo_cb)

    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False, callback=callbacks)
    else:
        print(f"[resume] zaten hedef adim sayisina ({timesteps}) ulasilmis, "
             f"egitim atlaniyor.")

    model.save(out / "model_final")
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"Egitim tamamlandi: {out} (toplam num_timesteps={model.num_timesteps})")


if __name__ == "__main__":
    main()
