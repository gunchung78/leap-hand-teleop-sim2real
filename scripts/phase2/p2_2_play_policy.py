#!/usr/bin/env python3
"""p2_2_play_policy.py — 학습된 rotate_z 정책을 **일반 MuJoCo(CPU)** 에서 재생하고 숫자를 찍는다 (S2).

MJX(학습)와 MuJoCo(트윈 sim_node, 실기 전 단계)는 같은 모델이지만 다른 적분기/접촉 구현이다. 정책이
MJX 밖에서도 큐브를 돌리는지, 그리고 실기에 내려보내기 전에 알아야 할 것들을 여기서 잰다:

    큐브 z 각속도 평균 (rad/s)     목표 그 자체
    떨어지기까지 시간 (s)          에피소드 25 s 를 버티나
    관절 토크 RMS / 최대 (N·m)     qfrc_actuator(actuatorfrcrange ±0.2196 로 잘린 값). 예상 전류 mA = 토크 / 0.366
                                  (업스트림 xml 의 환산, 0.2196 = 600 mA). Lite 한계 350 mA = 0.128 N·m
    벌림(rot) 관절 사용 범위 (deg)  텔레옵 제한(±3°)과 충돌하는지

관측은 학습과 같게 만든다: [관절각 16 + 잡음 ±0.05, 직전 행동 16] (history_len 1 → 32차원).
행동 → 목표각 = 기본자세 + 0.6·a (학습 env 와 같은 식, 클립 없음).

    conda activate leap-mjx
    python scripts/phase2/p2_2_play_policy.py --ckpt logs/phase2/<run>/checkpoints            # 최신 체크포인트
    python scripts/phase2/p2_2_play_policy.py --ckpt ... --episodes 5 --video out.mp4        # 영상
    python scripts/phase2/p2_2_play_policy.py --ckpt ... --viewer                             # 창으로 보기
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

TORQUE_PER_AMP = 0.366          # leap_rh_mjx.xml 주석: max torque = 600/1000 * 0.366
LITE_CURR_LIM_MA = 350.0
ROT_IDX = [1, 5, 9]             # if_rot mf_rot rf_rot (MuJoCo 순서)
JOINT_NAMES = ["if_mcp", "if_rot", "if_pip", "if_dip", "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
               "rf_mcp", "rf_rot", "rf_pip", "rf_dip", "th_cmc", "th_axl", "th_mcp", "th_ipl"]


def load_policy(ckpt: Path, deterministic: bool = True):
    """brax ppo checkpoint.load_policy 와 같지만, brax 0.14.2 의 버그를 피한다.

    업스트림 train 이 쓴 ppo_network_config.json 에 `mean_kernel_init_fn: null` 이 들어가는데
    brax 의 load_config 는 None 도 KERNEL_INITIALIZER 에서 찾다가 KeyError 를 낸다. None 인 키는 빼고
    나머지는 같은 절차로 망을 만든다. 정책 추론 함수(obs, rng) -> (action, extras) 를 돌려준다.
    """
    import json

    from brax.training import networks as brax_networks
    from brax.training import checkpoint as brax_ckpt
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    cfg = json.loads((ckpt / "ppo_network_config.json").read_text())
    kw = dict(cfg["network_factory_kwargs"])
    for k in list(kw):
        if kw[k] is None:
            kw.pop(k)                                   # brax 버그 회피
        elif k.endswith("kernel_init_fn"):
            kw[k] = brax_networks.KERNEL_INITIALIZER[kw[k]]
    if "activation" in kw:
        kw["activation"] = brax_networks.ACTIVATION[kw["activation"]]
    obs_size = {k: tuple(v["shape"]) for k, v in cfg["observation_size"].items()}
    preprocess = running_statistics.normalize if cfg["normalize_observations"] else (lambda x, _: x)
    net = ppo_networks.make_ppo_networks(obs_size, cfg["action_size"], preprocess_observations_fn=preprocess, **kw)
    params = brax_ckpt.load(ckpt)
    return ppo_networks.make_inference_fn(net)(params, deterministic=deterministic)


def latest_ckpt(path: Path) -> Path:
    if (path / "ppo_network_config.json").exists():
        return path
    cands = sorted([p for p in path.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
    if not cands:
        raise SystemExit(f"{path} 안에 체크포인트 디렉터리(숫자 이름)가 없다")
    return cands[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoints/ 디렉터리 또는 그 안의 스텝 디렉터리")
    ap.add_argument("--env_name", default="LeapCubeRotateZAxis")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=25.0, help="에피소드 길이 (학습 500 스텝 = 25 s)")
    ap.add_argument("--noise", type=float, default=0.05, help="관절각 관측 잡음 (학습 0.05). 0 이면 끔")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default=None, help="mp4 경로 (mediapy)")
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    import jax
    import mujoco
    from mujoco_playground import registry

    ckpt = latest_ckpt(Path(args.ckpt).resolve())
    print(f"체크포인트 {ckpt}")
    policy = jax.jit(load_policy(ckpt, deterministic=True))

    cfg = registry.get_default_config(args.env_name)
    cfg.impl = "jax"
    env = registry.load(args.env_name, config=cfg)
    m = env.mj_model                      # 같은 모델, CPU MuJoCo
    d = mujoco.MjData(m)
    n_sub = int(round(cfg.ctrl_dt / m.opt.timestep))
    key = m.key("home")
    hand_q = np.arange(16)                # 손 관절이 qpos 앞 16 (playground 구성)
    default_pose = np.array(key.qpos[:16])
    cube_qadr = m.jnt_qposadr[m.body("cube").jntadr[0]]
    s_angvel = m.sensor("cube_angvel").adr[0]
    s_pos = m.sensor("cube_position").adr[0]
    print(f"ctrl_dt {cfg.ctrl_dt}  timestep {m.opt.timestep}  substeps {n_sub}  action_scale {cfg.action_scale}")

    rng = np.random.default_rng(args.seed)
    frames = []
    renderer = mujoco.Renderer(m, 480, 640) if args.video else None
    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(m, d)

    rows = []
    for ep in range(args.episodes):
        mujoco.mj_resetDataKeyframe(m, d, key.id)
        d.qpos[:16] = default_pose + 0.1 * rng.standard_normal(16)
        yaw = rng.uniform(-np.pi, np.pi)
        d.qpos[cube_qadr + 3:cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.ctrl[:] = d.qpos[:16]
        d.mocap_pos[:] = [-100.0, -100.0, -100.0]          # 목표 큐브(goal) 숨김 — env.reset 과 같게
        mujoco.mj_forward(m, d)
        last_act = np.zeros(16)
        angvels, torques, rots = [], [], []
        t_drop = None
        steps = int(args.seconds / cfg.ctrl_dt)
        pk = jax.random.PRNGKey(ep)
        t_start = time.perf_counter()
        for k in range(steps):
            q = d.qpos[hand_q].copy()
            if args.noise > 0:
                q = q + rng.uniform(-1, 1, 16) * args.noise
            obs = {"state": np.concatenate([q, last_act]).astype(np.float32),
                   "privileged_state": np.zeros(105, np.float32)}
            pk, sk = jax.random.split(pk)
            act, _ = policy(obs, sk)
            act = np.asarray(act)
            d.ctrl[:] = default_pose + act * cfg.action_scale
            for _ in range(n_sub):
                mujoco.mj_step(m, d)
            last_act = act
            angvels.append(d.sensordata[s_angvel + 2])
            torques.append(d.qfrc_actuator[:16].copy())   # 관절 actuatorfrcrange(±0.2196) 로 잘린 뒤의 토크. actuator_force 는 잘리기 전
            rots.append(d.qpos[hand_q][ROT_IDX].copy())
            if t_drop is None and d.sensordata[s_pos + 2] < -0.05:
                t_drop = (k + 1) * cfg.ctrl_dt
                break
            if renderer is not None and k % 2 == 0:
                renderer.update_scene(d, camera=-1)
                frames.append(renderer.render().copy())
            if viewer is not None:
                viewer.sync()
                rest = cfg.ctrl_dt - (time.perf_counter() - t_start) % cfg.ctrl_dt
                time.sleep(max(0.0, min(rest, cfg.ctrl_dt)))
        tq = np.array(torques)
        ma = np.abs(tq) / TORQUE_PER_AMP * 1000.0
        rots = np.degrees(np.array(rots))
        rows.append(dict(
            ep=ep, angvel=float(np.mean(angvels)), t_drop=t_drop, held=len(angvels) * cfg.ctrl_dt,
            tq_rms=float(np.sqrt((tq ** 2).mean())), tq_max=float(np.abs(tq).max()),
            ma_max=float(ma.max()), over=float((ma > LITE_CURR_LIM_MA).mean()),
            rot_min=rots.min(axis=0), rot_max=rots.max(axis=0),
            worst=JOINT_NAMES[int(np.argmax(np.abs(tq).max(axis=0)))],
        ))
        r = rows[-1]
        print(f"ep {ep}: 각속도 z {r['angvel']:+.2f} rad/s  버틴 시간 {r['held']:.1f} s"
              f"{'' if t_drop is None else f'  (떨어짐 {t_drop:.1f} s)'}  토크 RMS {r['tq_rms']:.3f} 최대 {r['tq_max']:.3f} N·m"
              f"  예상 전류 최대 {r['ma_max']:.0f} mA  >350 비율 {r['over']:.1%} ({r['worst']})"
              f"  rot 범위 if[{r['rot_min'][0]:.0f},{r['rot_max'][0]:.0f}] mf[{r['rot_min'][1]:.0f},{r['rot_max'][1]:.0f}]"
              f" rf[{r['rot_min'][2]:.0f},{r['rot_max'][2]:.0f}] deg")

    if viewer is not None:
        viewer.close()
    if renderer is not None and frames:
        import mediapy as media
        media.write_video(args.video, frames, fps=1.0 / cfg.ctrl_dt / 2)
        print(f"영상 {args.video} ({len(frames)} 프레임)")

    n = len(rows)
    print(f"\n요약 ({n} 에피소드, {args.seconds:.0f} s, 잡음 {args.noise})")
    print(f"  큐브 z 각속도 평균 {np.mean([r['angvel'] for r in rows]):+.2f} rad/s"
          f"   떨어뜨린 에피소드 {sum(r['t_drop'] is not None for r in rows)}/{n}"
          f"   토크 RMS {np.mean([r['tq_rms'] for r in rows]):.3f} N·m  최대 {max(r['tq_max'] for r in rows):.3f}"
          f"   예상 전류 최대 {max(r['ma_max'] for r in rows):.0f} mA  >350 비율 {np.mean([r['over'] for r in rows]):.1%}")
    rmin = np.min([r['rot_min'] for r in rows], axis=0); rmax = np.max([r['rot_max'] for r in rows], axis=0)
    print(f"  rot 관절 사용 범위 deg: if_rot [{rmin[0]:.0f},{rmax[0]:.0f}]  mf_rot [{rmin[1]:.0f},{rmax[1]:.0f}]  rf_rot [{rmin[2]:.0f},{rmax[2]:.0f}]"
          f"   (텔레옵 제한: mf ±3, if ≤+3, rf ≥−3)")
    print("  읽는 법: 전류 최대가 350 을 크게 넘으면 실기에서 브리지가 얼린다 → v1(토크 상한·벌점). rot 범위가 ±3° 를"
          " 넘으면 정책 경로에선 텔레옵 제한을 못 쓴다 → S5 에서 정책용 클립 표를 따로 정한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
