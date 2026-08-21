#!/usr/bin/env python3
"""녹화한 손 랜드마크로 리타겟터의 **자세 충실도**를 잰다.

왜 이 스크립트가 있는가
---------------------
p1_diag_compare_retargeters.py 는 엄지-검지 **손끝 간격**만 잰다. 그 자로만 보면
dex-retargeting 이 이긴다. 그런데 실제로 쓰면 "손을 쥐어도 로봇이 손처럼 안 쥔다"는
문제가 남았다. 자가 놓치고 있던 것이 있었다.

집기에서 중요한 건 손끝 간격이 맞다. 하지만 **텔레오퍼레이션에서 사람이 보는 것**은
자세다. 사람이 주먹을 쥐면 MCP(밑마디)가 접혀야 한다. 손끝만 목표로 주면 MCP 를
접든 펴든 손끝 위치는 맞출 수 있어서(널스페이스), 최적화가 MCP 를 오히려 펴고
PIP/DIP 에 굽힘을 몰아넣는 해로 가도 지표상 벌점이 없다. 결과가 주먹이 아니라
갈고리다. 이 스크립트는 그 축을 잰다.

무엇을 재는가
-----------
사람 랜드마크에서 뽑은 MCP/PIP/DIP 굽힘각과, 리타겟 결과 로봇 관절각을 나란히 놓는다.
검지/중지/약지 세 손가락 평균. 엄지는 제외한다 — LEAP 엄지는 축 배치가 사람과 달라
각 대응이 성립하지 않는다(핀치 여부로 봐야 한다. p1_diag_compare_retargeters.py 참고).

같이 찍는 값
    잔차mm   ours 의 IK 손끝 잔차 / dex 의 벡터 오차. 서로 다른 뜻이라 참고용이다.
    지터도   프레임 사이 관절각 최대 변화. 정지 자세 녹화라 클수록 나쁘다.
    핀치mm   MuJoCo 로 실제 물리를 돌렸을 때 th_tip <-> if_tip geom 간격.
             LEAP 이 사람보다 두 배 넘게 크니 절대값이 아니라 **닫히는 추세**를 본다.

쓰는 법
------
    python scripts/phase1/p1_diag_record_poses.py   # 먼저 녹화 (SPACE 로 자세마다 시작)
    python scripts/phase1/p1_diag_pose_fidelity.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from leap_hand_mapping import joint_map as jm             # noqa: E402
from leap_hand_mapping.retarget import LeapRetargeter     # noqa: E402
from leap_hand_mapping.retarget_dex import DexRetargeter  # noqa: E402

MJCF_SCENE = os.path.join(REPO, "third_party/mujoco_menagerie/leap_hand/scene_right.xml")

# 검지/중지/약지의 (MCP, PIP, DIP, TIP) 랜드마크 번호
HUMAN_FINGERS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16)]

WARMUP = 10  # 평활/저역통과가 안정될 때까지 버리는 프레임 수


def human_angles(lm: np.ndarray) -> np.ndarray:
    """사람 랜드마크에서 MCP/PIP/DIP 굽힘각(deg). 세 손가락 평균, 굽힘이 양수.

    MCP 는 손바닥 법선을 기준으로 잰다. 손바닥 평면은 손목(0)-검지MCP(5)-새끼MCP(17)
    로 만든다. PIP/DIP 는 이웃한 두 마디 사이 각이라 부호 규약이 필요 없다.
    """
    n = np.cross(lm[5] - lm[0], lm[17] - lm[0])
    n /= np.linalg.norm(n)

    def between(a, b):
        return np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))

    out = []
    for M, P, D, T in HUMAN_FINGERS:
        m = lm[M] - lm[0]
        m /= np.linalg.norm(m)
        v = lm[P] - lm[M]
        mcp = np.arctan2(np.dot(v, n), np.dot(v, m))  # 손바닥 쪽으로 접히면 양수
        out.append((mcp, between(lm[P] - lm[M], lm[D] - lm[P]),
                    between(lm[D] - lm[P], lm[T] - lm[D])))
    return np.degrees(np.array(out)).mean(axis=0)


def make_retargeter(kind: str, calib: np.ndarray | None):
    if kind == "ours":
        r = LeapRetargeter(gui=False)
        if calib is not None:
            for w in calib:
                r.observe_calibration(w)
            r.finish_calibration()
        return r
    return DexRetargeter(hand_type="Right", retargeting_type=kind)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", default=os.path.join(REPO, "thumb_capture.npz"))
    ap.add_argument("--which", nargs="+", default=["ours", "dexpilot", "vector"],
                    choices=["ours", "dexpilot", "vector"])
    ap.add_argument("--calib-frames", type=int, default=30,
                    help="ours 의 엄지 정렬에 쓸 편 손 프레임 수")
    args = ap.parse_args()

    if not os.path.exists(args.capture):
        print(f"녹화 파일이 없다: {args.capture}")
        print("먼저: python scripts/phase1/p1_diag_record_poses.py")
        return 2

    import mujoco

    cap = np.load(args.capture, allow_pickle=True)
    labels = [str(x) for x in cap["pose_labels"]]
    poses = [np.asarray(cap[f"pose{i}"], dtype=float) for i in range(3, 3 + len(labels))]
    calib = np.asarray(cap["calib_rest"], dtype=float)[:args.calib_frames] \
        if "calib_rest" in cap and args.calib_frames else None

    model = mujoco.MjModel.from_xml_path(MJCF_SCENE)
    jm.apply_model_limits(model)   # menagerie 엄지 범위 정정
    data = mujoco.MjData(model)
    g_th = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "th_tip")
    g_if = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "if_tip")
    steps = max(1, int((1 / 30.0) / model.opt.timestep))

    idx = {n: i for i, n in enumerate(jm.MUJOCO_JOINT_NAMES)}
    GROUPS = [[idx[f"{f}_{j}"] for f in ("if", "mf", "rf")] for j in ("mcp", "pip", "dip")]

    print(f"녹화 {os.path.relpath(args.capture, REPO)}  자세 {len(labels)}개")
    # 사람 수치는 녹화에서 직접 센다 (박아 둔 숫자는 녹화를 바꾸면 어긋난다).
    gaps = [np.linalg.norm(w[:, 4] - w[:, 8], axis=1).mean() * 1000 for w in poses]
    print("사람 실측 엄지-검지 손끝 간격(mm): "
          + " / ".join(f"{l} {g:.1f}" for l, g in zip(labels, gaps)) + "\n")

    for kind in args.which:
        r = make_retargeter(kind, calib if kind == "ours" else None)
        tag = "ours (손끝 위치 IK)" if kind == "ours" else f"dex-retargeting ({kind})"
        if kind == "ours" and calib is not None:
            tag += f"  엄지 보정각 {np.degrees(r.thumb_align_angle()):.1f} deg"
        print(tag)
        print(f"  {'자세':<12}{'mcp사람':>8}{'mcp로봇':>8}{'pip사람':>8}{'pip로봇':>8}"
              f"{'dip사람':>8}{'dip로봇':>8}{'잔차mm':>8}{'지터도':>7}{'핀치mm':>8}{'ms':>6}")

        for label, frames in zip(labels, poses):
            r.reset()
            q_prev = np.zeros(jm.NUM_JOINTS)
            H, R, res, jit, pinch, ms = [], [], [], [], [], []
            for k, w in enumerate(frames):
                t0 = time.time()
                q = r.retarget(w, dt=1 / 30.0)
                ms.append((time.time() - t0) * 1000)
                if k < WARMUP:
                    q_prev = q
                    continue
                H.append(human_angles(w))
                R.append([np.degrees(q[g]).mean() for g in GROUPS])
                e = r.tip_error()
                res.append(float(np.nanmean(e) if kind != "ours" else np.mean(e[1::2])) * 1000)
                jit.append(np.degrees(np.abs(q - q_prev).max()))
                q_prev = q
                data.ctrl[:] = q
                for _ in range(steps):
                    mujoco.mj_step(model, data)
                pinch.append(np.linalg.norm(data.geom_xpos[g_th] - data.geom_xpos[g_if]) * 1000)

            h, rb = np.mean(H, axis=0), np.mean(R, axis=0)
            print(f"  {label:<12}{h[0]:8.1f}{rb[0]:8.1f}{h[1]:8.1f}{rb[1]:8.1f}"
                  f"{h[2]:8.1f}{rb[2]:8.1f}{np.mean(res):8.2f}{np.mean(jit):7.2f}"
                  f"{np.mean(pinch):8.1f}{np.mean(ms):6.2f}")
        r.close()
        print()

    print("읽는 법: mcp 열이 사람과 같은 방향으로 움직이는지를 먼저 본다.")
    print("주먹에서 사람 mcp 가 크게 양수인데 로봇이 0 근처거나 음수면, 그 리타겟터는")
    print("손끝만 맞추고 손 모양은 버리고 있다는 뜻이다 (주먹이 아니라 갈고리가 된다).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
