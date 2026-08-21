#!/usr/bin/env python3
"""엄지 해법 후보들을 녹화 데이터로 같은 자로 잰다.

왜 이 스크립트가 있는가
---------------------
`--retargeter ours` 를 기본으로 되돌린 뒤 남은 문제가 엄지다. 원인은 retarget.py 의
"왜 엄지만 안 맞는가" 주석에 적힌 그대로다 — 손가락마다 **자기 뿌리를 원점으로 자기
배율로** 목표를 만들어서, 사람이 엄지를 검지에 붙여도 로봇 목표는 149mm 벌어져 있다.
IK 는 그 목표를 3.6mm 오차로 정확히 달성한다. **IK 가 아니라 목표가 틀렸다.**

여기서 재는 후보들
    base   엄지 결합을 끈 상태(--no-thumb-couple). 목표 = 자기 뿌리 원점 + 자기 배율
    C      th_axl 을 IK 변수에서 빼고 0 으로 고정.
           th_axl 은 회전축이 엄지 손끝을 거의 지나서 1 rad 당 손끝이 12~22mm 밖에
           안 움직인다(th_cmc 는 135mm). 손끝만 보는 IK 에서는 사실상 자유변수라
           한계(-27도; 한계표 정정 전에는 -20도)에 붙어 버린다.
    A1     DexPilot 투영. 사람 엄지-검지 간격이 project_dist(30mm) 밑이면 목표 간격을
           0 으로 눌러 "붙여라"로 바꾸고, escape_dist(50mm) 위면 손대지 않는다.
           두 상수는 dex-retargeting 의 leap_hand_right_dexpilot.yml 값 그대로다.
    A2     A1 + 50mm 위에서도 엄지 목표를 **검지 목표 기준 상대**로 배치한다.
           간격은 사람 간격 x 검지 배율. 로봇이 사람보다 1.91배 크니 비례가 맞다.
    A2C    A2 + C
    hybrid 네 손가락은 ours, 엄지 4관절만 dex 것으로 교체
    now    지금 본 경로에 들어가 있는 코드 (retarget._couple_thumb). A2 와 같아야 한다.

A2 를 채택해 본 경로에 넣었다. 이 스크립트는 그 결정의 근거로 남는다.

엄지와 네 손가락은 자코비안이 분리된다(손가락 손끝은 엄지 관절에 안 달렸다).
그래서 엄지 목표를 바꿔도 MCP 열이 안 변한다 — 표에서 직접 확인할 수 있다.

쓰는 법
------
    python scripts/phase1/p1_diag_record_poses.py   # 먼저 녹화 (SPACE 로 자세마다 시작)
    python scripts/phase1/p1_diag_thumb_options.py
"""

import os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from leap_hand_mapping import joint_map as jm
from leap_hand_mapping.retarget import LeapRetargeter
from leap_hand_mapping.retarget_dex import DexRetargeter

TIP_IF, TIP_TH, DIP_TH = 1, 7, 6
THJ = [12, 13, 14, 15]
AXL = 13
PROJECT, ESCAPE = 0.03, 0.05     # DexPilot 기본값 (dex-retargeting leap dexpilot yml)

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--thumb-mode", default="map", choices=["map", "ik"],
                 help="LeapRetargeter 의 thumb_mode. 본 경로 기본은 map. ik 는 예전 경로")
_ap.add_argument("--capture", default=os.path.join(REPO, "thumb_capture.npz"))
_args = _ap.parse_args()

cap = np.load(_args.capture, allow_pickle=True)
labels = [str(x) for x in cap["pose_labels"]]
poses = [np.asarray(cap[f"pose{i}"], float) for i in range(3, 3 + len(labels))]
CALIB = np.asarray(cap["calib_rest"], float)[:30]


class Exp(LeapRetargeter):
    def __init__(self, mode, **kw):
        self.mode = mode
        super().__init__(**kw)

    def compute_targets(self, world):
        t = super().compute_targets(world)
        if "A" not in self.mode:
            return t   # base / C / hybrid / now 는 부모 결과를 그대로 쓴다
        g_h = float(np.linalg.norm(world[4] - world[8]))
        p_if, p_th = t[TIP_IF], t[TIP_TH]
        v = p_th - p_if
        g0 = float(np.linalg.norm(v))
        if g0 < 1e-9:
            return t
        u = v / g0
        if g_h <= PROJECT:
            g_new = 0.0
        elif g_h >= ESCAPE:
            # A1 은 손대지 않는다. A2 는 검지 배율로 상대 배치한다.
            g_new = g0 if "A1" in self.mode else g_h * self.frozen_scales["index"]
        else:
            far = g0 if "A1" in self.mode else g_h * self.frozen_scales["index"]
            g_new = (g_h - PROJECT) / (ESCAPE - PROJECT) * far
        shift = (p_if + u * g_new) - p_th
        t = t.copy()
        t[TIP_TH] += shift
        t[DIP_TH] += shift
        return t

    def _solve_dls(self, targets, seed=None):
        if "C" not in self.mode:
            return super()._solve_dls(targets, seed)
        q = (self._q if seed is None else np.asarray(seed, float)).copy()
        q[AXL] = 0.0
        lo, hi = jm.LIMITS_MJ_LOWER, jm.LIMITS_MJ_UPPER
        eye = np.eye(jm.NUM_JOINTS)
        w = self.target_weights
        for _ in range(self.ik_iterations):
            self._set_joints(q)
            error = (targets - self._ee_positions()).ravel() * w
            if np.abs(error).max() < self.ik_tolerance:
                break
            J = self._jacobian(q) * w[:, None]
            J[:, AXL] = 0.0                      # th_axl 을 자유변수에서 뺀다
            dq = np.linalg.solve(J.T @ J + (self.ik_damping ** 2) * eye, J.T @ error)
            dq = np.clip(dq, -self.ik_max_step, self.ik_max_step)
            new_q = np.clip(q + dq, lo, hi)
            new_q[AXL] = 0.0
            if np.abs(new_q - q).max() < 1e-9:
                break
            q = new_q
        return q


def build(mode):
    # 본 경로의 결합은 끈다. 여기서는 후보 로직을 이 파일 안에서 직접 준다.
    r = Exp(mode, gui=False, thumb_couple=(mode == "now"), thumb_mode=_args.thumb_mode)
    for w in CALIB:
        r.observe_calibration(w)
    r.finish_calibration()
    return r


fk = build("base")          # 순기구학 전용 (모든 모드의 간격을 같은 모델로 잰다)
dex = DexRetargeter(hand_type="Right", retargeting_type="dexpilot")

IDX = {n: i for i, n in enumerate(jm.MUJOCO_JOINT_NAMES)}
MCPJ = [IDX[f"{f}_mcp"] for f in ("if", "mf", "rf")]

print(f"{'모드':<8}{'자세':<12}{'핀치mm':>8}{'엄지잔차':>9}{'검지잔차':>9}"
      f"{'th_cmc':>8}{'th_axl':>8}{'th_mcp':>8}{'MCP도':>7}{'지터':>6}{'ms':>6}")
print("-" * 92)

import time
for mode in ["base", "C", "A1", "A2", "A2C", "hybrid", "now"]:
    r = build("base" if mode == "hybrid" else mode)
    for label, frames in zip(labels, poses):
        r.reset()
        G, RT, RI, TH, MC, JT, MS = [], [], [], [], [], [], []
        q_prev = np.zeros(16)
        for k, w in enumerate(frames):
            t0 = time.time()
            q = r.retarget(w, dt=1/30)
            if mode == "hybrid":
                q = q.copy(); q[THJ] = dex.retarget(w, dt=1/30)[THJ]
            MS.append((time.time() - t0) * 1000)
            if k < 10:
                q_prev = q; continue
            e = r.tip_error(); RT.append(e[TIP_TH]*1000); RI.append(e[TIP_IF]*1000)
            fk._set_joints(q); ee = fk._ee_positions()
            G.append(np.linalg.norm(ee[TIP_TH] - ee[TIP_IF]) * 1000)
            TH.append(np.degrees(q[THJ])); MC.append(np.degrees(q[MCPJ]).mean())
            JT.append(np.degrees(np.abs(q - q_prev).max())); q_prev = q
        th = np.mean(TH, axis=0)
        print(f"{mode:<8}{label:<12}{np.mean(G):8.1f}{np.mean(RT):9.1f}{np.mean(RI):9.1f}"
              f"{th[0]:8.1f}{th[1]:8.1f}{th[2]:8.1f}{np.mean(MC):7.1f}"
              f"{np.mean(JT):6.2f}{np.mean(MS):6.1f}")
    r.close()
    print()
# 사람 쪽 수치는 녹화에서 직접 센다. 예전에는 여기 숫자를 박아 두었는데, 녹화를
# 다시 만든 뒤 그 숫자가 옛 데이터(오염된 타이머 녹화)의 것으로 남아 표와 어긋났다.
_gap = [np.linalg.norm(w[:, 4] - w[:, 8], axis=1).mean() * 1000 for w in poses]
print(f"[thumb_mode={_args.thumb_mode}]  녹화 {os.path.basename(_args.capture)}")
print("사람 실측 엄지-검지 간격: " + " / ".join(f"{l} {g:.1f}" for l, g in zip(labels, _gap)) + " mm")
_dexg = []
for frames in poses:
    _d = DexRetargeter(hand_type="Right", retargeting_type="dexpilot")
    _g = []
    for k, w in enumerate(frames):
        q = _d.retarget(w, dt=1/30)
        if k >= 10:
            fk._set_joints(q); ee = fk._ee_positions()
            _g.append(np.linalg.norm(ee[TIP_TH] - ee[TIP_IF]) * 1000)
    _dexg.append(np.mean(_g))
print("dex(dexpilot) 핀치 간격:  " + " / ".join(f"{l} {g:.1f}" for l, g in zip(labels, _dexg)) + " mm")
fk.close()
