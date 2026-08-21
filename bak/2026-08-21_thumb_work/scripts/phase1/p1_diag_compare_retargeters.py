#!/usr/bin/env python3
"""녹화한 손 랜드마크로 두 리타겟터를 같은 자로 잰다.

왜 이 스크립트가 있는가
---------------------
"어느 리타겟터를 기본으로 쓸 것인가"를 느낌이 아니라 숫자로 정하기 위해서다.
이 저장소는 한 번 배율(scaling_factor)을 내가 만든 지표만 보고 upstream 기본값에서
바꿨다가 되돌린 적이 있다. 그 뒤로는 **재현 가능한 측정**만 근거로 쓴다.

무엇을 재는가
-----------
조작에서 중요한 것은 개별 손끝의 절대 위치가 아니라 **손끝 사이 거리**다. 집으려면
엄지와 검지가 만나야 한다. 그래서 엄지-검지 손끝 간격을 사람/ours/dex 세 가지로
같은 로봇 모델(Bidex URDF)의 순기구학으로 재서 나란히 놓는다.

'완벽한 센서' 검사
----------------
카메라가 원인인지 알고리즘이 원인인지 가른다. 사람 엄지 끝 랜드마크를 검지 끝에
정확히 겹쳐 놓고(=간격 0mm, 무한히 좋은 3D 센서) 목표를 다시 계산한다. 목표 간격이
그래도 크면 카메라를 바꿔도 소용없다는 뜻이다.

쓰는 법
------
    python scripts/phase1/p1_diag_record_poses.py   # 먼저 녹화 (SPACE 로 자세마다 시작)
    python scripts/phase1/p1_diag_compare_retargeters.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from leap_hand_mapping import hand_tracker as ht          # noqa: E402
from leap_hand_mapping.retarget import LeapRetargeter     # noqa: E402
from leap_hand_mapping.retarget_dex import DexRetargeter  # noqa: E402

# _ee_positions() 는 손가락마다 (앞마디, 손끝) 2개씩 index/middle/ring/thumb 순이다.
TIP_INDEX = 1
TIP_THUMB = 7


def tip_gap(model: LeapRetargeter, q: np.ndarray) -> float:
    """관절각 q 일 때 엄지-검지 손끝 간격(m). 모델은 순기구학으로만 쓴다."""
    model._set_joints(q)
    ee = model._ee_positions()
    return float(np.linalg.norm(ee[TIP_THUMB] - ee[TIP_INDEX]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", default=os.path.join(REPO, "thumb_capture.npz"),
                    help="p1_diag_record_poses.py 로 만든 녹화 파일")
    args = ap.parse_args()

    if not os.path.exists(args.capture):
        print(f"녹화 파일이 없다: {args.capture}\n"
              f"  python scripts/phase1/p1_diag_record_poses.py --save {args.capture}")
        return 1

    d = np.load(args.capture, allow_pickle=True)
    labels = [str(x) for x in d["pose_labels"]]
    poses = [k for k in d.files if k.startswith("pose")and k != "pose_labels"]
    poses.sort(key=lambda k: int(k[4:]))

    ours = LeapRetargeter()
    for w in d["calib_rest"]:
        ours.observe_calibration(w)
    if not ours.finish_calibration():
        print("캘리브레이션 표본 부족")
        return 1
    dex = DexRetargeter()
    fk = LeapRetargeter()          # dex 출력을 같은 모델로 재기 위한 순기구학 전용

    print(f"\n녹화 {os.path.relpath(args.capture, REPO)}  "
          f"배율 ours={ {k: round(v, 2) for k, v in ours.frozen_scales.items()} } "
          f"dex={dex.scaling:.2f}\n")
    print("엄지-검지 손끝 간격 (mm, 평균 / 그 자세의 최소)")
    print(f"{'자세':<16}{'사람':>16}{'ours':>16}{'dex':>16}")

    for key, label in zip(poses, labels):
        W = d[key]
        hum = np.array([np.linalg.norm(w[ht.THUMB_TIP] - w[ht.INDEX_TIP]) for w in W]) * 1e3
        a = np.array([tip_gap(ours, ours.retarget(w, 0.02)) for w in W]) * 1e3
        b = np.array([tip_gap(fk, dex.retarget(w, 0.02)) for w in W]) * 1e3
        cell = lambda v: f"{v.mean():7.1f} /{v.min():6.1f}"
        print(f"{label:<16}{cell(hum):>16}{cell(a):>16}{cell(b):>16}")

    # --- 완벽한 센서 가정 -------------------------------------------------
    print("\n완벽한 3D 센서 가정 (사람 엄지 끝을 검지 끝에 정확히 겹침 = 간격 0mm)")
    for key, label in zip(poses, labels):
        W = d[key]
        g = []
        for w in W:
            w2 = w.copy()
            w2[ht.THUMB_TIP] = w2[ht.INDEX_TIP]
            t = ours.compute_targets(w2)
            g.append(np.linalg.norm(t[TIP_THUMB] - t[TIP_INDEX]))
        g = np.array(g) * 1e3
        print(f"  {label:<14} ours 의 IK 목표 간격 {g.mean():6.1f} mm (최소 {g.min():.1f})")
    print("  -> 이 값이 크면 원인은 카메라가 아니라 리타겟터다.")

    # --- 처리 시간 --------------------------------------------------------
    # 반복 횟수가 자세마다 다르므로 전 자세를 이어 붙여서 잰다. 첫 호출은 지연
    # 초기화(순기구학 모델 적재)가 섞이니 워밍업으로 버린다.
    W = np.concatenate([d[k] for k in poses])
    print()
    for name, r in (("ours", ours), ("dex", dex)):
        for w in W[:20]:
            r.retarget(w, 0.02)
        t0 = time.perf_counter()
        for w in W:
            r.retarget(w, 0.02)
        print(f"  {name:<5} {(time.perf_counter() - t0) / len(W) * 1e3:.1f} ms/frame "
              f"(n={len(W)})")

    ours.close()
    fk.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
