"""엄지 리타겟팅만 따로 진단한다.

엄지는 사람과 LEAP 의 기하 차이가 가장 큰 부분이라 증상도 여기서 먼저 나온다.
이 스크립트는 자세 몇 개를 순서대로 시키면서, 매 자세마다

  - 엄지 목표가 LEAP 도달거리 안에 있는가
  - IK 가 실제로 거기 갔는가
  - 관절이 한계에 붙어 있는가 (붙어 있으면 작업공간 밖이라는 신호)

를 출력한다. 결과를 그대로 복사해 붙이면 원인을 가릴 수 있다.

    python scripts/diag_thumb.py

화면 없이 돌리려면 --no-window. 자세마다 3초씩 준다(--hold 로 변경).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leap_hand_mapping import hand_tracker as ht  # noqa: E402
from leap_hand_mapping import joint_map as jm  # noqa: E402
from leap_hand_mapping.retarget import (  # noqa: E402
    FINGERS,
    FINGER_JOINTS,
    LeapRetargeter,
)

# (이름, 화면 안내). 캘리브레이션이 끝난 뒤 순서대로 시킨다.
POSES = [
    ("편 손", "손을 활짝 편 상태로 유지"),
    ("엄지만 붙임", "네 손가락은 편 채로 엄지만 손바닥 쪽으로"),
    ("검지-엄지 핀치", "검지와 엄지 끝을 맞댄다"),
    ("주먹", "주먹을 쥔다"),
]


def collect(cap, tracker, seconds, label, hint, show):
    import cv2

    frames = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = tracker.preprocess(frame)
        obs = tracker.process(frame)
        if obs is not None:
            frames.append(obs)
            if show:
                ht.draw_landmarks(frame, obs)
        if show:
            left = seconds - (time.time() - t0)
            cv2.putText(frame, f"{label}  {left:.0f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(frame, hint, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("diag", frame)
            cv2.waitKey(1)
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--hold", type=float, default=3.0, help="자세당 유지 시간(초)")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    import cv2

    show = not args.no_window
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"카메라 {args.camera} 를 열 수 없다.")
        return 2

    tracker = ht.HandTracker(mirror=args.mirror)
    rt = LeapRetargeter(smoothing=1.0, max_speed=1e9)

    print("=" * 68)
    print("엄지 진단. 안내대로 자세를 잡아 주세요.")
    print("=" * 68)
    print(f"\n[1/{len(POSES) + 1}] 캘리브레이션 — 손을 **펴서** 보여 주세요 ({args.hold:.0f}초)")
    for obs in collect(cap, tracker, args.hold, "calibration", "hold hand OPEN", show):
        rt.observe_calibration(obs.world)
    if not rt.finish_calibration():
        print("  표본 부족. 손이 안 잡혔습니다. 조명/거리를 확인하세요.")
        cap.release()
        return 1
    print(f"  보정각 {np.degrees(rt.thumb_align_angle()):.1f} deg")
    print(f"  LEAP 엄지 안착방향(손바닥계) {np.round(rt.reference.thumb_rest, 3)}")
    print(f"  LEAP 엄지 도달거리 {rt.reference.reach['thumb'] * 1000:.1f} mm,"
          f" 말단마디 {rt.reference.distal_length['thumb'] * 1000:.1f} mm")

    lo = jm.LIMITS_INTERSECTION_MJ_LOWER
    hi = jm.LIMITS_INTERSECTION_MJ_UPPER
    tj = FINGER_JOINTS["thumb"]

    print("\n" + "-" * 68)
    print(f"{'자세':<16} {'배율':>5} {'목표거리':>9} {'손끝잔차':>9} {'재시도':>6}  한계에 붙은 관절")
    print("-" * 68)

    rows = []
    for i, (label, hint) in enumerate(POSES, start=2):
        print(f"\n[{i}/{len(POSES) + 1}] {label} — {hint} ({args.hold:.0f}초)", flush=True)
        frames = collect(cap, tracker, args.hold, label, hint, show)
        if len(frames) < 5:
            print(f"  검출 {len(frames)}프레임 — 건너뜀")
            continue

        # 가운데 프레임들만 쓴다. 자세를 바꾸는 동안이 섞이지 않게.
        sample = frames[len(frames) // 3: 2 * len(frames) // 3] or frames
        scales, dists, resid, restarts, q_last = [], [], [], [], None
        per_finger = []
        for obs in sample:
            rt.reset()
            q_last = rt.retarget(obs.world)
            scales.append(rt.measure_scales(obs.world)["thumb"])
            tg = rt.compute_targets(obs.world)
            dists.append(np.linalg.norm(tg[7] - rt.reference.root["thumb"]))
            err = rt.tip_error()
            resid.append(err[7])
            per_finger.append(err[1::2])
            restarts.append(rt.last_restarts)
        per_finger = np.mean(per_finger, axis=0)

        pinned = [
            jm.MUJOCO_JOINT_NAMES[j]
            for j in tj
            if abs(q_last[j] - lo[j]) < 1e-3 or abs(q_last[j] - hi[j]) < 1e-3
        ]
        rows.append((label, np.mean(scales), np.mean(dists), np.mean(resid),
                     np.mean(restarts), pinned, q_last[tj].copy(), per_finger))

    cap.release()
    tracker.close()
    if show:
        cv2.destroyAllWindows()

    print("\n" + "=" * 68)
    print("결과 — 이 블록을 복사해서 주면 됩니다")
    print("=" * 68)
    print(f"보정각 {np.degrees(rt.thumb_align_angle()):.1f} deg,"
          f" LEAP 엄지 도달거리 {rt.reference.reach['thumb'] * 1000:.1f} mm")
    print(f"{'자세':<16} {'배율':>5} {'목표거리':>9} {'손끝잔차':>9} {'재시도':>6}  한계관절")
    for label, sc, d, r, rs, pinned, q, pf in rows:
        print(f"{label:<16} {sc:>5.2f} {d * 1000:>8.1f}mm {r * 1000:>8.1f}mm {rs:>6.1f}  "
              f"{','.join(pinned) if pinned else '-'}")
    print(f"\n측정 배율(고정값 {'사용중' if rt.frozen_scales else '없음'}):"
          f" {' '.join(f'{f}={rt.frozen_scales[f]:.2f}' for f in FINGERS) if rt.frozen_scales else '-'}")
    print(f"\n손가락별 손끝 잔차 (재시도를 누가 일으키는지)")
    print(f"  {'자세':<14} " + " ".join(f"{f:>9}" for f in FINGERS))
    for label, sc, d, r, rs, pinned, q, pf in rows:
        print(f"  {label:<14} " + " ".join(f"{v * 1000:>8.1f}mm" for v in pf))
    print()
    for label, sc, d, r, rs, pinned, q, pf in rows:
        print(f"  {label:<14} th_cmc/axl/mcp/ipl = {np.round(q, 2)}")

    print("\n읽는 법")
    print("  목표거리 > 도달거리   -> 스케일 문제. --scale 로 줄인다")
    print("  한계관절이 매번 같음  -> 방향이 작업공간 밖. 캘리브레이션을 다시 잡는다")
    print("  재시도 5.0 고정       -> IK 가 목표에 못 간다. 위 둘 중 하나가 원인")
    print("  잔차는 작은데 이상함  -> 목표 자체가 틀린 것. 사람 엄지 랜드마크를 의심")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
