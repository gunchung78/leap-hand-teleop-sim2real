"""dex-retargeting 의 scaling_factor 를 실측으로 고른다.

왜 필요한가
----------
dex-retargeting 의 LEAP 설정은 `scaling_factor: 1.6` 을 기본으로 쓴다. 이 값은
사람 손목->손끝 벡터를 1.6배 해서 로봇의 목표로 삼는다는 뜻인데, 손 크기와
MediaPipe 의 추정 스케일에 따라 적정값이 달라진다. 너무 크면 목표가 로봇 도달거리를
넘어가서 손가락이 늘 뻗은 채 포화되고, 너무 작으면 덜 움직인다.

이 스크립트는 한 번 촬영해 둔 실제 랜드마크로 여러 배율을 **같은 데이터에** 돌려
비교한다. 촬영이 한 번이라 배율끼리 조건이 같다.

    python scripts/phase1/p1_diag_tune_dex_scale.py

촬영 중에는 편 손 -> 주먹 -> 편 손 을 천천히 반복한다. 가동 범위 전체를 봐야
배율이 큰 자세에서만 포화되는지 알 수 있다.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from leap_hand_mapping import hand_tracker as ht  # noqa: E402
from leap_hand_mapping.retarget_dex import DexRetargeter  # noqa: E402

FINGER_TIPS = [("엄지", 4, "thumb_tip_head"), ("검지", 8, "index_tip_head"),
               ("중지", 12, "middle_tip_head"), ("약지", 16, "ring_tip_head")]


def capture(camera: int, seconds: float, mirror: bool, show: bool) -> list:
    import cv2

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"카메라 {camera} 를 열 수 없다.")
        return []
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    tracker = ht.HandTracker(mirror=mirror)

    print(f"촬영 {seconds:.0f}초. 편 손 -> 주먹 -> 편 손 을 천천히 반복하세요.", flush=True)
    frames = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = tracker.preprocess(frame)
        obs = tracker.process(frame)
        if obs is not None:
            frames.append(obs.world.copy())
            if show:
                ht.draw_landmarks(frame, obs)
        if show:
            left = seconds - (time.time() - t0)
            cv2.putText(frame, f"open -> fist -> open   {left:.0f}s  ({len(frames)})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.imshow("tune", frame)
            cv2.waitKey(1)
    cap.release()
    tracker.close()
    if show:
        cv2.destroyAllWindows()
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--scales", type=float, nargs="*",
                    default=[0.8, 1.0, 1.2, 1.4, 1.6])
    args = ap.parse_args()

    frames = capture(args.camera, args.seconds, args.mirror, not args.no_window)
    print(f"검출 {len(frames)} 프레임")
    if len(frames) < 20:
        print("표본이 부족하다. 손이 화면에 계속 보이게 하고 다시 실행할 것.")
        return 1

    probe = DexRetargeter(retargeting_type="vector", max_speed=1e9)
    mano = [probe.to_mano_frame(w) for w in frames]

    print("\n=== 치수 ===")
    probe._ensure_fk()
    p, client, uid, joint_index, _ = probe._fk
    for j in joint_index.values():
        try:
            p.resetJointState(uid, j, 0.0, physicsClientId=client)
        except Exception:
            pass
    for name, idx, link in FINGER_TIPS:
        human = np.array([np.linalg.norm(m[idx] - m[0]) for m in mano]) * 1000
        robot = np.linalg.norm(probe._link_position(link)) * 1000
        # 사람의 최대(가장 뻗었을 때)를 로봇 도달거리에 맞추는 배율
        fit = robot / max(human.max(), 1e-6)
        print(f"  {name}  사람 손목->손끝 평균 {human.mean():5.1f} / 최대 {human.max():5.1f} mm"
              f"   LEAP {robot:5.1f} mm   포화 없는 배율 <= {fit:.2f}")
    probe.close()

    print("\n=== 배율별 ===")
    print(f"{'방식':<9} {'배율':>5} {'벡터오차 평균':>13} {'최대':>9} {'지터':>8} {'가동폭':>8}")
    best = None
    for kind in ("vector", "dexpilot"):
        for scale in args.scales:
            d = DexRetargeter(retargeting_type=kind, scaling_factor=scale, max_speed=8.0)
            d.reset()
            errs, jit, qs, prev = [], [], [], None
            for world in frames:
                q = d.retarget(world, dt=1 / 30)
                errs.append(np.nanmean(d.tip_error()))
                qs.append(q)
                if prev is not None:
                    jit.append(np.abs(q - prev).max())
                prev = q
            d.close()
            qs = np.array(qs)
            # 가동폭: 관절이 실제로 얼마나 움직였나. 포화되면 작아진다.
            span = np.degrees(np.mean(qs.max(axis=0) - qs.min(axis=0)))
            err = np.mean(errs) * 1000
            print(f"{kind:<9} {scale:>5.1f} {err:>12.2f}mm "
                  f"{np.max(errs) * 1000:>8.1f}mm {np.degrees(np.mean(jit)):>7.2f}d {span:>7.1f}d")
            score = err - span   # 오차는 낮고 가동폭은 큰 쪽
            if best is None or score < best[0]:
                best = (score, kind, scale, err, span)

    print(f"\n추천: --dex-type {best[1]} --dex-scale {best[2]}"
          f"  (벡터오차 {best[3]:.1f}mm, 가동폭 {best[4]:.1f}deg)")
    print("가동폭이 배율을 올려도 안 커지면 그 지점부터 포화된 것이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
