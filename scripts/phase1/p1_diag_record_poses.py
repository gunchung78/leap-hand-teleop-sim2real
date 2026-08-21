#!/usr/bin/env python3
"""사람 손 자세를 **깨끗하게** 녹화한다. 모든 오프라인 진단의 입력 데이터.

왜 다시 만드는가
--------------
예전 녹화(p1_diag_thumb.py --save)는 타이머로 돌았다. "핀치 3초" 라고 띄우고 바로
세기 시작하니, 사용자가 자세를 잡는 **도중**이 그대로 들어갔다. 실제로 확인하면:

    검지-엄지 핀치  94프레임   간격 시계열  50 98 97 98 96 96 95 93 94 93 51 21 23 23 24 ...
    엄지만 붙임     94프레임   간격 시계열  75 74 74 74 74 74 74 74 73 67 55 53 52 52 ...

핀치 녹화의 2/3 는 손을 펴고 있었고(95mm), 진짜 핀치는 마지막 1/3 뿐이다(23mm).
그 평균이 "사람 핀치 57.2mm" 가 되어 README 표에 올라갔다. 로봇 쪽 숫자도 같은
프레임을 평균했으니 핀치 행은 사실상 "손을 펴는 중" 이었다. **사람 데이터가 틀리면
그 위의 모든 비교가 틀린다.** 이 스크립트는 그것을 막는다.

어떻게 다른가
-----------
- 타이머가 없다. 자세를 잡고 **스페이스바**를 눌러야 녹화가 시작된다.
- 화면에 MediaPipe 가 지금 읽는 값을 실시간으로 띄운다: 엄지-검지 간격(mm),
  손가락 굽힘각, 안정도. 사용자가 "센서가 내 손을 어떻게 보는지" 를 보면서 자세를
  잡을 수 있다. 핀치인데 60mm 로 보이면 그 자리에서 고쳐 잡는다.
- 녹화 중 손이 움직인 프레임(프레임간 평균 이동 > 임계)은 버린다. 안정 프레임이
  목표 수만큼 모일 때까지 받는다.
- 자세가 하나 더 있다: **편 손 회전**. 모양은 고정한 채 손목만 돌린다. 이 자세에서
  관절각이 흔들리면 센서/손목 프레임 문제고, 안 흔들리면 리타겟팅 문제다.
  센서 탓인지 알고리즘 탓인지를 가르는 자세라 일부러 넣었다.

저장 형식은 예전과 같다(calib_rest / calib_fold / pose3.. / pose_labels). 기존 진단
스크립트가 그대로 읽는다.

    python scripts/phase1/p1_diag_record_poses.py            # thumb_capture.npz 에 저장
    python scripts/phase1/p1_diag_pose_fidelity.py           # 그 파일로 진단
    python scripts/phase1/p1_diag_thumb_options.py
    python scripts/phase1/p1_diag_compare_retargeters.py

키:  SPACE 녹화 시작   r 이 자세 다시   s 이 자세 건너뜀   q 저장 없이 종료
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from leap_hand_mapping import hand_tracker as ht  # noqa: E402

# (라벨, 화면 안내, 안정 자세인가). 안정 자세는 움직인 프레임을 버리고, 움직이는
# 자세(회전)는 전부 받는다.
POSES = [
    ("편 손", "hand OPEN, palm to camera", True),
    ("엄지만 붙임", "4 fingers open, THUMB folded to palm", True),
    ("검지-엄지 핀치", "PINCH: thumb tip touches index tip, others open", True),
    ("주먹", "FIST", True),
    ("편 손 회전", "hand OPEN, slowly ROTATE wrist left/right, keep shape", False),
]

# 검지/중지/약지의 (MCP, PIP, DIP, TIP)
FINGERS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16)]


def gap_mm(w: np.ndarray) -> float:
    return float(np.linalg.norm(w[ht.THUMB_TIP] - w[ht.INDEX_TIP]) * 1000)


def bend_deg(w: np.ndarray) -> tuple[float, float, float]:
    """세 손가락 평균 MCP/PIP/DIP 굽힘(deg). p1_diag_pose_fidelity 와 같은 정의."""
    n = np.cross(w[5] - w[0], w[17] - w[0])
    n /= np.linalg.norm(n)

    def between(a, b):
        return np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))

    out = []
    for M, P, D, T in FINGERS:
        m = w[M] - w[0]
        m /= np.linalg.norm(m)
        v = w[P] - w[M]
        out.append((np.arctan2(v @ n, v @ m),
                    between(w[P] - w[M], w[D] - w[P]),
                    between(w[D] - w[P], w[T] - w[D])))
    return tuple(np.degrees(np.mean(out, axis=0)))


def motion_mm(prev: np.ndarray | None, cur: np.ndarray) -> float:
    if prev is None:
        return 0.0
    return float(np.linalg.norm(cur - prev, axis=1).mean() * 1000)


def put(frame, text, y, color=(255, 255, 255), scale=0.6):
    import cv2
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def record_pose(cap, tracker, label, hint, stable, args) -> np.ndarray | None:
    """자세 하나를 받는다. 반환: (N,21,3) world 랜드마크, s 로 건너뛰면 None, q 면 예외."""
    import cv2

    target = args.frames if stable else args.moving_frames
    recent_gap: list[float] = []
    recent_mot: list[float] = []
    prev = None
    recording = False
    kept: list[np.ndarray] = []
    dropped = 0
    t_start = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = tracker.preprocess(frame)
        obs = tracker.process(frame)
        w = None
        if obs is not None:
            w = np.asarray(obs.world, dtype=float)
            g = gap_mm(w)
            mot = motion_mm(prev, w)
            prev = w
            recent_gap.append(g)
            recent_mot.append(mot)
            recent_gap = recent_gap[-10:]
            recent_mot = recent_mot[-10:]
            ht.draw_landmarks(frame, obs)
        else:
            prev = None

        is_stable = (len(recent_mot) >= 10 and np.mean(recent_mot) < args.still_mm
                     and np.std(recent_gap) < 2.0)

        # --- 화면 ---
        put(frame, f"[{label}]  {hint}", 28, (0, 200, 255), 0.7)
        if w is not None:
            mcp, pip, dip = bend_deg(w)
            put(frame, f"thumb-index gap {g:6.1f} mm   motion {mot:4.1f} mm/f", 58)
            put(frame, f"bend  mcp {mcp:5.1f}  pip {pip:5.1f}  dip {dip:5.1f} deg", 84)
            col = (0, 220, 0) if is_stable else (0, 120, 255)
            put(frame, "STABLE" if is_stable else "moving", 112, col, 0.8)
        else:
            put(frame, "no hand", 58, (0, 0, 255), 0.8)

        if recording:
            put(frame, f"REC {len(kept)}/{target}  dropped {dropped}", 150, (0, 0, 255), 0.9)
        else:
            put(frame, "SPACE=record   r=redo   s=skip   q=quit", 150, (200, 200, 200), 0.55)
            if stable:
                put(frame, "hold still until STABLE, then SPACE", 175, (200, 200, 200), 0.55)
            else:
                put(frame, "press SPACE, then rotate slowly", 175, (200, 200, 200), 0.55)
        cv2.imshow("record", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            raise KeyboardInterrupt
        if key == ord("s"):
            return None
        if key == ord("r"):
            recording, kept, dropped = False, [], 0
            continue
        if key == ord(" ") and not recording:
            recording, kept, dropped = True, [], 0
            t_start = time.time()
            continue

        if recording:
            if w is not None:
                if stable and mot > args.still_mm * 2.5:
                    dropped += 1
                else:
                    kept.append(w)
            if len(kept) >= target:
                return np.array(kept)
            if time.time() - t_start > args.timeout:
                print(f"  {args.timeout:.0f}초 안에 {target}프레임을 못 모았다 "
                      f"({len(kept)}개, 버림 {dropped}). 그만큼만 쓴다.")
                return np.array(kept) if kept else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--hand", default="Right", choices=["Right", "Left"])
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--save", default="thumb_capture.npz")
    ap.add_argument("--frames", type=int, default=60, help="안정 자세당 모을 프레임 수")
    ap.add_argument("--moving-frames", type=int, default=150, help="회전 자세 프레임 수")
    ap.add_argument("--still-mm", type=float, default=1.0,
                    help="이 값(프레임간 평균 이동 mm) 아래면 '안정'으로 본다")
    ap.add_argument("--timeout", type=float, default=20.0, help="자세당 녹화 제한 시간(초)")
    args = ap.parse_args()

    import cv2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"카메라 {args.camera} 를 열 수 없다. ls /dev/video* 로 확인할 것.")
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    tracker = ht.HandTracker(handedness=args.hand, mirror=args.mirror)

    if os.path.exists(args.save):
        print(f"주의: {args.save} 가 이미 있다. 끝까지 가면 덮어쓴다. q 로 빠지면 안 건드린다.")

    print("=" * 70)
    print("자세 녹화. 화면의 gap/bend 는 MediaPipe 가 지금 읽는 값이다.")
    print("자세를 잡고 STABLE 이 뜨면 SPACE. r 다시 / s 건너뜀 / q 저장 없이 종료")
    print("=" * 70)

    rec: dict = {}
    labels: list[str] = []
    summary = []
    try:
        for i, (label, hint, stable) in enumerate(POSES):
            print(f"\n[{i + 1}/{len(POSES)}] {label} — {hint}", flush=True)
            frames = record_pose(cap, tracker, label, hint, stable, args)
            if frames is None or len(frames) == 0:
                print("  건너뜀")
                continue
            gaps = np.array([gap_mm(w) for w in frames])
            mots = np.array([motion_mm(frames[k - 1], frames[k]) for k in range(1, len(frames))])
            bends = np.array([bend_deg(w) for w in frames]).mean(axis=0)
            print(f"  {len(frames)}프레임  간격 {gaps.mean():.1f}±{gaps.std():.1f} mm"
                  f"  이동 {mots.mean():.2f} mm/f  mcp/pip/dip {bends[0]:.0f}/{bends[1]:.0f}/{bends[2]:.0f} deg")
            summary.append((label, len(frames), gaps.mean(), gaps.std(), mots.mean(), bends))
            rec[f"pose{len(labels) + 3}"] = frames
            labels.append(label)
    except KeyboardInterrupt:
        print("\n중단. 저장하지 않는다.")
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
        return 1

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()

    if not labels:
        print("받은 자세가 없다.")
        return 1

    # 예전 형식과 맞춘다. 캘리브레이션 표본은 편 손 / 엄지 붙임 자세를 그대로 쓴다.
    rec["calib_rest"] = rec.get("pose3", np.zeros((0, 21, 3))) if labels[0] == "편 손" \
        else np.zeros((0, 21, 3))
    fold_idx = labels.index("엄지만 붙임") if "엄지만 붙임" in labels else None
    rec["calib_fold"] = rec[f"pose{fold_idx + 3}"] if fold_idx is not None else np.zeros((0, 21, 3))
    rec["pose_labels"] = np.array(labels)
    np.savez_compressed(args.save, **rec)

    print("\n" + "-" * 70)
    print(f"{'자세':<14}{'프레임':>6}{'간격mm':>10}{'±':>6}{'이동mm/f':>9}{'mcp':>6}{'pip':>6}{'dip':>6}")
    for label, n, gm, gs, mm, b in summary:
        print(f"{label:<14}{n:>6}{gm:>10.1f}{gs:>6.1f}{mm:>9.2f}{b[0]:>6.0f}{b[1]:>6.0f}{b[2]:>6.0f}")
    print("-" * 70)
    print(f"{args.save} 에 저장했다. 이 블록을 그대로 붙여 주면 된다.")
    print("읽는 법: 핀치 간격이 20mm 안팎이어야 한다. 50mm 를 넘으면 손끝이 안 닿았거나")
    print("         MediaPipe 가 가려진 엄지를 놓친 것이다. r 로 다시 찍는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
