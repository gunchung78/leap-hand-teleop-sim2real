"""웹캠 손 추적만 따로 확인한다. 로봇은 쓰지 않는다.

텔레오퍼레이션이 안 될 때 원인이 추적인지 리타겟팅인지 가르는 데 쓴다.
먼저 이걸 통과시키고 p1_3_teleop_mujoco.py 로 갈 것.

    python scripts/phase1/p1_1_check_hand_tracking.py

확인할 것
--------
1. 화면 좌상단에 손이 "Right" 로 잡히는가.
   오른손을 들었는데 "Right" 가 아니면 영상이 거울상이다. --mirror 를 켠다.
   MediaPipe 는 거울상 영상에서 손 좌우를 반대로 판정하고, 이때 world 랜드마크의
   손대칭도 같이 뒤집혀서 리타겟팅이 손등 쪽으로 굽는 거울상이 된다.
2. 손바닥 폭이 40mm 근처로 안정적인가.
   이 값이 로봇으로 보내는 스케일의 분모다. 프레임마다 크게 흔들리면
   조명이나 거리를 조절할 것.
3. 손가락을 하나씩 굽힐 때 빨간 점(리타겟팅에 실제로 쓰는 8개)이 따라오는가.

--no-window 를 주면 창 없이 수치만 출력한다(원격 접속 등).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from leap_hand_mapping import hand_tracker as ht  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hand", default="Right", choices=["Right", "Left"])
    ap.add_argument("--mirror", action="store_true", help="영상 좌우 반전")
    ap.add_argument("--no-window", action="store_true", help="창 없이 수치만")
    ap.add_argument("--seconds", type=float, default=0.0, help="지정 시간 뒤 자동 종료 (0=무한)")
    args = ap.parse_args()

    import cv2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"카메라 {args.camera} 를 열 수 없다. ls /dev/video* 로 확인할 것.")
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    tracker = ht.HandTracker(handedness=args.hand, mirror=args.mirror)
    print(f"카메라 {args.camera}, {args.hand} 손 추적. q 또는 Ctrl-C 로 종료.")

    widths: list[float] = []
    frames = detected = 0
    start = last_report = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임을 못 읽었다")
                break
            frame = tracker.preprocess(frame)
            obs = tracker.process(frame)
            frames += 1

            if obs is not None:
                detected += 1
                widths.append(obs.palm_width())
                if not args.no_window:
                    ht.draw_landmarks(frame, obs)

            now = time.time()
            if now - last_report >= 1.0:
                rate = frames / (now - start)
                if widths:
                    w = np.array(widths[-30:]) * 1000
                    print(f"{rate:5.1f} fps  검출 {detected}/{frames}"
                          f"  손바닥 폭 {w.mean():5.1f} +- {w.std():4.1f} mm")
                else:
                    print(f"{rate:5.1f} fps  검출 {detected}/{frames}  (손이 안 잡힘)")
                last_report = now

            if not args.no_window:
                label = (f"{obs.handedness} {obs.score:.2f}  palm {obs.palm_width()*1000:.0f}mm"
                         if obs else "no hand")
                color = (0, 200, 0) if obs else (0, 0, 255)
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow("hand tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.seconds and now - start >= args.seconds:
                break
    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        cap.release()
        tracker.close()
        if not args.no_window:
            cv2.destroyAllWindows()

    if frames:
        print(f"\n프레임 {frames}, 검출 {detected} ({detected / frames:.0%})")
    if widths:
        w = np.array(widths) * 1000
        print(f"손바닥 폭 평균 {w.mean():.1f} mm, 표준편차 {w.std():.1f} mm")
        print(f"LEAP 손바닥 폭이 90.9 mm 이므로 스케일은 약 {90.9 / w.mean():.2f} 배가 된다.")
    elif detected == 0:
        print("손이 한 번도 안 잡혔다. 조명, 손이 화면에 다 들어오는지, --mirror 를 확인할 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
