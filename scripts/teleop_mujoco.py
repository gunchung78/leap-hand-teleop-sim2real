"""웹캠으로 LEAP Hand 를 텔레오퍼레이션한다. Phase 1 본체.

    웹캠 -> MediaPipe 21 랜드마크 -> 손바닥 좌표계 정규화 + LEAP 치수 스케일링
         -> PyBullet IK -> MuJoCo 디지털 트윈 (-> 선택적으로 실기)

기본은 **시뮬레이터만** 움직인다. 실기는 --real 을 줘야 붙는다.
문서 8장 로드맵의 Day 3 이 시뮬 전용이라 그렇게 맞췄다.

사용법
------
    python scripts/check_hand_tracking.py     # 먼저 추적부터 확인
    python scripts/teleop_mujoco.py           # 시뮬만
    python scripts/teleop_mujoco.py --pybullet-gui   # IK 목표점까지 같이 본다
    python scripts/teleop_mujoco.py --real           # 실기까지

손이 안 잡히면 로봇은 **직전 자세를 유지**한다. 영점으로 튀지 않는다.
사람이 프레임 밖으로 나갈 때마다 손이 확 펴지면 물건을 놓치기 때문이다.
--release-after 초 이상 놓치면 그때는 천천히 영점으로 돌아간다.

실기 주의(문서 4.5)
------------------
  - Dynamixel Wizard 가 떠 있으면 포트를 점유해 연결이 안 된다.
  - 과부하로 빨간 LED 가 점멸하면 전원을 껐다 켜야 한다.
  - 전류가 임계를 넘으면 이 스크립트는 **명령을 그 자리에 얼린다**. 손을 빼서
    자세를 풀면 자동으로 다시 따라간다.
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
from leap_hand_mapping.retarget import LeapRetargeter  # noqa: E402
from leap_hand_mapping.retarget_dex import DexRetargeter  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_SCENE = os.path.join(REPO, "third_party/mujoco_menagerie/leap_hand/scene_right.xml")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hand", default="Right", choices=["Right", "Left"])
    ap.add_argument("--mirror", action="store_true",
                    help="영상 좌우 반전. 오른손이 Right 로 안 잡히면 켤 것")

    ap.add_argument("--retargeter", default="dex", choices=["dex", "ours"],
                    help="dex-retargeting(기본) 또는 직접 구현한 위치 IK(비교용)")
    ap.add_argument("--dex-type", default="dexpilot", choices=["dexpilot", "vector"],
                    help="dexpilot 은 손끝끼리의 거리까지 목표에 넣는다 (집기에 유리)")
    ap.add_argument("--dex-scale", type=float, default=None,
                    help="dex-retargeting 의 scaling_factor 덮어쓰기 (기본 1.6, upstream 값)")
    ap.add_argument("--dex-alpha", type=float, default=None,
                    help="dex-retargeting low-pass 계수 덮어쓰기 (기본 0.2, 작을수록 부드럽다)")

    ap.add_argument("--scale", type=float, default=None,
                    help="[--retargeter ours] 자동 배율에 곱하는 배수")
    ap.add_argument("--smoothing", type=float, default=0.4,
                    help="지수 평활 계수. 1이면 평활 없음, 작을수록 부드럽고 느리다")
    ap.add_argument("--max-speed", type=float, default=8.0, help="관절 속도 상한 (rad/s)")
    ap.add_argument("--distal-mode", default="leap", choices=["leap", "scaled"],
                    help="앞마디 목표를 LEAP 말단 마디 길이로 되짚을지, 사람에서 스케일할지")
    ap.add_argument("--dip-weight", type=float, default=0.3,
                    help="앞마디 목표의 가중치. 손끝은 항상 1.0 이다")
    ap.add_argument("--calib-frames", type=int, default=30,
                    help="엄지 정렬 캘리브레이션에 쓸 프레임 수 (0=건너뜀)")

    ap.add_argument("--real", action="store_true", help="실기도 함께 구동")
    ap.add_argument("--port", default=None)
    ap.add_argument("--kp", type=int, default=600, help="진동하면 400 부근으로 (문서 4.2)")

    ap.add_argument("--pybullet-gui", action="store_true", help="IK 목표점을 PyBullet 창으로")
    ap.add_argument("--no-viewer", action="store_true", help="MuJoCo 뷰어 끄기")
    ap.add_argument("--no-window", action="store_true", help="카메라 창 끄기")
    ap.add_argument("--release-after", type=float, default=1.5,
                    help="손을 이 시간(초) 이상 놓치면 천천히 영점으로 (0=유지)")
    ap.add_argument("--seconds", type=float, default=0.0, help="지정 시간 뒤 자동 종료")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    import cv2
    import mujoco

    jm.self_check()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"카메라 {args.camera} 를 열 수 없다. ls /dev/video* 로 확인할 것.")
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    tracker = ht.HandTracker(handedness=args.hand, mirror=args.mirror)
    use_dex = args.retargeter == "dex"
    if use_dex:
        retargeter = DexRetargeter(
            hand_type=args.hand,
            retargeting_type=args.dex_type,
            scaling_factor=args.dex_scale,
            low_pass_alpha=args.dex_alpha,
            max_speed=args.max_speed,
        )
        print(f"리타겟팅: dex-retargeting ({args.dex_type})")
        print(f"  설정 {os.path.basename(retargeter.config_path)}")
    else:
        retargeter = LeapRetargeter(
            gui=args.pybullet_gui,
            scale=args.scale,
            distal_mode=args.distal_mode,
            smoothing=args.smoothing,
            max_speed=args.max_speed,
            dip_weight=args.dip_weight,
        )
        print("리타겟팅: 직접 구현 (손끝 위치 IK)")

    model = mujoco.MjModel.from_xml_path(MJCF_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    hand = None
    if args.real:
        from leap_hand_mapping.real_hand import LeapHandDriver, find_port

        port = args.port or find_port()
        if port is None:
            print("U2D2 포트를 못 찾았다. ls /dev/serial/by-id 로 확인하고 --port 로 지정할 것.")
            return 2
        print(f"실기 연결: {port} (kP={args.kp})")
        hand = LeapHandDriver(port=port, kp=args.kp)
        time.sleep(0.5)

    viewer = None
    if not args.no_viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)

    # dex-retargeting 은 손목 프레임을 매 프레임 추정하고 벡터를 맞추므로
    # 사람마다 다른 엄지 안착 각도가 문제되지 않는다. 캘리브레이션이 필요 없다.
    if args.calib_frames and not use_dex:
        # 한때 자세를 두 개(편 손 + 엄지 접기) 받아 정렬 회전의 roll 까지 정하려 했다.
        # roll 이 미결정인 것은 사실이지만, 실측 손 녹화로 A/B 한 결과 결과가 거의
        # 안 바뀌었다(엄지-검지 간격 154.2 -> 150.0mm, 엄지 잔차 12.2 -> 13.3mm).
        # 사람 엄지는 CMC 에서 보면 손바닥에 붙여도 방향이 5.8 도밖에 안 변해서
        # 두 번째 자세가 관측량으로 약하다. 사용자에게 자세를 더 시킬 값어치가 없다.
        # 근거는 retarget.finish_calibration 주석 참고.
        print(f"\n[엄지 정렬] 손을 **펴서** 카메라에 보여 주세요. {args.calib_frames} 프레임 모읍니다.")
        print("  사람 엄지가 손바닥 대비 놓인 방향은 LEAP 과 크게 다르고 사람마다도 다릅니다.")
        collected = 0
        t_calib = time.time()
        while collected < args.calib_frames and time.time() - t_calib < 30:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = tracker.preprocess(frame)
            obs = tracker.process(frame)
            if obs is not None:
                retargeter.observe_calibration(obs.world)
                collected += 1
                if not args.no_window:
                    ht.draw_landmarks(frame, obs)
            if not args.no_window:
                cv2.putText(frame, f"calibrating thumb {collected}/{args.calib_frames}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.imshow("teleop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if retargeter.finish_calibration():
            print(f"  완료. 보정각 {np.degrees(retargeter.thumb_align_angle()):.1f} deg\n")
        else:
            print("  표본이 부족해 건너뜁니다. 엄지 정확도가 떨어집니다.\n")

    print("손을 카메라에 보이면 따라간다. q 또는 Ctrl-C 로 종료.")
    print(f"MuJoCo 시나리오: {os.path.relpath(MJCF_SCENE, REPO)}")

    q_cmd = np.zeros(jm.NUM_JOINTS)
    frames = detected = frozen = 0
    lost_since: float | None = None
    over_current: list = []
    tip_errors: list[float] = []
    loop_ms: list[float] = []
    restarts: list[int] = []
    jitter: list[float] = []   # 프레임 사이 관절각 변화. 진짜 지터인지 보는 지표
    start = last_report = time.time()
    prev = start

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임을 못 읽었다")
                break
            now = time.time()
            dt = min(max(now - prev, 1e-3), 0.1)
            prev = now
            frames += 1

            frame = tracker.preprocess(frame)
            t0 = time.time()
            obs = tracker.process(frame)

            if obs is not None:
                detected += 1
                lost_since = None
                q_prev = q_cmd
                q_cmd = retargeter.retarget(obs.world, dt=dt)
                tip_errors.append(
                    float(np.nanmean(retargeter.tip_error()))
                    if use_dex else float(retargeter.tip_error()[1::2].mean())
                )

                restarts.append(retargeter.last_restarts)
                jitter.append(float(np.abs(q_cmd - q_prev).max()))
                if not args.no_window:
                    ht.draw_landmarks(frame, obs)
            else:
                # 손을 놓치면 직전 자세를 유지한다. 오래 놓치면 천천히 영점으로.
                if lost_since is None:
                    lost_since = now
                elif args.release_after and now - lost_since > args.release_after:
                    q_cmd = jm.clip_mujoco(q_cmd * max(0.0, 1.0 - 1.5 * dt))
                    retargeter.set_pose(q_cmd)
            loop_ms.append((time.time() - t0) * 1000)

            # 디지털 트윈. 위치 액추에이터에 명령만 주고 물리를 돌린다.
            # qpos 를 직접 넣지 않는 이유는 충돌과 추종 지연이 보여야 트윈 구실을
            # 하기 때문이다(Phase 0 의 벌림 충돌이 이 방식으로 재현됐다).
            data.ctrl[:] = q_cmd
            steps = max(1, int(dt / model.opt.timestep))
            for _ in range(min(steps, 50)):
                mujoco.mj_step(model, data)

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()

            if hand is not None:
                over_current = hand.check_current()
                if over_current:
                    frozen += 1
                    # 명령을 얼린다. 스톨 상태에서 계속 밀면 Lite 기어가 상한다.
                else:
                    hand.command_mujoco(q_cmd)

            if now - last_report >= 1.0:
                fps = frames / (now - start)
                tip = np.mean(tip_errors[-30:]) * 1000 if tip_errors else float("nan")
                jit = np.degrees(np.mean(jitter[-30:])) if jitter else float("nan")
                rst = np.mean(restarts[-30:]) if restarts else 0.0
                # dex 는 벡터를 맞추므로 mm 잔차가 같은 뜻이 아니다. 라벨을 구분한다.
                cost = (f"벡터오차 {tip:5.2f} mm" if use_dex
                        else f"손끝잔차 {tip:5.2f} mm  재시도 {rst:3.1f}")
                msg = (f"{fps:5.1f} fps  검출 {detected}/{frames}"
                       f"  {cost}  지터 {jit:5.2f} deg"
                       f"  처리 {np.mean(loop_ms[-30:]):4.1f} ms  접촉 {data.ncon:2d}")
                if over_current:
                    msg += f"  ! 전류 초과 {over_current}"
                print(msg)
                last_report = now

            if not args.no_window:
                status = f"{obs.handedness} {obs.score:.2f}" if obs else "no hand"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 0) if obs else (0, 0, 255), 2)
                cv2.imshow("teleop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.seconds and now - start >= args.seconds:
                break
    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        cap.release()
        tracker.close()
        retargeter.close()
        if not args.no_window:
            cv2.destroyAllWindows()
        if hand is not None:
            hand.command_mujoco(np.zeros(jm.NUM_JOINTS))
            time.sleep(0.5)
            hand.disable()
        if viewer is not None:
            try:
                if viewer.is_running():
                    viewer.close()
            except Exception:
                pass

    elapsed = time.time() - start
    print(f"\n프레임 {frames}, {elapsed:.1f}초, 평균 {frames / max(elapsed, 1e-6):.1f} fps")
    if frames:
        print(f"손 검출률 {detected / frames:.0%}")
    if tip_errors and use_dex:
        t = np.array(tip_errors) * 1000
        print(f"목표 벡터 오차 평균 {t.mean():.2f} mm, 최대 {t.max():.2f} mm"
              f" (dexpilot 이면 손끝간 거리 6개 + 손목->손끝 4개)")
    elif tip_errors:
        t = np.array(tip_errors) * 1000
        print(f"IK 손끝 잔차 평균 {t.mean():.2f} mm, 최대 {t.max():.2f} mm (앞마디는 보조 목표라 제외)")
    if jitter:
        j = np.degrees(np.array(jitter))
        print(f"프레임간 관절각 변화 평균 {j.mean():.2f} deg, 95% {np.percentile(j, 95):.2f} deg,"
              f" 최대 {j.max():.2f} deg")
    if loop_ms:
        print(f"추적+리타겟 처리 평균 {np.mean(loop_ms):.1f} ms, 95% {np.percentile(loop_ms, 95):.1f} ms")
    if restarts and not use_dex:
        print(f"IK 재시도 평균 {np.mean(restarts):.2f}회/프레임 (0 이면 첫 판에 다 풀렸다는 뜻)")
    if hand is not None and frozen:
        print(f"전류 초과로 명령을 얼린 프레임 {frozen}회")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
