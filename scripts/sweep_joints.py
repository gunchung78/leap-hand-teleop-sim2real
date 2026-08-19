"""관절을 하나씩 순차 구동해 MuJoCo 와 실기를 육안 대조한다.

인수인계 문서 5장이 제시한 검증 절차 그대로다.

    "실기와 MuJoCo에 동일한 16차원 각도 벡터를 넣고 한 관절씩 순차 구동하여 육안 대조.
     매핑이 틀리면 즉시 드러난다."

scripts/verify_mapping_fk.py 가 기하학적으로는 이미 매핑을 확정했지만, 그것은
어디까지나 URDF 모델끼리의 비교다. 실물 배선/모터 ID 가 도면대로인지는 실기를
직접 돌려 봐야 알 수 있다. 이 스크립트가 그 마지막 한 칸을 채운다.

사용법:
    # 시뮬레이터만 (실기 없이 먼저 확인)
    python scripts/sweep_joints.py

    # 실기까지 동시 구동
    python scripts/sweep_joints.py --real

    # 특정 관절만
    python scripts/sweep_joints.py --real --joints 12 13 14 15

실기 주의(문서 4.5):
  - Dynamixel Wizard 가 떠 있으면 포트를 점유해 연결이 안 된다. 먼저 종료할 것.
  - 과부하로 빨간 LED 가 점멸하면 전원을 껐다 켜야 복구된다.
  - 이 스크립트는 전류를 상시 감시하다가 임계를 넘으면 해당 관절 구동을 중단한다.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leap_hand_mapping import joint_map as jm  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_SCENE = os.path.join(REPO, "third_party/mujoco_menagerie/leap_hand/scene_right.xml")


# 벌림(rot) 축은 손을 편 자세에서 훑으면 옆 손가락과 부딪힌다.
# MuJoCo 충돌 검출로 재 보면 평평한 자세에서는 ±0.14 rad(7.7도)까지밖에 못 벌린다.
# 실기에서 모터 0/4/8 만 전류 임계를 넘긴 것도 이 때문이었다(스톨).
#
# 훑는 손가락 자신의 pip/dip 를 굽혀 두면 옆 손가락 위를 지나가므로 범위가 열린다.
# 굽힘량별 충돌 없는 최대 진폭(세 손가락 공통):
#     0.0 -> 0.14 rad     0.8 -> 0.27 rad     1.2 -> 0.59 rad
#     0.4 -> 0.20 rad     1.0 -> 0.41 rad     1.4 -> 1.047 rad (전 범위)
# 1.4 에서 전 범위가 열리므로 여유를 둬 1.5 를 쓴다. pip/dip 상한(1.885/2.042) 안이다.
ROT_TO_PIP_DIP = {1: (2, 3), 5: (6, 7), 9: (10, 11)}
CLEARANCE_CURL = 1.5


def clearance_posture(j: int) -> np.ndarray:
    """관절 j 를 훑기 전에 잡아 둘 기준 자세."""
    q = np.zeros(jm.NUM_JOINTS)
    if j in ROT_TO_PIP_DIP:
        pip, dip = ROT_TO_PIP_DIP[j]
        q[pip] = q[dip] = CLEARANCE_CURL
    return q


def sweep_profile(lo: float, hi: float, amplitude: float, steps: int) -> np.ndarray:
    """0 -> +a -> 0 -> -a -> 0 왕복 궤적. 관절 범위 안으로 잘라 낸다."""
    hi = min(hi, amplitude)
    lo = max(lo, -amplitude)
    quarter = np.linspace(0.0, 1.0, steps)
    return np.concatenate(
        [hi * quarter, hi * quarter[::-1], lo * quarter, lo * quarter[::-1]]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="실기도 함께 구동")
    ap.add_argument("--port", default=None, help="U2D2 포트 (기본: 자동 탐색)")
    ap.add_argument("--kp", type=int, default=600, help="Lite 에서 진동하면 400 부근으로 (문서 4.2)")
    ap.add_argument("--joints", type=int, nargs="*", default=None, help="MuJoCo 관절 인덱스")
    ap.add_argument("--amplitude", type=float, default=0.6, help="최대 진폭 (rad)")
    ap.add_argument("--steps", type=int, default=40, help="1/4 구간당 스텝 수")
    ap.add_argument("--rate", type=float, default=50.0, help="명령 주파수 (Hz)")
    ap.add_argument("--no-viewer", action="store_true", help="뷰어 없이 실행")
    args = ap.parse_args()

    import mujoco

    jm.self_check()
    model = mujoco.MjModel.from_xml_path(MJCF_SCENE)
    data = mujoco.MjData(model)

    targets = args.joints if args.joints else list(range(jm.NUM_JOINTS))
    bad = [j for j in targets if not 0 <= j < jm.NUM_JOINTS]
    if bad:
        print(f"관절 인덱스는 0~15 여야 한다. 잘못된 값: {bad}")
        return 2

    hand = None
    if args.real:
        from leap_hand_mapping.real_hand import LeapHandDriver, find_port

        port = args.port or find_port()
        if port is None:
            print("U2D2 포트를 못 찾았다. ls /dev/serial/by-id 로 확인하고 --port 로 지정할 것.")
            return 2
        print(f"실기 연결: {port} (전류 제한 {350}mA 고정, kP={args.kp})")
        hand = LeapHandDriver(port=port, kp=args.kp)
        time.sleep(0.5)

    viewer = None
    if not args.no_viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)

    dt = 1.0 / args.rate
    errors: dict[int, float] = {}
    home = np.zeros(jm.NUM_JOINTS)
    current_q = home.copy()

    def apply(q: np.ndarray) -> bool:
        """시뮬과 실기에 같은 자세를 적용. 전류 임계를 넘으면 True."""
        nonlocal current_q
        q = jm.clip_mujoco(q)
        current_q = q
        data.qpos[:] = q
        data.ctrl[:] = q
        mujoco.mj_forward(model, data)
        if viewer is not None:
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
        if hand is not None:
            hand.command_mujoco(q)
            over = hand.check_current()
            if over:
                print(f"     ! 전류 임계 초과 {over}")
                return True
        time.sleep(dt)
        return False

    def ramp_to(q_target: np.ndarray, seconds: float = 1.0) -> bool:
        """현재 자세에서 목표 자세로 부드럽게 이동. 급격한 점프를 막는다."""
        start = current_q.copy()
        for a in np.linspace(0.0, 1.0, max(2, int(seconds * args.rate))):
            if apply(start + a * (q_target - start)):
                return True
        return False

    try:
        for j in targets:
            lo = jm.LIMITS_INTERSECTION_MJ_LOWER[j]
            hi = jm.LIMITS_INTERSECTION_MJ_UPPER[j]
            motor = jm.MUJOCO_TO_MOTOR[j]
            print(
                f"\n[{j:2d}] MuJoCo '{jm.MUJOCO_JOINT_NAMES[j]}'  <->  실기 모터 ID {motor}"
                f"   범위 [{lo:+.3f},{hi:+.3f}]"
            )
            print("     이 관절 하나만 움직인다. 시뮬과 실기가 같은 관절인지 눈으로 확인할 것.")

            base = clearance_posture(j)
            if j in ROT_TO_PIP_DIP:
                print(
                    f"     벌림 축이라 이 손가락의 pip/dip 를 {CLEARANCE_CURL} rad 굽혀 두고 훑는다."
                    " (편 자세로는 7.7도만 넘어도 옆 손가락과 부딪힌다)"
                )
            if ramp_to(base):
                print("     자세를 잡는 중 전류 초과 -> 이 관절 건너뜀")
                ramp_to(home)
                continue

            tracked = []
            aborted = False
            for value in sweep_profile(lo, hi, args.amplitude, args.steps):
                q = base.copy()
                q[j] = value
                if apply(q):
                    print("     -> 이 관절 구동 중단")
                    aborted = True
                    break
                if hand is not None:
                    tracked.append(abs(hand.read_mujoco()[j] - jm.clip_mujoco(q)[j]))

            ramp_to(home)
            time.sleep(0.3)
            if tracked:
                errors[j] = float(np.mean(tracked))
                print(
                    f"     평균 추종 오차 {np.degrees(errors[j]):.2f} deg"
                    f"{'  (중단됨)' if aborted else ''}"
                )
    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        if hand is not None:
            hand.command_mujoco(np.zeros(jm.NUM_JOINTS))
            time.sleep(0.3)
            hand.disable()
        if viewer is not None:
            # 사용자가 창을 닫았거나 이미 종료된 뒤면 close() 가 GLFW 오류를 낸다.
            try:
                if viewer.is_running():
                    viewer.close()
            except Exception:
                pass

    if errors:
        vals = np.degrees(np.array(list(errors.values())))
        print(f"\n[실기 추종 오차] 관절 {len(errors)}개, 평균 {vals.mean():.2f} deg, 최대 {vals.max():.2f} deg")
        worst = max(errors, key=errors.get)
        print(f"  최악: {jm.MUJOCO_JOINT_NAMES[worst]} (모터 {jm.MUJOCO_TO_MOTOR[worst]}) {np.degrees(errors[worst]):.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
