"""실기 연결 전 점검. **모터를 절대 움직이지 않는다.**

토크를 켜지 않고 연결만 해서 다음을 확인한다.

  1. 포트 존재 / 읽기·쓰기 권한
  2. FTDI latency timer (문서 4.4 — 안 낮추면 제어 주파수가 크게 떨어진다)
  3. 4Mbps 로 통신이 되는지
  4. 모터 ID 0~15 가 전부 응답하는지
  5. 현재 관절각과 전류

sweep_joints.py --real 로 손을 움직이기 전에 항상 이걸 먼저 돌릴 것.

사용법:
    python scripts/preflight_real_hand.py
    python scripts/preflight_real_hand.py --port /dev/serial/by-id/usb-FTDI_...
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leap_hand_mapping import joint_map as jm  # noqa: E402
from leap_hand_mapping.real_hand import find_port  # noqa: E402

ADDR_TORQUE_ENABLE = 64
OK, WARN, BAD = "[ OK ]", "[경고]", "[실패]"


def check_port(port: str) -> bool:
    print(f"\n1) 포트 {port}")
    real = os.path.realpath(port)
    if not os.path.exists(real):
        print(f"   {BAD} 장치가 없다. 손 전원과 USB 연결을 확인할 것.")
        return False
    print(f"   {OK} 존재 -> {real}")
    if os.access(real, os.R_OK | os.W_OK):
        print(f"   {OK} 읽기/쓰기 권한 있음")
        return True
    print(f"   {BAD} 권한 없음. 아래 중 하나를 실행할 것.")
    print("        sudo usermod -aG dialout $USER   # 영구적. 재로그인 필요")
    print(f"        sudo chmod 666 {real}            # 임시. 재연결하면 초기화됨")
    return False


def check_latency(port: str) -> None:
    real = os.path.basename(os.path.realpath(port))
    path = f"/sys/bus/usb-serial/devices/{real}/latency_timer"
    print(f"\n2) FTDI latency timer ({path})")
    if not os.path.exists(path):
        print(f"   {WARN} 확인 불가. 건너뛴다.")
        return
    with open(path) as f:
        value = int(f.read().strip())
    if value <= 2:
        print(f"   {OK} {value} ms")
    else:
        print(f"   {WARN} {value} ms — 기본값이라 제어 주파수가 크게 떨어진다 (문서 4.4)")
        print(f"        echo 1 | sudo tee {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--baudrate", type=int, default=4_000_000)
    args = ap.parse_args()

    port = args.port or find_port()
    print("=" * 68)
    print("LEAP Hand Lite 실기 사전 점검 (모터를 움직이지 않는다)")
    print("=" * 68)
    if port is None:
        print(f"\n{BAD} /dev/serial/by-id 에서 포트를 못 찾았다.")
        print("     ls /dev/serial/by-id 로 확인하고 --port 로 직접 지정할 것.")
        return 2
    if not check_port(port):
        return 2
    check_latency(port)

    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "third_party/LEAP_Hand_API/python",
        ),
    )
    from leap_hand_utils.dynamixel_client import DynamixelClient

    motors = list(range(jm.NUM_JOINTS))
    print(f"\n3) {args.baudrate} bps 로 연결")
    client = DynamixelClient(motors, port, args.baudrate)
    try:
        client.connect()
    except OSError as e:
        print(f"   {BAD} {e}")
        print("        - Dynamixel Wizard 가 떠 있으면 포트를 점유한다. 종료할 것 (문서 4.5)")
        print("        - 5V 전원이 들어와 있는지 확인할 것")
        return 2
    print(f"   {OK} 포트 열림 (토크는 꺼진 상태)")

    try:
        # 토크 '비활성화'를 써서 응답 여부만 본다. 모터는 움직이지 않는다.
        print("\n4) 모터 ID 0~15 응답 확인")
        failed = client.write_byte(motors, 0, ADDR_TORQUE_ENABLE)
        alive = [m for m in motors if m not in failed]
        if failed:
            print(f"   {BAD} 무응답 ID: {sorted(failed)}  (응답 {len(alive)}/16)")
            print("        - 데이지체인 배선과 커넥터를 확인할 것")
            print("        - Dynamixel Wizard 로 해당 모터의 ID/baudrate(4Mbps) 확인")
            print("        - 과부하로 빨간 LED 점멸 중이면 전원을 껐다 켜야 복구된다")
        else:
            print(f"   {OK} 16개 전부 응답")

        print("\n5) 현재 상태 (토크 꺼짐 — 손은 힘없이 늘어져 있어야 정상)")
        pos = np.asarray(client.read_pos())
        cur = np.asarray(client.read_cur())
        q_mj = jm.leaphand_to_mujoco(pos)
        lo, hi = jm.LIMITS_INTERSECTION_MJ_LOWER, jm.LIMITS_INTERSECTION_MJ_UPPER
        print(f"\n   {'모터':>4s} {'MuJoCo 관절':<10s} {'실기(rad)':>10s} {'MuJoCo(rad)':>12s} {'전류(mA)':>9s}  범위")
        print("   " + "-" * 68)
        outside = []
        for j in range(jm.NUM_JOINTS):
            mid = int(jm.MUJOCO_TO_MOTOR[j])
            inside = lo[j] <= q_mj[j] <= hi[j]
            if not inside:
                outside.append(j)
            print(
                f"   {mid:>4d} {jm.MUJOCO_JOINT_NAMES[j]:<10s} {pos[mid]:>10.3f}"
                f" {q_mj[j]:>12.3f} {cur[mid]:>9.1f}  {'안' if inside else '범위 밖!'}"
            )
        if outside:
            print(f"\n   {WARN} 범위 밖 관절: {[jm.MUJOCO_JOINT_NAMES[j] for j in outside]}")
            print("        영점 교정이 틀어졌거나 손이 접힌 자세일 수 있다.")
            print("        손을 편 자세로 두고 다시 확인할 것.")
        if np.abs(cur).max() > 50:
            print(f"\n   {WARN} 토크가 꺼져 있는데 전류가 흐른다(최대 {np.abs(cur).max():.0f}mA). 배선을 확인할 것.")

        print("\n" + "=" * 68)
        if not failed and not outside:
            print("점검 통과. 다음 단계:")
            print("  python scripts/sweep_joints.py                     # 시뮬만 먼저")
            print(f"  python scripts/sweep_joints.py --real --port {port} --joints 0")
            print("  (관절 하나부터. 이상 없으면 --joints 없이 전체)")
        else:
            print("점검에서 문제가 나왔다. 위 안내를 먼저 처리할 것.")
        print("=" * 68)
        return 0 if not failed else 1
    finally:
        client.set_torque_enabled(motors, False, retries=0)
        client.disconnect()
        print("\n연결 해제 (토크 꺼짐)")


if __name__ == "__main__":
    raise SystemExit(main())
