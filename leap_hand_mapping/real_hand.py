"""LEAP Hand v1 **Lite** 실기 드라이버.

LEAP_Hand_API/python/main.py 의 LeapNode 를 그대로 쓰지 않는 이유는 두 가지다.

1. LeapNode 는 포트를 '/dev/ttyUSB0' -> '/dev/ttyUSB1' -> 'COM13' 순으로 하드코딩해서
   시도한다. 인수인계 문서 4.3 은 USB 포트를 바꿔도 안 변하는
   /dev/serial/by-id/<고정ID> 를 쓰라고 한다.
2. Lite 모터 보호를 위해 전류 상시 감시가 필요하다(문서 4.5).

전류 제한은 350 으로 고정한다. 문서 4.1 이 강조하듯 Lite 는 엔지니어링 플라스틱
기어라서 550(Full 값)으로 올리면 기어 이빨이 파손된다. 이 값은 인자로 노출하지 않는다.

Dynamixel 컨트롤 테이블 주소는 main.py 와 동일하다.
    11 = Operating Mode (5 = current-based position control)
    80 = D gain, 82 = I gain, 84 = P gain
   102 = Goal Current
"""

from __future__ import annotations

import os
import sys

import numpy as np

from leap_hand_mapping import joint_map as jm

# LEAP_Hand_API 의 python/ 를 import 경로에 넣는다.
# dynamixel_client 가 'from leap_hand_utils...' 형태로 자기 패키지를 찾기 때문이다.
_API_PY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party/LEAP_Hand_API/python",
)

# Lite 전용. 절대 올리지 말 것 (문서 4.1).
CURRENT_LIMIT_LITE = 350

# 벌림(MCP Side) 모터. 게인을 나머지보다 낮게 준다. main.py 와 동일.
SIDE_MOTORS = [0, 4, 8]


class LeapHandDriver:
    """실기 LEAP Hand 제어. 각도 입출력은 모두 **MuJoCo 관절 순서**를 쓴다.

    모터 ID 순서와 pi 오프셋 변환은 이 클래스가 내부에서 처리하므로,
    바깥 코드(텔레오퍼레이션 노드, RL 정책 배포)는 시뮬레이터 규약만 알면 된다.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 4_000_000,
        kp: int = 600,
        ki: int = 0,
        kd: int = 200,
        current_warn: float = 300.0,
    ) -> None:
        if _API_PY not in sys.path:
            sys.path.insert(0, _API_PY)
        from leap_hand_utils.dynamixel_client import DynamixelClient

        self.motors = list(range(jm.NUM_JOINTS))
        self.current_warn = current_warn
        self.client = DynamixelClient(self.motors, port, baudrate)
        self.client.connect()

        n = np.ones(len(self.motors))
        self.client.sync_write(self.motors, n * 5, 11, 1)  # current-based position 모드
        self.client.set_torque_enabled(self.motors, True)
        self.client.sync_write(self.motors, n * kp, 84, 2)
        self.client.sync_write(SIDE_MOTORS, np.ones(3) * (kp * 0.75), 84, 2)
        self.client.sync_write(self.motors, n * ki, 82, 2)
        self.client.sync_write(self.motors, n * kd, 80, 2)
        self.client.sync_write(SIDE_MOTORS, np.ones(3) * (kd * 0.75), 80, 2)
        self.client.sync_write(self.motors, n * CURRENT_LIMIT_LITE, 102, 2)

        self.command_mujoco(np.zeros(jm.NUM_JOINTS))

    def command_mujoco(self, q_mujoco) -> np.ndarray:
        """MuJoCo 순서의 관절각을 실기에 명령. 범위 클립 후 전송한다."""
        target = jm.safe_leaphand_command(q_mujoco)
        self.client.write_desired_pos(self.motors, target)
        return target

    def read_mujoco(self) -> np.ndarray:
        """실기 관절각을 읽어 MuJoCo 순서/규약으로 변환해 반환."""
        return jm.leaphand_to_mujoco(self.client.read_pos())

    def read_current(self) -> np.ndarray:
        """모터 전류(mA). 모터 ID 순서 그대로."""
        return np.asarray(self.client.read_cur())

    def check_current(self) -> list[tuple[int, float]]:
        """경고 임계를 넘은 (모터ID, 전류) 목록. 비어 있으면 정상."""
        cur = np.abs(self.read_current())
        return [(i, float(cur[i])) for i in np.where(cur > self.current_warn)[0]]

    def disable(self) -> None:
        self.client.set_torque_enabled(self.motors, False)

    def __enter__(self) -> "LeapHandDriver":
        return self

    def __exit__(self, *exc) -> None:
        self.disable()


def find_port() -> str | None:
    """/dev/serial/by-id 에서 U2D2 로 보이는 포트를 찾는다 (문서 4.3)."""
    by_id = "/dev/serial/by-id"
    if not os.path.isdir(by_id):
        return None
    entries = sorted(os.listdir(by_id))
    # U2D2 는 FTDI 단일 채널 컨버터로 잡힌다. Dual RS232-HS 는 보통 다른 장치다.
    for name in entries:
        if "FTDI" in name and "Dual" not in name:
            return os.path.join(by_id, name)
    return os.path.join(by_id, entries[0]) if entries else None
