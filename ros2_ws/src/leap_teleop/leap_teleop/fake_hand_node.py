"""fake_hand_node — 업스트림 leaphand_node 의 **인터페이스만** 흉내내는 시험용 더미.

실기 없이 hand_bridge_node 와 real.launch.py 의 배선을 검증하려고 만들었다. Dynamixel 을
만지지 않는다. 업스트림과 같은 토픽/서비스 이름과 규약(LEAPhand 규약, 모터 ID 순서,
편 손 = π)을 쓴다:

    구독  /cmd_leap            sensor_msgs/JointState
    서비스 /leap_position       leap_hand/LeapPosition
    서비스 /leap_pos_vel_eff    leap_hand/LeapPosVelEff

동작: 받은 명령을 1차 지연(tau)으로 따라가는 자세를 돌려준다. 전류(effort)는
`fake_current` 파라미터 값을 모든 모터에 돌려준다 — 300 이상으로 올리면 브리지의
전류 동결이 발동하는지 볼 수 있다.

    ros2 launch leap_teleop real.launch.py fake:=true
    ros2 param set /fake_hand_node fake_current 400.0     # 동결 시험
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState

from leap_hand.srv import LeapPosition, LeapPosVelEff
from leap_hand_mapping import joint_map as jm


class FakeHandNode(Node):
    def __init__(self) -> None:
        super().__init__("fake_hand_node")
        self.declare_parameter("tau", 0.08)
        self.declare_parameter("fake_current", 0.0)
        self.pos = np.full(jm.NUM_JOINTS, jm.SIM_TO_REAL_OFFSET)   # 편 손 = π (LEAPhand 규약)
        self.goal = self.pos.copy()
        self.vel = np.zeros(jm.NUM_JOINTS)
        self._t = time.time()
        self.create_subscription(JointState, "/cmd_leap", self._on_cmd, 10)
        self.create_service(LeapPosition, "/leap_position", self._srv_pos)
        self.create_service(LeapPosVelEff, "/leap_pos_vel_eff", self._srv_pve)
        self.create_timer(1.0 / 200.0, self._step)
        self.get_logger().info("가짜 실기. /cmd_leap 구독, /leap_position /leap_pos_vel_eff 서비스 (Dynamixel 없음)")

    def _on_cmd(self, msg: JointState) -> None:
        if len(msg.position) == jm.NUM_JOINTS:
            self.goal = np.array(msg.position, dtype=float)

    def _step(self) -> None:
        now = time.time()
        dt = min(now - self._t, 0.05)
        self._t = now
        tau = max(float(self.get_parameter("tau").value), 1e-3)
        new = self.pos + (self.goal - self.pos) * min(1.0, dt / tau)
        self.vel = (new - self.pos) / max(dt, 1e-6)
        self.pos = new

    def _srv_pos(self, req, res):
        res.position = self.pos.tolist()
        return res

    def _srv_pve(self, req, res):
        res.position = self.pos.tolist()
        res.velocity = self.vel.tolist()
        res.effort = [float(self.get_parameter("fake_current").value)] * jm.NUM_JOINTS
        return res


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakeHandNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
