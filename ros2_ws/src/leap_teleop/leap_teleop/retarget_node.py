"""retarget_node — /hand/landmarks -> /leap/joint_cmd (JointState, MuJoCo 관절 순서).

리타겟터는 leap_hand_mapping.retarget.LeapRetargeter (직접 구현한 손끝 위치 IK, 커밋
7cccfdd 상태)다. 라이브로 비교한 모든 버전 중 이것이 가장 잘 됐다(README 머리 참고).
이 노드는 그것을 호출만 한다 — 알고리즘은 ROS 를 모른다.

메시지
  입력  sensor_msgs/PointCloud2 21점 (tracker_node)
  출력  sensor_msgs/JointState  name = MUJOCO_JOINT_NAMES, position = 16개 rad,
        header.stamp = **입력 stamp 그대로** (촬영 시각 전파)

손 유실 처리
  hold_timeout(1.5 s) 까지는 직전 자세를 유지한다. 프레임 밖으로 나갈 때마다 손이
  펴지면 물건을 놓치기 때문이다. 넘어가면 release_time(1 s) 에 걸쳐 영점으로 램프하고,
  램프 중에는 이 노드가 타이머로 명령을 낸다(그때 stamp 는 now — 합성 명령이다).

파라미터
  smoothing(0.4) max_speed(8.0 rad/s) distal_mode("leap") scale(0.0=자동)
  hold_timeout(1.5) release_time(1.0)
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from leap_hand_mapping import joint_map as jm
from leap_hand_mapping.retarget import LeapRetargeter
from leap_teleop import SENSOR_QOS, TOPIC_JOINT_CMD, TOPIC_LANDMARKS


class RetargetNode(Node):
    def __init__(self) -> None:
        super().__init__("retarget_node")
        self.declare_parameter("smoothing", 0.4)
        self.declare_parameter("max_speed", 8.0)
        self.declare_parameter("distal_mode", "leap")
        self.declare_parameter("scale", 0.0)
        self.declare_parameter("hold_timeout", 1.5)
        self.declare_parameter("release_time", 1.0)
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        jm.self_check()
        scale = float(p("scale")) or None
        self.rt = LeapRetargeter(
            gui=False,
            scale=scale,
            distal_mode=str(p("distal_mode")),
            smoothing=float(p("smoothing")),
            max_speed=float(p("max_speed")),
        )
        self.hold_timeout = float(p("hold_timeout"))
        self.release_time = float(p("release_time"))

        self.pub = self.create_publisher(JointState, TOPIC_JOINT_CMD, SENSOR_QOS)
        self.sub = self.create_subscription(PointCloud2, TOPIC_LANDMARKS, self._on_landmarks, SENSOR_QOS)

        self.q = np.zeros(jm.NUM_JOINTS)
        self._last_stamp: Time | None = None      # 마지막 입력의 stamp
        self._last_rx = None                       # 마지막 입력을 받은 벽시계
        self._releasing = False
        self._n = 0
        self._lat = []
        self._ms = []
        self._last_log = time.time()
        self.release_timer = self.create_timer(1.0 / 30.0, self._release_tick)
        self.get_logger().info(
            f"{TOPIC_LANDMARKS} -> {TOPIC_JOINT_CMD}  smoothing={p('smoothing')}"
            f" max_speed={p('max_speed')} distal_mode={p('distal_mode')}"
        )

    def _on_landmarks(self, msg: PointCloud2) -> None:
        pts = pc2.read_points_numpy(msg, field_names=("x", "y", "z")).astype(float)
        if pts.shape != (21, 3):
            self.get_logger().warning(f"랜드마크 모양이 {pts.shape}, 21x3 이어야 한다. 버림")
            return
        stamp = Time.from_msg(msg.header.stamp)
        dt = 1.0 / 30.0
        if self._last_stamp is not None:
            dt = min(max((stamp - self._last_stamp).nanoseconds * 1e-9, 1e-3), 0.1)
        self._last_stamp = stamp
        self._last_rx = time.time()
        self._releasing = False

        t0 = time.perf_counter()
        self.q = self.rt.retarget(pts, dt=dt)
        self._ms.append((time.perf_counter() - t0) * 1000)
        self._publish(self.q, msg.header.stamp)

        self._lat.append((self.get_clock().now() - stamp).nanoseconds * 1e-6)
        self._n += 1
        now = time.time()
        if now - self._last_log >= 5.0:
            self.get_logger().info(
                f"명령 {self._n}  리타겟 {np.mean(self._ms[-150:]):.1f} ms"
                f"  촬영->명령 지연 {np.mean(self._lat[-150:]):.1f} ms"
                f"  손끝잔차 {np.mean(self.rt.tip_error()[1::2]) * 1000:.1f} mm"
            )
            self._last_log = now

    def _release_tick(self) -> None:
        """손을 오래 놓치면 천천히 영점으로. 그 전까지는 아무것도 안 한다(하류가 직전 자세 유지)."""
        if not rclpy.ok() or self._last_rx is None:
            return
        gone = time.time() - self._last_rx
        if gone <= self.hold_timeout:
            return
        if not self._releasing:
            self._releasing = True
            self.get_logger().info(f"손 유실 {gone:.1f}s — {self.release_time:.1f}s 에 걸쳐 영점으로")
        # 1/30 s 마다 남은 거리의 일정 비율만큼 줄인다. release_time 뒤 대략 영점.
        k = min(1.0, (1.0 / 30.0) / max(self.release_time, 1e-3))
        self.q = jm.clip_mujoco(self.q * (1.0 - k))
        if np.abs(self.q).max() < 1e-3:
            self.q[:] = 0.0
            self._last_rx = None     # 영점 도달. 다음 입력까지 조용히
        self.rt.set_pose(self.q)
        self._publish(self.q, self.get_clock().now().to_msg())

    def _publish(self, q: np.ndarray, stamp) -> None:
        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = "leap_mujoco"
        msg.name = list(jm.MUJOCO_JOINT_NAMES)
        msg.position = [float(v) for v in q]
        self.pub.publish(msg)

    def destroy_node(self) -> bool:
        try:
            self.rt.close()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RetargetNode()
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
