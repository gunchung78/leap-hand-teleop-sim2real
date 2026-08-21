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

떨림 (지터)
  정지한 손에서도 리타겟 출력이 프레임마다 0.5~1도(최대 4도) 흔들린다 — MediaPipe 랜드마크
  노이즈 + IK 가 매 프레임 재시도 시드 사이를 오가는 탓이다. 실기는 kP 600 으로 그걸 전부
  따라가서 눈에 띄게 떤다. 두 손잡이:
    smoothing   리타겟터의 지수 평활. 0.4 -> 0.2 로 내리면 정지 지터가 절반(녹화 실측 1.05 -> 0.55도)
    deadband_deg 출력 데드밴드. 직전에 보낸 명령과의 최대 관절 차이가 이 값보다 작으면 **직전
                명령을 그대로 다시 보낸다.** 정지 떨림은 0 이 되고, 그 이상 움직이면 그대로 따라간다.
                기본 0.5도. 0 이면 끔
  시뮬과 실기가 같은 명령을 받게 하려고 여기(리타겟 출력)에서 거른다. 브리지에서 거르면
  트윈과 실기가 달라진다.

파라미터
  smoothing(0.4) max_speed(8.0 rad/s) distal_mode("leap") scale(0.0=자동)
  hold_timeout(1.5) release_time(1.0) deadband_deg(0.5) restart_mm(1.0) pip_target(false) tip_mode(realtip)
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
        self.declare_parameter("deadband_deg", 0.5)
        # IK 재시도 임계(mm). 7cccfdd 기본 1 mm 는 실제 손에서 절대 못 미치는 값이라(정상 잔차 ~5 mm)
        # 매 프레임 시드 5개를 전부 돌고, 프레임마다 이기는 해가 갈려 관절각이 튄다. 15 로 올리면
        # 재시도 0, 잔차 그대로, 처리 35 -> 6 ms (f118b58 의 실측). 알고리즘은 그대로, 인자만 노출
        self.declare_parameter("restart_mm", 1.0)
        # PIP 관절점을 목표에 추가 (손가락당 3점). 정지 떨림 1.11 -> 0.33도 (녹화 실측). 기본 꺼짐
        self.declare_parameter("pip_target", False)
        # 손끝점 정의: realtip(패드 접촉점, 축에서 20도 이탈) | axis(손가락 축 위 점). 기본 realtip
        self.declare_parameter("tip_mode", "realtip")
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.deadband = np.radians(float(p("deadband_deg")))
        self._q_sent = None
        self._held = 0

        jm.self_check()
        scale = float(p("scale")) or None
        self.rt = LeapRetargeter(
            gui=False,
            scale=scale,
            distal_mode=str(p("distal_mode")),
            smoothing=float(p("smoothing")),
            max_speed=float(p("max_speed")),
            restart_threshold=float(p("restart_mm")) / 1000.0,
            pip_target=bool(p("pip_target")),
            tip_mode=str(p("tip_mode")),
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
            f" max_speed={p('max_speed')} distal_mode={p('distal_mode')} pip_target={p('pip_target')}"
            f" tip_mode={p('tip_mode')} 목표점 {len(self.rt.ee_spec)}"
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
        q_out = self.q
        if (self.deadband > 0 and self._q_sent is not None
                and np.abs(self.q - self._q_sent).max() < self.deadband):
            q_out = self._q_sent            # 데드밴드 안: 직전 명령 유지 (정지 떨림 제거)
            self._held += 1
        self._publish(q_out, msg.header.stamp)

        self._lat.append((self.get_clock().now() - stamp).nanoseconds * 1e-6)
        self._n += 1
        now = time.time()
        if now - self._last_log >= 5.0:
            self.get_logger().info(
                f"명령 {self._n}  리타겟 {np.mean(self._ms[-150:]):.1f} ms"
                f"  촬영->명령 지연 {np.mean(self._lat[-150:]):.1f} ms"
                f"  손끝잔차 {np.mean(self.rt.tip_error()[self.rt.tip_index]) * 1000:.1f} mm"
                f"  데드밴드 유지 {self._held}/{self._n}  재시도 {self.rt.last_restarts}"
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
        self._publish(self.q, self.get_clock().now().to_msg(), frame_id="leap_mujoco/release")

    def _publish(self, q: np.ndarray, stamp, frame_id: str = "leap_mujoco") -> None:
        # frame_id 로 "추적에서 나온 명령"과 "손 유실 램프(합성)"를 구분한다. 지표 스크립트가
        # 지연을 잴 때 합성 명령(stamp=now)을 빼기 위해서다.
        self._q_sent = np.array(q, dtype=float, copy=True)
        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
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
    except Exception:
        # launch 가 SIGINT 를 보내면 컨텍스트가 먼저 닫혀 spin 이 RCLError 를 던질 수 있다.
        # 정상 종료 경합이면 삼키고, 살아 있는데 난 예외면 그대로 올린다.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
