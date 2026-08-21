"""hand_bridge_node — /leap/joint_cmd -> 안전 래퍼 -> /cmd_leap (업스트림 leaphand_node).

실기로 나가는 **유일한 경로**다. 로드맵 Day 4 의 핵심이라 안전 장치를 여기 다 모은다.
LEAP Hand v1 Lite 는 플라스틱 기어(XL330)라 스톨 상태로 밀면 이빨이 나간다(문서 4.1/4.5).

| 장치      | 내용 |
|-----------|------|
| 데드맨    | /teleop/enable (Bool) 이 true 일 때만 /cmd_leap 을 낸다. **기본 false.** false 로
|           | 떨어지면 그 자리에서 멈춘다(영점으로 튀지 않는다) |
| 시작 동기 | enable 되면 먼저 /leap_position 으로 실기 현재 자세를 읽어 거기서부터 램프한다.
|           | 읽기 전에는 명령을 내지 않는다 — 켜자마자 목표로 점프하는 일을 막는다 |
| 범위 클립 | jm.clip_mujoco (MJCF ∩ 실기 규약) |
| 속도 제한 | max_speed rad/s. cmd_rate(60 Hz) 타이머로 목표를 향해 램프 |
| 전류 동결 | /leap_pos_vel_eff 를 poll_rate(30 Hz) 폴링. |effort| > current_warn(300) 인 모터가
|           | 있으면 **명령을 그 자리에 얼린다.** 전부 current_release(250) 아래로 내려오면 푼다 |
| 명령 시효 | /leap/joint_cmd 가 cmd_timeout(2 s) 동안 없으면 마지막 목표를 유지한다(움직이지 않음) |
| 규약 변환 | jm.safe_leaphand_command — MuJoCo 순서 -> 모터 ID 순서 + π 오프셋 |
| 피드백    | 읽은 값을 MuJoCo 규약으로 되돌려 /real/joint_states (effort = 전류, mA 단위 추정) |

서비스 호출은 call_async + done-callback 이다. 콜백 안에서 동기 호출을 하면 단일
스레드 실행기에서 데드락이다. 업스트림 readme 는 조회를 90 samples/s 아래로 권한다.

이 노드는 Dynamixel 을 직접 만지지 않는다. 그건 업스트림 leaphand_node 의 일이고,
거기엔 kP 600 / curr_lim 350 / port 를 런치(real.launch.py)에서 넘긴다.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from leap_hand.srv import LeapPosition, LeapPosVelEff
from leap_hand_mapping import joint_map as jm
from leap_teleop import SENSOR_QOS, TOPIC_ENABLE, TOPIC_JOINT_CMD, TOPIC_REAL_STATES

TOPIC_CMD_LEAP = "/cmd_leap"          # 업스트림 leaphand_node 구독 토픽 (LEAPhand 규약, 모터 ID 순서)
SRV_POSITION = "/leap_position"
SRV_POS_VEL_EFF = "/leap_pos_vel_eff"


class HandBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("hand_bridge_node")
        self.declare_parameter("max_speed", 8.0)
        self.declare_parameter("cmd_rate", 60.0)
        self.declare_parameter("poll_rate", 30.0)
        self.declare_parameter("current_warn", 300.0)
        self.declare_parameter("current_release", 250.0)
        self.declare_parameter("cmd_timeout", 2.0)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.max_speed = float(p("max_speed"))
        self.current_warn = float(p("current_warn"))
        self.current_release = float(p("current_release"))
        self.cmd_timeout = float(p("cmd_timeout"))
        jm.self_check()

        self.target = None           # MuJoCo 순서, 마지막 /leap/joint_cmd
        self.target_rx = None
        self.current = None          # 지금 실기에 보내고 있는 자세 (MuJoCo 순서). None = 아직 동기 안 됨
        self.enabled = False
        self.frozen = False
        self.over = []               # 전류 초과 모터 (MuJoCo 인덱스, mA)
        self._sync_pending = False
        self._poll_pending = False
        self._n_cmd = self._n_frozen = 0
        self._last_log = time.time()

        self.pub_cmd = self.create_publisher(JointState, TOPIC_CMD_LEAP, 10)
        self.pub_state = self.create_publisher(JointState, TOPIC_REAL_STATES, SENSOR_QOS)
        self.create_subscription(JointState, TOPIC_JOINT_CMD, self._on_cmd, SENSOR_QOS)
        self.create_subscription(Bool, TOPIC_ENABLE, self._on_enable, 10)
        self.cli_pos = self.create_client(LeapPosition, SRV_POSITION)
        self.cli_pve = self.create_client(LeapPosVelEff, SRV_POS_VEL_EFF)

        self.create_timer(1.0 / float(p("cmd_rate")), self._cmd_tick)
        self.create_timer(1.0 / float(p("poll_rate")), self._poll_tick)
        self.get_logger().info(
            f"{TOPIC_JOINT_CMD} -> {TOPIC_CMD_LEAP}  데드맨 {TOPIC_ENABLE} (기본 false)"
            f"  max_speed {self.max_speed} rad/s  전류 동결 >{self.current_warn:.0f}"
        )

    # ---- 입력 ----
    def _on_cmd(self, msg: JointState) -> None:
        if len(msg.name) == len(msg.position) and msg.name:
            q = np.zeros(jm.NUM_JOINTS) if self.target is None else self.target.copy()
            idx = {n: i for i, n in enumerate(jm.MUJOCO_JOINT_NAMES)}
            for n, v in zip(msg.name, msg.position):
                if n in idx:
                    q[idx[n]] = v
        elif len(msg.position) == jm.NUM_JOINTS:
            q = np.array(msg.position, dtype=float)
        else:
            self.get_logger().warning("관절 이름/개수가 안 맞는 명령. 버림")
            return
        self.target = jm.clip_mujoco(q)
        self.target_rx = time.time()

    def _on_enable(self, msg: Bool) -> None:
        if msg.data == self.enabled:
            return
        self.enabled = bool(msg.data)
        if self.enabled:
            self.current = None          # 실기 현재 자세부터 다시 동기
            self.get_logger().info("데드맨 ON — 실기 현재 자세를 읽고 거기서부터 램프한다")
        else:
            self.get_logger().info("데드맨 OFF — 그 자리에서 멈춘다")

    # ---- 명령 ----
    def _cmd_tick(self) -> None:
        if not rclpy.ok() or not self.enabled:
            return
        if self.current is None:
            self._request_sync()
            return
        if self.frozen or self.target is None:
            return
        if self.target_rx is not None and time.time() - self.target_rx > self.cmd_timeout:
            return                         # 명령 시효 지남. 마지막 자세 유지
        dt = 1.0 / 60.0
        step = np.clip(self.target - self.current, -self.max_speed * dt, self.max_speed * dt)
        self.current = jm.clip_mujoco(self.current + step)
        self._send(self.current)

    def _send(self, q_mujoco: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "leap_motor"
        msg.name = [str(i) for i in range(jm.NUM_JOINTS)]          # 모터 ID
        msg.position = [float(v) for v in jm.safe_leaphand_command(q_mujoco)]
        self.pub_cmd.publish(msg)
        self._n_cmd += 1

    def _request_sync(self) -> None:
        if self._sync_pending:
            return
        if not self.cli_pos.service_is_ready():
            self._log_throttled("leaphand_node 의 /leap_position 서비스를 기다리는 중")
            return
        self._sync_pending = True
        fut = self.cli_pos.call_async(LeapPosition.Request())
        fut.add_done_callback(self._on_sync)

    def _on_sync(self, fut) -> None:
        self._sync_pending = False
        try:
            pos = np.array(fut.result().position, dtype=float)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warning(f"실기 자세 읽기 실패: {e}")
            return
        if pos.shape != (jm.NUM_JOINTS,):
            self.get_logger().warning(f"실기 자세 길이 {pos.shape}. 무시")
            return
        self.current = jm.clip_mujoco(jm.leaphand_to_mujoco(pos))
        self.get_logger().info(f"실기 자세 동기 완료. 최대 |q| {np.degrees(np.abs(self.current).max()):.1f} deg")

    # ---- 피드백 / 전류 감시 ----
    def _poll_tick(self) -> None:
        if not rclpy.ok() or self._poll_pending:
            return
        if not self.cli_pve.service_is_ready():
            return
        self._poll_pending = True
        fut = self.cli_pve.call_async(LeapPosVelEff.Request())
        fut.add_done_callback(self._on_state)

    def _on_state(self, fut) -> None:
        self._poll_pending = False
        try:
            res = fut.result()
        except Exception as e:  # noqa: BLE001
            self._log_throttled(f"실기 상태 읽기 실패: {e}")
            return
        pos = np.array(res.position, dtype=float)
        vel = np.array(res.velocity, dtype=float)
        eff = np.array(res.effort, dtype=float)
        if pos.shape != (jm.NUM_JOINTS,):
            return
        q = jm.leaphand_to_mujoco(pos)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "leap_mujoco"
        msg.name = list(jm.MUJOCO_JOINT_NAMES)
        msg.position = [float(v) for v in q]
        if vel.shape == (jm.NUM_JOINTS,):
            msg.velocity = [float(v) for v in jm.motor_to_mujoco_order(vel)]
        if eff.shape == (jm.NUM_JOINTS,):
            eff_mj = jm.motor_to_mujoco_order(eff)
            msg.effort = [float(v) for v in eff_mj]
            self._check_current(np.abs(eff_mj))
        self.pub_state.publish(msg)

        now = time.time()
        if now - self._last_log >= 5.0:
            state = "OFF" if not self.enabled else ("FROZEN" if self.frozen else "ON")
            self.get_logger().info(f"데드맨 {state}  명령 {self._n_cmd}  동결 {self._n_frozen}회"
                                   + (f"  전류초과 {self.over}" if self.over else ""))
            self._last_log = now

    def _check_current(self, cur: np.ndarray) -> None:
        over = [(jm.MUJOCO_JOINT_NAMES[i], round(float(cur[i]))) for i in np.where(cur > self.current_warn)[0]]
        if over and not self.frozen:
            self.frozen = True
            self._n_frozen += 1
            self.get_logger().warning(f"전류 초과 {over} — 명령 동결. 손을 빼서 자세를 풀면 자동 해제")
        elif self.frozen and np.all(cur < self.current_release):
            self.frozen = False
            self.get_logger().info("전류 정상. 동결 해제")
        self.over = over

    def _log_throttled(self, text: str) -> None:
        now = time.time()
        if now - self._last_log >= 5.0:
            self.get_logger().info(text)
            self._last_log = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HandBridgeNode()
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
