"""policy_node — 학습된 rotate_z 정책(npz) 이 /leap/joint_cmd 를 낸다. 사람 손 대신 신경망이 명령한다.

    입력  sensor_msgs/JointState  /sim/joint_states 또는 /real/joint_states (source 파라미터)
    출력  sensor_msgs/JointState  /leap/joint_cmd  (MuJoCo 관절 이름 16, rad) — 텔레옵과 같은 토픽이라
          sim_node / hand_bridge_node 가 그대로 먹는다. 데드맨·합류 램프·전류 동결도 그대로 작동한다.

정책 파일은 scripts/phase2/p2_3_export_policy.py 가 만든 npz (numpy 만으로 추론). 학습 env 와 같은 식:
    관측  [관절각 16, 직전 행동 16]  → 정규화 → MLP(silu) → tanh(loc) = 행동 a ∈ [-1,1]^16
    목표각 = default_pose + action_scale * a           (클립은 sim_node / bridge 가 한다)
    주기   ctrl_dt (20 Hz)

관측 잡음 `noise`(런치 기본 0.05 = 학습값). 0 이면 결정적 정책이 대칭 자세에 정체해 트윈에서 회전이 ≈0 이었다(08-24).
관절 상태가 hold_timeout 동안 안 오면 명령을 멈춘다(정책이 낡은 상태로 돌지 않게).

    ros2 launch leap_teleop policy.launch.py policy:=models/rotate_z_v0.npz          # 트윈(큐브 장면)
    ros2 launch leap_teleop policy.launch.py policy:=... real:=true                   # + 실기 (데드맨 SPACE 없음 → CLI)
"""

from __future__ import annotations

import os
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from leap_hand_mapping import joint_map as jm
from leap_teleop import SENSOR_QOS, TOPIC_ENABLE, TOPIC_JOINT_CMD, TOPIC_REAL_STATES, TOPIC_SIM_STATES

import leap_hand_mapping

REPO = os.path.dirname(os.path.dirname(os.path.abspath(leap_hand_mapping.__file__)))   # pip -e 라 저장소 루트


def numpy_policy(z):
    """p2_3_export_policy.numpy_policy 와 같은 식 (ROS2 환경에 그 스크립트의 의존성이 없어 복제)."""
    n = int(z["n_layers"])
    Ws = [z[f"W{i}"] for i in range(n)]
    bs = [z[f"b{i}"] for i in range(n)]
    mean, std = z["obs_mean"], z["obs_std"]

    def silu(x):
        return x / (1.0 + np.exp(-x))

    def act(obs):
        x = (np.asarray(obs, dtype=np.float64) - mean) / std
        for W, b in zip(Ws[:-1], bs[:-1]):
            x = silu(x @ W + b)
        x = x @ Ws[-1] + bs[-1]
        return np.tanh(x[: x.shape[-1] // 2])

    return act


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_node")
        self.declare_parameter("policy", "")
        self.declare_parameter("source", "sim")          # sim | real
        self.declare_parameter("noise", 0.0)             # 관측 관절각 잡음 (rad, 균일 ±)
        self.declare_parameter("hold_timeout", 0.5)
        self.declare_parameter("require_enable", False)  # true 면 /teleop/enable 이 켜진 동안만 명령
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        path = str(p("policy"))
        if not path:
            raise RuntimeError("policy 파라미터(npz 경로)가 비었다. scripts/phase2/p2_3_export_policy.py 로 만든다")
        if not os.path.isabs(path):
            path = os.path.join(REPO, path)
        z = np.load(path, allow_pickle=False)
        self.policy = numpy_policy(z)
        self.default_pose = np.asarray(z["default_pose"], dtype=float)
        self.action_scale = float(z["action_scale"])
        self.ctrl_dt = float(z["ctrl_dt"])
        names = [str(n) for n in z["joint_names"]]
        if names != list(jm.MUJOCO_JOINT_NAMES):
            raise RuntimeError(f"정책의 관절 순서가 MuJoCo 순서와 다르다: {names}")
        self.noise = float(p("noise"))
        self.hold_timeout = float(p("hold_timeout"))
        self.require_enable = bool(p("require_enable"))

        src = str(p("source"))
        topic = TOPIC_SIM_STATES if src == "sim" else TOPIC_REAL_STATES
        self.q = None
        self.q_rx = None
        self.last_act = np.zeros(jm.NUM_JOINTS)
        self.enabled = not self.require_enable
        self.create_subscription(JointState, topic, self._on_state, SENSOR_QOS)
        self.create_subscription(Bool, TOPIC_ENABLE, self._on_enable, 10)
        self.pub = self.create_publisher(JointState, TOPIC_JOINT_CMD, SENSOR_QOS)
        self.timer = self.create_timer(self.ctrl_dt, self._tick)
        self._n = 0
        self._last_log = time.time()
        self._infer_ms = []
        self.get_logger().info(
            f"{os.path.relpath(path, REPO)}  입력 {topic}  {1 / self.ctrl_dt:.0f} Hz  action_scale {self.action_scale}"
            f"  기본자세 {np.degrees(self.default_pose).round(0).astype(int).tolist()}  잡음 {self.noise}")

    def _on_enable(self, msg: Bool) -> None:
        if self.require_enable:
            self.enabled = bool(msg.data)

    def _on_state(self, msg: JointState) -> None:
        if len(msg.position) != jm.NUM_JOINTS:
            return
        if msg.name and list(msg.name) != list(jm.MUJOCO_JOINT_NAMES):
            idx = {n: i for i, n in enumerate(msg.name)}
            try:
                q = np.array([msg.position[idx[n]] for n in jm.MUJOCO_JOINT_NAMES], dtype=float)
            except KeyError:
                return
        else:
            q = np.array(msg.position, dtype=float)
        self.q = q
        self.q_rx = time.time()

    def _tick(self) -> None:
        if not rclpy.ok() or self.q is None or not self.enabled:
            return
        if time.time() - self.q_rx > self.hold_timeout:
            return                                  # 상태가 끊겼다. 낡은 상태로 정책을 돌리지 않는다
        q = self.q
        if self.noise > 0:
            q = q + np.random.uniform(-1, 1, jm.NUM_JOINTS) * self.noise
        t0 = time.perf_counter()
        a = self.policy(np.concatenate([q, self.last_act]))
        self._infer_ms.append((time.perf_counter() - t0) * 1000)
        self.last_act = a
        target = self.default_pose + self.action_scale * a
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "leap_mujoco/policy"
        msg.name = list(jm.MUJOCO_JOINT_NAMES)
        msg.position = [float(v) for v in target]
        self.pub.publish(msg)
        self._n += 1
        now = time.time()
        if now - self._last_log >= 5.0:
            self.get_logger().info(f"명령 {self._n}  추론 {np.mean(self._infer_ms[-100:]):.2f} ms"
                                   f"  |a| 평균 {np.abs(a).mean():.2f}  rot 목표 deg "
                                   f"{np.degrees(target[[1, 5, 9]]).round(0).astype(int).tolist()}")
            self._last_log = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
