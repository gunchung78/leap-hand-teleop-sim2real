"""sim_node — /leap/joint_cmd -> MuJoCo 디지털 트윈 -> /sim/joint_states.

qpos 를 직접 넣지 않는다. **액추에이터에 명령을 주고 물리를 돌린다.** 충돌과 추종 지연이
보여야 트윈 구실을 한다 (Phase 0 의 벌림 충돌이 이 방식으로 재현됐다).

뷰어는 mujoco.viewer.launch_passive, 물리 스텝은 ROS 타이머. 한 프로세스 안에서 GIL 로
직렬화되므로 충돌하지 않는다. 실기 브리지가 같은 /leap/joint_cmd 를 먹으므로 "명령 토픽
하나가 시뮬과 실기를 동시에 먹인다"가 성립한다.

메시지
  입력  sensor_msgs/JointState /leap/joint_cmd (MuJoCo 관절 순서, rad). 이름으로 맞춘다
  출력  sensor_msgs/JointState /sim/joint_states  position=qpos velocity=qvel
        effort=actuator_force, stamp=now. 로그에 촬영->시뮬 반영 지연(now - cmd.stamp)

파라미터
  model_path  기본 third_party/mujoco_menagerie/leap_hand/scene_right.xml (저장소 기준)
  viewer(true) rate(60 Hz 타이머)  publish_rate(60)
"""

from __future__ import annotations

import os
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState

import leap_hand_mapping
from leap_hand_mapping import joint_map as jm
from leap_teleop import SENSOR_QOS, TOPIC_JOINT_CMD, TOPIC_SIM_STATES

REPO = os.path.dirname(os.path.dirname(os.path.abspath(leap_hand_mapping.__file__)))
DEFAULT_SCENE = os.path.join(REPO, "third_party", "mujoco_menagerie", "leap_hand", "scene_right.xml")


class SimNode(Node):
    def __init__(self) -> None:
        super().__init__("sim_node")
        self.declare_parameter("model_path", DEFAULT_SCENE)
        self.declare_parameter("viewer", True)
        self.declare_parameter("rate", 60.0)
        self.declare_parameter("publish_rate", 60.0)
        self.declare_parameter("scene", "twin")           # twin: menagerie 손만 | cube: playground 학습 장면(손바닥 위 큐브)
        self.declare_parameter("limits", "teleop")        # 이름 없는 명령 클립 표 (teleop | model)
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        import mujoco

        self.mujoco = mujoco
        self.limits = str(p("limits"))
        scene = str(p("scene"))
        if scene == "cube":
            from leap_hand_mapping.cube_scene import SCENE_XML, load_cube_model
            path = str(SCENE_XML)
            self.model = load_cube_model()
        else:
            path = str(p("model_path"))
            if not os.path.exists(path):
                raise FileNotFoundError(f"MuJoCo 모델이 없다: {path} (README 환경 구성의 menagerie clone 참고)")
            self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        if self.model.nkey > 0:
            # 학습 장면: 키프레임 home = 학습 기본 자세 + 큐브 위치. 목표 큐브(mocap)는 학습처럼 숨긴다
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            if self.model.nmocap > 0:
                self.data.mocap_pos[:] = [-100.0, -100.0, -100.0]
        mujoco.mj_forward(self.model, self.data)
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "cube_angvel")
        self._cube_angvel_adr = int(self.model.sensor_adr[sid]) if sid >= 0 else None
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "cube_position")
        self._cube_pos_adr = int(self.model.sensor_adr[sid]) if sid >= 0 else None
        self._angvel_hist = []

        # 명령은 이름으로 매핑한다. 액추에이터 순서가 MuJoCo 관절 순서와 같음을 Phase 0 에서
        # 확인했지만, 그래도 이름으로 맞추면 모델이 바뀌어도 조용히 틀어지지 않는다.
        self._act_index = {}
        for i in range(self.model.nu):
            jid = self.model.actuator_trnid[i, 0]
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            self._act_index[jname] = i
        missing = [n for n in jm.MUJOCO_JOINT_NAMES if n not in self._act_index]
        if missing:
            raise RuntimeError(f"모델에 액추에이터가 없는 관절: {missing}")
        self._qpos_adr = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                          for n in jm.MUJOCO_JOINT_NAMES]
        self._qvel_adr = [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                          for n in jm.MUJOCO_JOINT_NAMES]

        self.viewer = None
        if bool(p("viewer")):
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.sub = self.create_subscription(JointState, TOPIC_JOINT_CMD, self._on_cmd, SENSOR_QOS)
        self.pub = self.create_publisher(JointState, TOPIC_SIM_STATES, SENSOR_QOS)

        self._last_cmd_stamp: Time | None = None
        self._last_cmd_rx = None
        self._lat = []
        self._n_cmd = 0
        self._prev = time.time()
        self._last_log = self._prev
        self.step_timer = self.create_timer(1.0 / float(p("rate")), self._step)
        self.pub_timer = self.create_timer(1.0 / float(p("publish_rate")), self._publish)
        self.get_logger().info(f"{os.path.relpath(path, REPO)}  viewer={p('viewer')}  범위 {self.limits}  {TOPIC_JOINT_CMD} -> ctrl")

    def _on_cmd(self, msg: JointState) -> None:
        q = jm.clip_mujoco(np.array(msg.position, dtype=float), self.limits) if len(msg.name) == 0 else None
        if q is None:
            # 이름으로 맞춘다. 모르는 이름은 무시, 빠진 관절은 직전 값 유지
            for name, pos in zip(msg.name, msg.position):
                i = self._act_index.get(name)
                if i is not None:
                    self.data.ctrl[i] = pos
        else:
            self.data.ctrl[:] = q
        self._last_cmd_stamp = Time.from_msg(msg.header.stamp)
        self._last_cmd_rx = time.time()
        self._n_cmd += 1
        self._lat.append((self.get_clock().now() - self._last_cmd_stamp).nanoseconds * 1e-6)

    def _step(self) -> None:
        if not rclpy.ok():
            return
        now = time.time()
        dt = min(max(now - self._prev, 1e-3), 0.1)
        self._prev = now
        steps = max(1, int(dt / self.model.opt.timestep))
        for _ in range(min(steps, 50)):
            self.mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            if not self.viewer.is_running():
                self.get_logger().info("뷰어가 닫혔다. 종료")
                raise KeyboardInterrupt
            self.viewer.sync()
        if self._cube_angvel_adr is not None:
            self._angvel_hist.append(float(self.data.sensordata[self._cube_angvel_adr + 2]))
        if now - self._last_log >= 5.0:
            lat = f"{np.mean(self._lat[-150:]):.1f} ms" if self._lat else "-"
            cube = ""
            if self._cube_angvel_adr is not None and self._angvel_hist:
                z = float(self.data.sensordata[self._cube_pos_adr + 2]) if self._cube_pos_adr is not None else float("nan")
                cube = f"  큐브 z각속도 {np.mean(self._angvel_hist[-300:]):+.2f} rad/s  높이 {z:+.3f} m"
                if self._cube_pos_adr is not None and z < -0.05:
                    cube += "  (떨어짐)"
            self.get_logger().info(f"명령 {self._n_cmd}  촬영->시뮬 지연 {lat}  접촉 {self.data.ncon}{cube}")
            self._last_log = now

    def _publish(self) -> None:
        if not rclpy.ok():
            return   # SIGINT 뒤 타이머가 한 번 더 도는 경합. 조용히 빠진다
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "leap_mujoco"
        msg.name = list(jm.MUJOCO_JOINT_NAMES)
        msg.position = [float(self.data.qpos[a]) for a in self._qpos_adr]
        msg.velocity = [float(self.data.qvel[a]) for a in self._qvel_adr]
        msg.effort = [float(self.data.actuator_force[self._act_index[n]]) for n in jm.MUJOCO_JOINT_NAMES]
        self.pub.publish(msg)

    def destroy_node(self) -> bool:
        try:
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimNode()
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
