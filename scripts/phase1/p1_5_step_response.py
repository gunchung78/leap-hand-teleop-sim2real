#!/usr/bin/env python3
"""p1_5_step_response.py — 관절별 계단 응답: 지연 · 상승 시간 · 정상상태 오차. 카메라 없이.

README 가 지적한 문제("명령 직후 바로 읽어 통신 지연이 섞인다")를 정면으로 푼다.
계단 명령을 /leap/joint_cmd 로 내고 /real/joint_states (또는 /sim/joint_states) 를 들으면서

    지연      명령 시각 -> 측정값이 스텝의 10% 를 넘는 시각
    상승 시간 10% -> 90%
    정상상태 오차  hold 끝에서 |목표 - 측정| (deg)

을 관절마다 잰다. 실기면 데드맨을 켜야 움직인다.

    ros2 launch leap_teleop real.launch.py fake:=true sim:=true show:=false   # 또는 실기
    ros2 topic pub --once /teleop/enable std_msgs/Bool "data: true"
    python scripts/phase1/p1_5_step_response.py --source real --step-deg 20 --hold 1.5

tracker/retarget 노드가 같이 떠 있으면 손이 보일 때 명령이 섞인다. 손을 치우거나
sim.launch 없이 브리지+실기만 띄울 것. 안전: 스텝은 영점 기준 +step_deg 하나, 관절 하나씩.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState

from leap_hand_mapping import joint_map as jm

QOS = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)


class Stepper(Node):
    def __init__(self, source: str) -> None:
        super().__init__("step_response")
        self.pub = self.create_publisher(JointState, "/leap/joint_cmd", QOS)
        self.create_subscription(JointState, f"/{source}/joint_states", self._on, QOS)
        self.samples: list[tuple[float, np.ndarray]] = []

    def _on(self, msg: JointState) -> None:
        if len(msg.position) == jm.NUM_JOINTS:
            self.samples.append((time.time(), np.array(msg.position)))

    def send(self, q: np.ndarray) -> None:
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = list(jm.MUJOCO_JOINT_NAMES)
        m.position = [float(v) for v in q]
        self.pub.publish(m)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="real", choices=["real", "sim"])
    ap.add_argument("--step-deg", type=float, default=20.0)
    ap.add_argument("--hold", type=float, default=1.5, help="스텝 뒤 유지 시간(초)")
    ap.add_argument("--joints", nargs="*", default=None, help="관절 이름 일부만 (기본 16개 전부)")
    args = ap.parse_args()

    rclpy.init()
    node = Stepper(args.source)
    names = args.joints or list(jm.MUJOCO_JOINT_NAMES)
    step = np.radians(args.step_deg)

    def spin(sec: float) -> None:
        t = time.time()
        while time.time() - t < sec:
            rclpy.spin_once(node, timeout_sec=0.02)
            node.send(q_now)

    q_now = np.zeros(jm.NUM_JOINTS)
    print(f"소스 /{args.source}/joint_states  스텝 {args.step_deg:.0f} deg  유지 {args.hold:.1f}s")
    spin(1.0)
    if not node.samples:
        print("상태 토픽이 안 들어온다. 런치(와 데드맨)를 확인할 것.")
        return 1

    print(f"\n{'관절':<8}{'지연ms':>8}{'상승ms':>8}{'정상오차deg':>12}{'표본':>6}")
    print("-" * 44)
    rows = []
    for name in names:
        j = jm.MUJOCO_JOINT_NAMES.index(name)
        target = np.clip(step, jm.LIMITS_INTERSECTION_MJ_LOWER[j], jm.LIMITS_INTERSECTION_MJ_UPPER[j])
        q_now = np.zeros(jm.NUM_JOINTS)
        spin(1.0)                                   # 영점에서 안정
        node.samples.clear()
        q_now = q_now.copy(); q_now[j] = target
        t_cmd = time.time()
        spin(args.hold)
        s = [(t - t_cmd, q[j]) for t, q in node.samples]
        q_now = np.zeros(jm.NUM_JOINTS)
        if len(s) < 5:
            print(f"{name:<8}{'-':>8}{'-':>8}{'-':>12}{len(s):>6}")
            continue
        ts = np.array([a for a, _ in s]); ys = np.array([b for _, b in s])
        y0 = ys[0]; span = target - y0
        lat = rise = float("nan")
        if abs(span) > 1e-4:
            frac = (ys - y0) / span
            i10 = np.argmax(frac >= 0.1) if np.any(frac >= 0.1) else None
            i90 = np.argmax(frac >= 0.9) if np.any(frac >= 0.9) else None
            if i10 is not None:
                lat = ts[i10] * 1000
            if i10 is not None and i90 is not None:
                rise = (ts[i90] - ts[i10]) * 1000
        ss = np.degrees(abs(target - ys[-3:].mean()))
        rows.append((name, lat, rise, ss))
        print(f"{name:<8}{lat:>8.0f}{rise:>8.0f}{ss:>12.2f}{len(s):>6}")
    spin(0.5)
    if rows:
        a = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)
        print("-" * 44)
        print(f"{'평균':<8}{np.nanmean(a[:,0]):>8.0f}{np.nanmean(a[:,1]):>8.0f}{np.nanmean(a[:,2]):>12.2f}")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
