#!/usr/bin/env python3
"""p1_4_teleop_metrics.py — 돌아가고 있는 ROS2 텔레오퍼레이션의 라이브 지표를 잰다.

"손 흔들어 보니 따라오더라"는 검증이 아니다. 런치를 띄워 둔 채 이 스크립트를 N 초 돌리면
토픽별 주파수, 촬영->각 단계 종단 지연, 시뮬-실기 추종 오차를 표로 찍는다.

    ros2 launch leap_teleop sim.launch.py            # 또는 real.launch.py [fake:=true]
    python scripts/phase1/p1_4_teleop_metrics.py --seconds 20

지연은 header.stamp(카메라 촬영 시각)를 끝까지 전파한 덕에 공짜로 나온다:
    /hand/landmarks   stamp = 촬영 시각           -> 수신 시 now - stamp = 추적 + 전송
    /leap/joint_cmd   stamp = 같은 촬영 시각 전파  -> now - stamp = 추적 + 리타겟 + 전송
    /sim/joint_states, /real/joint_states 는 stamp = now 라 지연 대신 주파수와 추종 오차를 본다.

시뮬-실기 추종 오차는 같은 시각의 /sim 과 /real 자세 차이(관절별 RMS, deg)다. 둘 다
같은 /leap/joint_cmd 를 먹으니 이 값이 "디지털 트윈이 실기를 얼마나 닮았나"다.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2

QOS = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
JOINT_NAMES = ["if_mcp", "if_rot", "if_pip", "if_dip", "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
               "rf_mcp", "rf_rot", "rf_pip", "rf_dip", "th_cmc", "th_axl", "th_mcp", "th_ipl"]
TOPICS = {
    "/hand/landmarks": PointCloud2,
    "/leap/joint_cmd": JointState,
    "/sim/joint_states": JointState,
    "/real/joint_states": JointState,
}


class Metrics(Node):
    def __init__(self) -> None:
        super().__init__("teleop_metrics")
        self.rx = {t: [] for t in TOPICS}        # 수신 시각
        self.lat = {t: [] for t in TOPICS}       # now - stamp (ms)
        self.sim = []                             # (t, q)
        self.real = []
        self.cmd = []                             # (t, q) 관절 명령
        self.lm = []                              # (t, 21x3) 랜드마크
        self.release = 0
        for t, typ in TOPICS.items():
            self.create_subscription(typ, t, lambda m, t=t: self._on(t, m), QOS)

    def _on(self, topic: str, msg) -> None:
        now = self.get_clock().now()
        self.rx[topic].append(now.nanoseconds * 1e-9)
        if topic == "/leap/joint_cmd" and msg.header.frame_id.endswith("/release"):
            self.release += 1          # 손 유실 램프(합성, stamp=now). 지연 통계에서 뺀다
            return
        self.lat[topic].append((now - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-6)
        if topic == "/sim/joint_states":
            self.sim.append((now.nanoseconds * 1e-9, np.array(msg.position)))
        elif topic == "/real/joint_states":
            self.real.append((now.nanoseconds * 1e-9, np.array(msg.position)))
        elif topic == "/leap/joint_cmd":
            self.cmd.append((now.nanoseconds * 1e-9, np.array(msg.position)))
        elif topic == "/hand/landmarks":
            from sensor_msgs_py import point_cloud2 as pc2
            pts = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))
            if pts.shape == (21, 3):
                self.lm.append((now.nanoseconds * 1e-9, pts.astype(float)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    rclpy.init()
    node = Metrics()
    t0 = time.time()
    print(f"{args.seconds:.0f}초 동안 듣는다. 손을 움직여 주세요.")
    try:
        while time.time() - t0 < args.seconds:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    el = time.time() - t0

    print()
    print(f"{'토픽':<22}{'수신':>6}{'Hz':>7}{'지연 평균 ms':>12}{'95%':>7}{'최대':>7}")
    print("-" * 62)
    for t in TOPICS:
        n = len(node.rx[t])
        hz = n / el if el > 0 else 0.0
        if n:
            lat = np.array(node.lat[t])
            # /sim, /real 은 stamp=now 라 지연이 0 근처다. 표에서는 의미 있는 둘만 숫자를 낸다
            show = t in ("/hand/landmarks", "/leap/joint_cmd")
            l1 = f"{lat.mean():12.1f}{np.percentile(lat, 95):7.1f}{lat.max():7.1f}" if show else f"{'-':>12}{'-':>7}{'-':>7}"
        else:
            l1 = f"{'-':>12}{'-':>7}{'-':>7}"
        extra = f"   (그중 손 유실 램프 {node.release})" if t == "/leap/joint_cmd" and node.release else ""
        print(f"{t:<22}{n:>6}{hz:>7.1f}{l1}{extra}")

    if node.sim and node.real:
        # 시각 정렬: real 표본마다 가장 가까운 sim 표본
        ts = np.array([t for t, _ in node.sim]); qs = np.array([q for _, q in node.sim])
        err = []
        for t, q in node.real:
            i = int(np.argmin(np.abs(ts - t)))
            if abs(ts[i] - t) < 0.05 and qs[i].shape == q.shape:
                err.append(q - qs[i])
        if err:
            e = np.degrees(np.array(err))
            rms = np.sqrt((e ** 2).mean(axis=0))
            print()
            print(f"시뮬-실기 추종 오차 (deg RMS, {len(err)} 쌍)  전체 {np.sqrt((e**2).mean()):.2f}"
                  f"  최대 관절 {rms.max():.2f}")
            print("  관절별: " + " ".join(f"{v:.1f}" for v in rms))
    else:
        print("\n시뮬-실기 추종 오차: /sim 또는 /real 이 없어 생략")

    # ---- 떨림: 어디서 떨리는가 ----
    # 마지막 2초 창에서 (a) 랜드마크, (b) 관절 명령, (c) 실기 위치의 프레임간 변화와 표준편차를 잰다.
    # 손을 가만히 들고 재면: 명령이 떨리면 리타겟/센서 쪽, 명령은 조용한데 실기가 떨리면 모터(PID) 쪽.
    def jitter(series, label, scale=1.0, unit="deg", names=None):
        if len(series) < 10:
            print(f"  {label}: 표본 부족 ({len(series)})")
            return
        t_end = series[-1][0]
        win = [q for t, q in series if t >= t_end - 2.0]
        if len(win) < 5:
            print(f"  {label}: 마지막 2초 표본 부족 ({len(win)})")
            return
        a = np.array(win) * scale
        d = np.abs(np.diff(a, axis=0))
        sd = a.std(axis=0)
        worst = np.argsort(sd)[::-1][:4]
        wtxt = "  ".join(f"{names[i] if names else i}:{sd[i]:.2f}" for i in worst)
        print(f"  {label:<14} 표본 {len(win):3d}  프레임간 변화 평균 {d.max(axis=1).mean():.2f} {unit}"
              f"  표준편차 평균 {sd.mean():.2f} {unit}  가장 떠는 관절 {wtxt}")
    print("\n정지 떨림 (마지막 2초 창 — 손을 가만히 든 채 끝내면 의미가 있다)")
    jitter([(t, q) for t, q in node.cmd], "관절 명령", np.degrees(1.0), "deg", JOINT_NAMES)
    jitter([(t, q) for t, q in node.real], "실기 위치", np.degrees(1.0), "deg", JOINT_NAMES)
    jitter([(t, q) for t, q in node.sim], "시뮬 위치", np.degrees(1.0), "deg", JOINT_NAMES)
    if node.lm:
        gap = [(t, np.array([np.linalg.norm(p[4] - p[8]) * 1000])) for t, p in node.lm]
        jitter(gap, "랜드마크 간격", 1.0, "mm")
    print("  읽는 법: 명령이 0.5도 밑인데 실기가 떨면 모터 PID(kP:=400). 명령이 떨면 smoothing:=0.2 deadband:=1.0,"
          " 랜드마크가 떨면 거리/조명(틀 안에서).")

    lm = node.lat["/hand/landmarks"]; cmd = node.lat["/leap/joint_cmd"]
    if lm and cmd:
        print(f"\n종단: 촬영->랜드마크 {np.mean(lm):.1f} ms, 촬영->관절명령 {np.mean(cmd):.1f} ms"
              f" (차이 {np.mean(cmd) - np.mean(lm):.1f} ms 가 리타겟 + 전송)")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
