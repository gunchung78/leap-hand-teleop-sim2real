"""policy.launch.py — 학습 정책(rotate_z)으로 LEAP 을 움직인다. Phase 2 S5/S6.

    ros2 launch leap_teleop policy.launch.py policy:=models/rotate_z_v0.npz            # 트윈: 큐브 장면 MuJoCo 만
    ros2 launch leap_teleop policy.launch.py policy:=... real:=true                     # + 실기 (브리지 + leaphand_node)
    ros2 launch leap_teleop policy.launch.py policy:=... real:=true fake:=true          # + 가짜 실기 (배선 시험)
    ros2 topic pub --once /teleop/enable std_msgs/msg/Bool "data: true"                # 실기 데드맨 ON (카메라 창이 없다)

구성
    sim_node(scene:=cube)  playground 학습 장면(손바닥 위 큐브) — 트윈이자 정책의 기본 상태 입력
    policy_node            /sim/joint_states (또는 real:=true 면 /real/joint_states) -> 정책 -> /leap/joint_cmd 20 Hz
    real:=true             hand_bridge_node + leaphand_node 가 같은 /leap/joint_cmd 를 먹는다 (Phase 1 과 동일 경로)

범위 클립은 limits:=model (두 모델 교집합). 텔레옵 기본(teleop, 벌림 ±3도)을 쓰면 정책이 벌림을 못 써 깨진다.
실기 파라미터(kP/curr_lim/port)는 real.launch 와 같은 기본값. curr_lim 350 은 올리지 말 것.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

DEFAULT_PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBIN91W-if00-port0"


def generate_launch_description():
    lc = LaunchConfiguration
    args = [
        DeclareLaunchArgument("policy", description="p2_3_export_policy.py 가 만든 npz (저장소 기준 상대 경로 가능)"),
        DeclareLaunchArgument("real", default_value="false", description="true 면 실기(브리지+leaphand_node)도, 정책 입력은 /real/joint_states"),
        DeclareLaunchArgument("fake", default_value="false", description="real:=true 와 함께: 실기 대신 fake_hand_node"),
        DeclareLaunchArgument("sim", default_value="true", description="MuJoCo 큐브 장면 트윈"),
        DeclareLaunchArgument("viewer", default_value="true"),
        DeclareLaunchArgument("limits", default_value="model", description="관절 범위 표: model | teleop"),
        DeclareLaunchArgument("noise", default_value="0.0", description="정책 관측 잡음 rad (학습 0.05)"),
        DeclareLaunchArgument("max_speed", default_value="8.0"),
        DeclareLaunchArgument("engage_speed", default_value="1.0"),
        DeclareLaunchArgument("current_warn", default_value="400.0"),
        DeclareLaunchArgument("port", default_value=DEFAULT_PORT),
        DeclareLaunchArgument("kP", default_value="400.0"),
        DeclareLaunchArgument("kD", default_value="200.0"),
        DeclareLaunchArgument("curr_lim", default_value="350.0", description="Lite 는 350. 올리지 말 것"),
    ]
    source = PythonExpression(["'real' if '", lc("real"), "' == 'true' else 'sim'"])
    sim = Node(package="leap_teleop", executable="sim_node", name="sim_node", output="screen", emulate_tty=True,
               condition=IfCondition(lc("sim")),
               parameters=[{"scene": "cube", "viewer": lc("viewer"), "limits": lc("limits")}])
    policy = Node(package="leap_teleop", executable="policy_node", name="policy_node", output="screen", emulate_tty=True,
                  parameters=[{"policy": lc("policy"), "source": source,
                               "noise": ParameterValue(lc("noise"), value_type=float),
                               "require_enable": ParameterValue(lc("real"), value_type=bool)}])
    bridge = Node(package="leap_teleop", executable="hand_bridge_node", name="hand_bridge_node", output="screen",
                  emulate_tty=True, condition=IfCondition(lc("real")),
                  parameters=[{"max_speed": ParameterValue(lc("max_speed"), value_type=float),
                               "engage_speed": ParameterValue(lc("engage_speed"), value_type=float),
                               "current_warn": ParameterValue(lc("current_warn"), value_type=float),
                               "limits": lc("limits")}])
    real = Node(package="leap_hand", executable="leaphand_node.py", name="leaphand_node", output="screen", emulate_tty=True,
                condition=IfCondition(PythonExpression(["'", lc("real"), "' == 'true' and '", lc("fake"), "' != 'true'"])),
                parameters=[{"kP": ParameterValue(lc("kP"), value_type=float), "kI": 0.0,
                             "kD": ParameterValue(lc("kD"), value_type=float),
                             "curr_lim": ParameterValue(lc("curr_lim"), value_type=float), "port": lc("port")}])
    fake = Node(package="leap_teleop", executable="fake_hand_node", name="fake_hand_node", output="screen", emulate_tty=True,
                condition=IfCondition(PythonExpression(["'", lc("real"), "' == 'true' and '", lc("fake"), "' == 'true'"])))
    return LaunchDescription(args + [sim, policy, bridge, real, fake])
