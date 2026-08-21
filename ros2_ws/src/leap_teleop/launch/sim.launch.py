"""sim.launch.py — 웹캠 텔레오퍼레이션, 시뮬만 (실기 없음). 로드맵 Day 3.

    ros2 launch leap_teleop sim.launch.py
    ros2 launch leap_teleop sim.launch.py show:=true mirror:=true camera:=1
    ros2 launch leap_teleop sim.launch.py viewer:=false      # 헤드리스 (토픽만)
    ros2 launch leap_teleop sim.launch.py tracker:=false     # 카메라 없이 sim_node 만 (계단 응답 시험)

노드 셋: tracker_node -> retarget_node -> sim_node. 토픽은 leap_teleop/__init__.py.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("camera", default_value="0", description="/dev/videoN"),
        DeclareLaunchArgument("hand", default_value="Right"),
        DeclareLaunchArgument("mirror", default_value="false",
                              description="오른손이 Right 로 안 잡히면 true (영상 좌우 반전)"),
        DeclareLaunchArgument("show", default_value="true", description="카메라 창 + 손 위치 틀"),
        DeclareLaunchArgument("viewer", default_value="true", description="MuJoCo 뷰어"),
        DeclareLaunchArgument("smoothing", default_value="0.4"),
        DeclareLaunchArgument("max_speed", default_value="8.0"),
        DeclareLaunchArgument("tracker", default_value="true",
                              description="false 면 tracker/retarget 없이 sim_node 만 (계단 응답 등 카메라 없는 시험)"),
    ]
    lc = LaunchConfiguration
    return LaunchDescription(args + [
        Node(package="leap_teleop", executable="tracker_node", name="tracker_node",
             output="screen", emulate_tty=True, condition=IfCondition(lc("tracker")),
             parameters=[{"camera": lc("camera"), "hand": lc("hand"),
                          "mirror": lc("mirror"), "show": lc("show")}]),
        Node(package="leap_teleop", executable="retarget_node", name="retarget_node",
             output="screen", emulate_tty=True, condition=IfCondition(lc("tracker")),
             parameters=[{"smoothing": lc("smoothing"), "max_speed": lc("max_speed")}]),
        Node(package="leap_teleop", executable="sim_node", name="sim_node",
             output="screen", emulate_tty=True,
             parameters=[{"viewer": lc("viewer")}]),
    ])
