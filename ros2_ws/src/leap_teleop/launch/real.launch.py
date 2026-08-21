"""real.launch.py — 웹캠 텔레오퍼레이션, 시뮬 + 실기 동시. 로드맵 Day 4.

    ros2 launch leap_teleop real.launch.py                 # 실기 (업스트림 leaphand_node)
    ros2 launch leap_teleop real.launch.py fake:=true      # 실기 대신 fake_hand_node (배선 시험)
    ros2 launch leap_teleop real.launch.py sim:=false      # MuJoCo 트윈 없이
    ros2 launch leap_teleop real.launch.py tracker:=false  # 카메라 없이 (계단 응답 시험)
    카메라 창에서 SPACE = 데드맨 토글 (ROBOT ON/OFF). 또는
    ros2 topic pub --once /teleop/enable std_msgs/Bool "data: true"    # 데드맨 ON. 이걸 줘야 움직인다

sim.launch.py 의 세 노드 + hand_bridge_node + leaphand_node(업스트림, 패치본).
명령 토픽 하나(/leap/joint_cmd)가 시뮬과 실기를 동시에 먹인다.

실기 파라미터는 여기서만 정한다 (인수인계 문서 4.1/4.2/4.3):
    curr_lim 350    Lite. 업스트림 런치의 500 을 쓰면 플라스틱 기어가 상한다. 올리지 말 것
    kP 600 kI 0 kD 200    진동하면 kP 400 부근으로
    port /dev/serial/by-id/...    열거 순서와 무관. ls /dev/serial/by-id 로 확인
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

DEFAULT_PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBIN91W-if00-port0"


def generate_launch_description():
    lc = LaunchConfiguration
    args = [
        DeclareLaunchArgument("camera", default_value="0"),
        DeclareLaunchArgument("hand", default_value="Right"),
        DeclareLaunchArgument("mirror", default_value="false"),
        DeclareLaunchArgument("show", default_value="true"),
        DeclareLaunchArgument("sim", default_value="true", description="MuJoCo 트윈도 같이"),
        DeclareLaunchArgument("viewer", default_value="true"),
        DeclareLaunchArgument("fake", default_value="false",
                              description="true 면 실기 대신 fake_hand_node (Dynamixel 없음)"),
        DeclareLaunchArgument("port", default_value=DEFAULT_PORT),
        DeclareLaunchArgument("kP", default_value="600.0"),
        DeclareLaunchArgument("kD", default_value="200.0"),
        DeclareLaunchArgument("curr_lim", default_value="350.0", description="Lite 는 350. 올리지 말 것"),
        DeclareLaunchArgument("max_speed", default_value="8.0"),
        DeclareLaunchArgument("smoothing", default_value="0.4", description="리타겟 지수 평활. 떨리면 0.2"),
        DeclareLaunchArgument("deadband", default_value="0.5", description="리타겟 출력 데드밴드(deg). 0=끔"),
        DeclareLaunchArgument("restart_mm", default_value="1.0", description="IK 재시도 임계(mm). 15 면 재시도 0"),
        DeclareLaunchArgument("current_warn", default_value="400.0",
                              description="브리지 전류 동결 임계 (업스트림 리더 단위 = raw x 1.34; 한계 350 은 ~469)"),
        DeclareLaunchArgument("engage_speed", default_value="1.0",
                              description="데드맨 ON 직후 목표에 합류하는 속도 rad/s. 합류 뒤 max_speed"),
        DeclareLaunchArgument("poll_rate", default_value="30.0",
                              description="브리지가 /leap_pos_vel_eff 를 읽는 주기. 읽기 오류가 잦으면 15 로"),
        DeclareLaunchArgument("tracker", default_value="true",
                              description="false 면 카메라/리타겟 없이 브리지+실기(+시뮬)만"),
    ]

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("leap_teleop"), "launch", "sim.launch.py")),
        launch_arguments={
            "camera": lc("camera"), "hand": lc("hand"), "mirror": lc("mirror"),
            "show": lc("show"), "viewer": lc("viewer"), "max_speed": lc("max_speed"),
            "tracker": lc("tracker"), "smoothing": lc("smoothing"), "deadband": lc("deadband"),
            "restart_mm": lc("restart_mm"),
        }.items(),
        condition=IfCondition(lc("sim")),
    )
    # sim:=false 면 MuJoCo 없이 tracker + retarget 만 (tracker:=false 면 그것도 없이)
    sim_off = [
        Node(package="leap_teleop", executable="tracker_node", name="tracker_node",
             output="screen", emulate_tty=True,
             condition=IfCondition(PythonExpression(["'", lc("sim"), "' == 'false' and '", lc("tracker"), "' == 'true'"])),
             parameters=[{"camera": lc("camera"), "hand": lc("hand"),
                          "mirror": lc("mirror"), "show": lc("show")}]),
        Node(package="leap_teleop", executable="retarget_node", name="retarget_node",
             output="screen", emulate_tty=True,
             condition=IfCondition(PythonExpression(["'", lc("sim"), "' == 'false' and '", lc("tracker"), "' == 'true'"])),
             parameters=[{"max_speed": lc("max_speed"), "smoothing": lc("smoothing"),
                          "deadband_deg": lc("deadband"), "restart_mm": lc("restart_mm")}]),
    ]

    bridge = Node(package="leap_teleop", executable="hand_bridge_node", name="hand_bridge_node",
                  output="screen", emulate_tty=True,
                  parameters=[{"max_speed": lc("max_speed"), "current_warn": lc("current_warn"),
                               "poll_rate": lc("poll_rate"), "engage_speed": lc("engage_speed")}])

    real = Node(package="leap_hand", executable="leaphand_node.py", name="leaphand_node",
                output="screen", emulate_tty=True,
                condition=UnlessCondition(lc("fake")),
                parameters=[{"kP": lc("kP"), "kI": 0.0, "kD": lc("kD"),
                             "curr_lim": lc("curr_lim"), "port": lc("port")}])
    fake = Node(package="leap_teleop", executable="fake_hand_node", name="fake_hand_node",
                output="screen", emulate_tty=True, condition=IfCondition(lc("fake")))

    return LaunchDescription(args + [sim_launch] + sim_off + [bridge, real, fake])
