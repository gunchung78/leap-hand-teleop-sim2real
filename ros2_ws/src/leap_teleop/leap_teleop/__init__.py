"""LEAP Hand v1 Lite 텔레오퍼레이션 ROS2 노드.

알고리즘은 전부 leap_hand_mapping (순수 파이썬, pip install -e .) 에 있고, 여기 노드는
토픽을 잇기만 한다. 카메라·ROS 없이 코어를 테스트할 수 있어야 하고, Phase 2 에서
학습한 정책을 배포할 때도 같은 코어를 쓴다.

    tracker_node      웹캠 -> /hand/landmarks (PointCloud2, 21점, MediaPipe world m)
    retarget_node     /hand/landmarks -> /leap/joint_cmd (JointState, MuJoCo 관절 순서)
    sim_node          /leap/joint_cmd -> MuJoCo 디지털 트윈 -> /sim/joint_states
    hand_bridge_node  /leap/joint_cmd -> 안전 래퍼 -> /cmd_leap (업스트림 leaphand_node)

header.stamp 는 카메라 촬영 시각을 끝까지 전파한다. 어느 노드에서든 now - stamp 로
종단 지연을 잰다.
"""

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

# 텔레오퍼레이션에서 밀린 프레임은 쓰레기다. 큐에 쌓아 지연을 만들지 않고 최신 것만 쓴다.
SENSOR_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)

TOPIC_LANDMARKS = "/hand/landmarks"
TOPIC_JOINT_CMD = "/leap/joint_cmd"
TOPIC_SIM_STATES = "/sim/joint_states"
TOPIC_REAL_STATES = "/real/joint_states"
TOPIC_ENABLE = "/teleop/enable"
