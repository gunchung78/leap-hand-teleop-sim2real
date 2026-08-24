# ROS2 로 묶기: 노드 다섯, 토픽 하나로 시뮬과 실기를 같이

> 원문: `../design/ros2_twin.md`, `../history/phase1_plan.md` 3·5장, `ros2_ws/src/leap_teleop/`, `../teleop_howto.md` "ROS2 로 돌리기"

## 요약

- 단일 스크립트를 왜 노드로 쪼갰는지, 어디서 쪼갰는지 설명한다.
- QoS, `header.stamp` 전파, 런치 파라미터 타입 같은 **ROS2 특유의 함정**을 정리한다.
- 업스트림 패키지를 **vendoring 하지 않고** 재현 가능하게 쓰는 법(복사 스크립트 + 패치)을 정리한다.

## 배경

### 노드 그래프

```
tracker_node ──/hand/landmarks──▶ retarget_node ──/leap/joint_cmd──┬──▶ sim_node ──▶ /sim/joint_states
 (웹캠, MediaPipe,                (7cccfdd 손끝 IK,                 │    (MuJoCo, ctrl, 60 Hz)
  SPACE 데드맨)                     평활·데드밴드·유실 램프)          │
                                                                    └──▶ hand_bridge_node ──/cmd_leap──▶ leaphand_node (업스트림+패치)
                                                                          (데드맨·동기·합류·클립·전류동결)      └──/leap_pos_vel_eff──▶ /real/joint_states
```

**명령 토픽 하나**(`/leap/joint_cmd`, MuJoCo 이름 16, rad)가 시뮬과 실기를 같이 먹인다 — 이게 "디지털 트윈"의 실체다.
실기로 나가는 길은 `hand_bridge_node` 하나뿐이고, 업스트림 `leaphand_node` 는 포트 파라미터와 `hold_on_start` 패치만 얹었다.

### 왜 쪼개나

| 경계 | 이유 |
|---|---|
| tracker ↔ retarget | 카메라를 바꾸거나(Vision Pro, 장갑) 리타겟터를 바꿀 때 서로 모르게 |
| retarget ↔ sim/bridge | 같은 명령을 여러 소비자가 — 트윈, 실기, 기록(`ros2 bag`), 나중엔 학습 정책이 명령 생산자가 된다 |
| bridge ↔ 업스트림 | 안전 로직(데드맨·동결)은 우리 것, 모터 통신은 업스트림 것. 업스트림 갱신에 우리 안전 로직이 안 깨진다 |

### ROS2 함정 네 가지 (전부 실제로 밟았다)

1. **QoS**: 센서 토픽은 `BEST_EFFORT depth 1`. 밀린 프레임은 쓰레기라 최신 것만 쓴다. `RELIABLE` 로 두면 느린 구독자가
   지연을 쌓는다.
2. **`header.stamp` 전파**: 카메라 촬영 시각을 `/hand/landmarks` → `/leap/joint_cmd` 까지 그대로 실어 보낸다. 어느 노드에서든
   `now − stamp` 가 종단 지연이다. 지연 측정 코드가 따로 필요 없다. (합성 명령 — 손 유실 램프 — 은 `frame_id` 로 구분해 제외.)
3. **런치 파라미터 타입**: `restart_mm:=15` 처럼 정수를 주면 INTEGER 파라미터가 되고 노드가 DOUBLE 을 기대해 죽는다.
   `ParameterValue(lc(...), value_type=float)` 로 고정 — 단, `IncludeLaunchDescription` 의 `launch_arguments` 에는 넣으면 안 된다
   (문자열 치환만).
4. **빌드 환경**: `/usr/bin/colcon` 으로 빌드하면 셔뱅이 시스템 파이썬이라 conda 패키지(mediapipe, mujoco)를 못 찾는다.
   conda 환경에 colcon 을 설치해 빌드하고, `ros2 run` 전에 `conda activate`.

### 업스트림 쓰는 법

`LEAP_Hand_API/ros2_module` 은 CC BY-NC 4.0 이라 저장소에 복사해 두지 않는다. `ros2_ws/setup_upstream.sh` 가 고정 커밋에서
복사하고 `patches/leap_hand_port_param.patch` 하나를 얹는다. 패치 내용: `port`/`baudrate` 파라미터, `hold_on_start`(토크 켤 때
현재 자세 유지). 업스트림 갱신 = 스크립트 재실행 + 패치 재적용.

## 실행

```bash
conda activate leap-hand && source /opt/ros/humble/setup.bash
pip install colcon-common-extensions empy==3.3.4 lark catkin_pkg     # 최초 1회 (conda 의 colcon)
bash ros2_ws/setup_upstream.sh && cd ros2_ws && colcon build --symlink-install && source install/setup.bash && cd ..

ros2 launch leap_teleop sim.launch.py                      # 시뮬만
ros2 topic hz /hand/landmarks /leap/joint_cmd /sim/joint_states
ros2 topic echo --once /leap/joint_cmd | head -30           # name 16개가 MuJoCo 순서인지
ros2 launch leap_teleop real.launch.py fake:=true          # 가짜 실기로 배선 확인 (Dynamixel 없이)
ros2 topic pub --once /teleop/enable std_msgs/msg/Bool "data: true"
ros2 launch leap_teleop real.launch.py                     # 🤖 실기. 카메라 창 SPACE = 데드맨
```

### 예시 출력 — 토픽 주파수 (`measurement.md` `p1_4`)

```
토픽                  수신    Hz   지연 평균 ms   95%   최대
/hand/landmarks       600  30.0        21.5   23.3   26.1
/leap/joint_cmd       600  30.0        32.1   33.9   37.8
/sim/joint_states    1201  60.0
/real/joint_states    578  28.9
```

## 결과 읽는 법

- 노드를 쪼갠 뒤 종단 지연이 32 ms — 단일 스크립트 때와 같다. 전송 비용은 1 ms 미만. ROS2 는 지연을 만들지 않는다, 밀린 큐가 만든다.
- `fake:=true` 는 업스트림 노드와 **같은 토픽·서비스 이름**만 흉내낸다. 안전 로직(데드맨, 동결)을 실기 없이 시험하는 용도.
  `fake_current` 파라미터로 전류 초과도 재현된다.
- `sim_node` 는 `qpos` 를 직접 넣지 않고 액추에이터에 명령을 주고 물리를 돌린다. 충돌과 추종 지연이 보여야 트윈 구실을 한다
  (시뮬에서 `if_rot`/`mf_rot` 충돌이 그렇게 드러났다).
