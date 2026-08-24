# LEAP Hand v1 Lite — 웹캠 텔레오퍼레이션과 디지털 트윈

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](docs/setup.md)
[![ROS 2 Humble](https://img.shields.io/badge/ROS%202-Humble-22314E.svg)](docs/setup.md)
[![MuJoCo 3.11](https://img.shields.io/badge/MuJoCo-3.11-orange.svg)](https://github.com/google-deepmind/mujoco)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

웹캠 앞에서 손을 움직이면 **MuJoCo 속 LEAP Hand 와 실물 LEAP Hand(Lite)가 같이 따라 움직이는** 오픈소스 스택이다.
MediaPipe 손 추적 → 손끝 위치 IK 리타겟팅 → ROS2 → MuJoCo 트윈 + 실기. 카메라 촬영에서 관절 명령까지 **32 ms**, 시뮬–실기 관절 오차 **RMS 1.95°**.

```
웹캠 → tracker_node → /hand/landmarks → retarget_node → /leap/joint_cmd ─┬─▶ sim_node (MuJoCo 트윈)
                                                                        └─▶ hand_bridge_node → 실기 (LEAP_Hand_API)
```

- **로봇 없이 시작할 수 있다** — 웹캠 하나면 MuJoCo 트윈이 손을 따라온다. 웹캠도 없으면 시뮬만 띄워 파이프라인을 확인한다.
- **재현 가능** — 이 문서의 모든 수치는 `scripts/` 로 다시 잴 수 있다. 참조 저장소는 커밋 고정.
- **안전 우선** — 실기로 나가는 길은 노드 하나(`hand_bridge_node`)뿐: 데드맨, 시작 자세 유지, 합류 램프, 전류 동결, 관절 범위 클립. 가짜 실기(`fake:=true`)로 이 로직을 실기 없이 시험한다.
- **ROS 없이도 코어를 쓴다** — `leap_hand_mapping/` 은 순수 파이썬 패키지. 단일 스크립트 텔레옵(`p1_3_teleop_mujoco.py`)이 있다.

## 요구 사항

| | 필요 | 비고 |
|---|---|---|
| OS | Ubuntu 22.04 | 검증된 구성. 다른 배포판은 ROS2 Humble 이 되면 된다 |
| Python | 3.10 (conda) | ROS2 Humble 의 `rclpy` 와 같은 버전이어야 한다 |
| ROS 2 | Humble (apt) | 단일 스크립트 경로(`scripts/phase1/p1_3_teleop_mujoco.py`)는 ROS 없이 돈다 |
| 웹캠 | 아무거나 | 없으면 시뮬만 확인 가능 |
| GPU | 불필요 | MediaPipe·MuJoCo 모두 CPU. 노트북 i7 에서 30 Hz |
| 로봇 | LEAP Hand v1 Lite + U2D2 + 5 V 30 A | 선택. 없으면 `fake:=true` |

## 설치 (한 번, 약 15분)

전체 절차·버전 고정 이유·문제 해결은 **[docs/setup.md](docs/setup.md)**. 아래는 복붙용 요약이다.

```bash
# 0. 시스템: ROS2 Humble 이 /opt/ros/humble 에 있어야 한다 (https://docs.ros.org/en/humble/Installation.html)
git clone https://github.com/gunchung78/leap-hand-teleop-sim2real.git && cd leap-hand-teleop-sim2real

# 1. 참조 저장소 (gitignore, 커밋 고정). LEAP_Hand_API / Bidex 는 CC BY-NC 라 직접 받는다
mkdir -p third_party && cd third_party
git clone https://github.com/google-deepmind/mujoco_menagerie.git  && git -C mujoco_menagerie      checkout da76818
git clone https://github.com/leap-hand/LEAP_Hand_API.git           && git -C LEAP_Hand_API          checkout b0d00c8
git clone https://github.com/leap-hand/Bidex_VisionPro_Teleop.git  && git -C Bidex_VisionPro_Teleop checkout 4914349
cd ..

# 2. conda 환경 + 코어 패키지 + MediaPipe 모델
conda create -n leap-hand python=3.10 -y && conda activate leap-hand
pip install mujoco==3.11.0 pybullet==3.2.7 mediapipe==1.0.1 opencv-python numpy dynamixel-sdk
pip install -e .
pip install empy==3.3.4 lark catkin_pkg colcon-common-extensions     # ROS2 빌드를 conda 파이썬으로
bash scripts/phase1/p1_0_fetch_mediapipe_model.sh                   # hand_landmarker.task → models/

# 3. ROS2 워크스페이스
source /opt/ros/humble/setup.bash
bash ros2_ws/setup_upstream.sh                                       # LEAP_Hand_API ros2_module 복사 + 패치
cd ros2_ws && colcon build --symlink-install && cd ..
```

확인: `python -m leap_hand_mapping.joint_map | tail -1` 이 `self_check 통과` 를 찍고, `which colcon` 이 conda 경로면 된다.

## 빠른 시작 (5분)

매 터미널에서 먼저:
```bash
conda activate leap-hand && source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
```

**1. 카메라·로봇 없이 — 파이프라인이 도는지**
```bash
python scripts/phase0/p0_1_verify_mapping_fk.py | tail -3     # "통과 — 매핑 테이블 확정"
ros2 launch leap_teleop sim.launch.py tracker:=false            # MuJoCo 창에 LEAP Hand 가 뜨면 된다
```

**2. 웹캠 — 손이 트윈을 움직인다**
```bash
ros2 launch leap_teleop sim.launch.py                           # 웹캠 → MuJoCo 손. 오른손을 카메라에 편다
python scripts/phase1/p1_4_teleop_metrics.py --seconds 20       # Hz · 종단 지연 · 추종 오차 · 떨림
```
왼손으로 잡히거나 거울상이면 `mirror:=true`. 단일 프로세스로 보려면 `python scripts/phase1/p1_3_teleop_mujoco.py`.

**3. 실기 — 먼저 가짜로, 그다음 진짜로**
```bash
ros2 launch leap_teleop real.launch.py fake:=true               # 가짜 실기: 배선·데드맨·전류 동결 로직만 시험
ros2 launch leap_teleop real.launch.py                          # 실기. 카메라 창에서 SPACE 를 누르는 동안만 움직인다 (데드맨)
```
실기를 처음 켠다면 **[docs/real_hand_bringup.md](docs/real_hand_bringup.md)** 순서(전원 → 포트 권한 → USB 지연 → 사전 점검 → 관절 하나씩)를 먼저 밟을 것.
텔레옵 전체 절차(촬영 환경, 튜닝, 안전 수칙)는 **[docs/teleop_howto.md](docs/teleop_howto.md)**.

## 문서 지도

| 읽을 것 | 언제 |
|---|---|
| **[docs/setup.md](docs/setup.md)** | 처음 설치할 때. 버전 고정 이유, 확인 절차, 문제 해결 |
| **[docs/teleop_howto.md](docs/teleop_howto.md)** | 텔레오퍼레이션을 돌리는 절차 (단일 스크립트 / ROS2), 튜닝, 문제 해결 |
| **[docs/real_hand_bringup.md](docs/real_hand_bringup.md)** | 실기를 처음 켤 때 (전원 → 권한 → 지연시간 → 사전 점검 → 관절 하나씩) |
| **[docs/guide/](docs/guide/README.md)** | 스택을 이해하고 시험할 때. 로봇 손 구조 → 관절 매핑 → 실기 안전 → 손 추적 → 리타겟팅 → ROS2 → 측정 → 디버깅 사례집 |
| [docs/design/](docs/design/) | 왜 이렇게 만들었나: [관절 매핑 검증](docs/design/joint_mapping.md) · [리타겟팅 파이프라인](docs/design/teleop_pipeline.md) · [ROS2 트윈과 측정](docs/design/ros2_twin.md) |
| [docs/history/](docs/history/) | 개발 기록(계획서·회고). 그때 기준의 문서 — 절차는 위 문서를 따를 것 |
| [data/](data/README.md) | 매핑 JSON, 사람 손 녹화 npz |
| [NOTICE.md](NOTICE.md) | 서드파티 라이선스. **LEAP_Hand_API 는 CC BY-NC 4.0** |

## 저장소 구조

```
leap_hand_mapping/        코어 파이썬 패키지 (pip install -e .), ROS 비의존
  joint_map.py            MuJoCo ↔ 실기 모터 매핑, 관절 범위, 클립     hand_tracker.py  웹캠 → MediaPipe 21점
  retarget.py             21점 → 16 관절각 (손끝 위치 IK)               real_hand.py     실기 드라이버 래퍼 (전류 350 mA 고정)
ros2_ws/src/leap_teleop/  ROS2 노드 5개 (tracker / retarget / sim / hand_bridge / fake_hand) + 런치 2개 (sim, real)
ros2_ws/setup_upstream.sh 업스트림 ROS2 노드(LEAP_Hand_API) 복사 + patches/ 적용   ← 저장소에 포함하지 않음
scripts/phase0/           관절 매핑 FK 검증, 실기 사전 점검, 관절 순차 구동, 읽기 오류율
scripts/phase1/           손 추적 확인, 리타겟 왕복 시험, 단일 스크립트 텔레옵, 지표(p1_4), 계단 응답(p1_5)
data/                     매핑 JSON, 사람 손 녹화 npz        models/   MediaPipe 모델(내려받음)
third_party/              참조 저장소 클론 (gitignore)        docs/     문서 (setup · howto · guide · design · history)
```

## 주요 수치 (실측, 재현 스크립트 표기)

| 항목 | 값 | 스크립트 |
|---|---|---|
| 관절 매핑 검증 (FK, 200 자세) | 채택안 std 0.06 mm vs 차점 25.6 mm | `p0_1_verify_mapping_fk.py` |
| 실기 계단 응답 (16관절, 20°) | 지연 83 ms · 상승 68 ms · 정상오차 0.81° | `p1_5_step_response.py --source real` |
| 라이브 종단 지연 (촬영→관절 명령) | 32 ms (추적 21.5 + 리타겟·전송 10.6) | `p1_4_teleop_metrics.py` |
| 시뮬–실기 추종 오차 | RMS 1.95° | 〃 |
| 정지 떨림 (명령 / 실기) | 0.75° / 0.96° (엄지 `th_cmc` 3° / 4.8°) | 〃 |
| 실기 읽기 오류율 (4 Mbps) | ~3% CRC, 무해 | `p0_4_read_reliability.py` |

측정 방법과 조건은 [docs/design/ros2_twin.md](docs/design/ros2_twin.md).

## 하드웨어

LEAP Hand v1 **Lite**(Dynamixel XL330-M288-T ×16, 플라스틱 기어), U2D2, 5V 30A. Lite 는 전류 한계 **350 mA** 를 넘기면 기어가 상한다 —
코드에 고정돼 있고 올리지 않는다. 손목 회전 자세 추적은 범위 밖(단안 깊이 한계).

## 자주 걸리는 것

| 증상 | 조치 |
|---|---|
| 노드에서만 `ModuleNotFoundError: mediapipe` | 시스템 colcon 으로 빌드됨 → `which colcon` 이 conda 경로인지 확인 후 `colcon build` 다시 |
| `rclpy._rclpy_pybind11 ... cpython-313` | conda base 가 활성 → `conda activate leap-hand` |
| `can't open camera by index` | 다른 프로세스가 카메라를 쥠 → `fuser /dev/video0` |
| 왼손으로 잡힘 / 거울상 | `mirror:=true` (단일 스크립트는 `--mirror`) |
| 실기가 안 움직임 | 데드맨(카메라 창 SPACE). `ros2 topic echo /teleop/enable` |

더 많은 사례: [docs/setup.md §5](docs/setup.md), [docs/teleop_howto.md 문제 해결](docs/teleop_howto.md), [docs/guide/debugging_casebook.md](docs/guide/debugging_casebook.md).

## 감사

[LEAP Hand](https://leap-hand.github.io) (CMU) 의 하드웨어·API·URDF, [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 의 LEAP 모델, [MediaPipe](https://github.com/google-ai-edge/mediapipe) Hand Landmarker 위에서 만들었다.

## 라이선스

이 저장소 코드는 [MIT](LICENSE). 참조하는 외부 소프트웨어와 조건은 [NOTICE.md](NOTICE.md) — 특히 LEAP_Hand_API / Bidex_VisionPro_Teleop 는 **CC BY-NC 4.0(비상업)** 이며 저장소에 포함하지 않고 이용자가 직접 받는다.
