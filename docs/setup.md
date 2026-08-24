# 환경 세팅 — 처음부터 끝까지

이 저장소를 **새 컴퓨터에서 같은 상태로** 만드는 절차다. 위에서 아래로 한 번만 따라가면 된다.
확인된 구성(2026-08-24, geon 의 노트북)을 기준으로 쓰되, 버전을 고정한 이유를 같이 적었다.

| 항목 | 값 |
|---|---|
| OS | Ubuntu 22.04.5 LTS (커널 6.8) |
| GPU / 드라이버 | RTX 3060 Laptop 6 GB / NVIDIA 580 (CUDA 12 호환) |
| ROS2 | Humble (apt, `/opt/ros/humble`, Python 3.10) |
| conda | Anaconda. 환경 `leap-hand`(py3.10, 텔레옵·ROS2) |
| 실기 | LEAP Hand v1 **Lite**, U2D2(FTDI FT232H), 5V 30A 전원 |

---

## 0. 시스템 패키지

```bash
sudo apt update
sudo apt install -y git build-essential ffmpeg
# ROS2 Humble (없으면): https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions   # 시스템 colcon 은 쓰지 않지만 의존성 때문에
# NVIDIA 드라이버: nvidia-smi 가 뜨면 된다 (CUDA 툴킷은 pip 의 jax[cuda12] 가 가져온다)
nvidia-smi
```

## 1. 저장소와 참조 저장소

```bash
git clone <this-repo> leap-hand-teleop-sim2real && cd leap-hand-teleop-sim2real
mkdir -p third_party && cd third_party
git clone https://github.com/google-deepmind/mujoco_menagerie.git      && git -C mujoco_menagerie      checkout da76818
git clone https://github.com/leap-hand/LEAP_Hand_API.git               && git -C LEAP_Hand_API          checkout b0d00c8
git clone https://github.com/leap-hand/Bidex_VisionPro_Teleop.git      && git -C Bidex_VisionPro_Teleop checkout 4914349
git clone https://github.com/dexsuite/dex-urdf.git                     && git -C dex-urdf              checkout f5e7132   # 참고용
cd ..
```

`third_party/` 는 gitignore 다(라이선스·크기). 위 커밋이 검증된 조합이다.

| 저장소 | 쓰는 곳 |
|---|---|
| `mujoco_menagerie` | MuJoCo LEAP 모델 (`leap_hand/right_hand.xml`, 트윈·FK 검증) |
| `LEAP_Hand_API` | 실기 드라이버(`python/`), ROS2 업스트림 노드(`ros2_module`, 복사+패치). CC BY-NC 4.0 |
| `Bidex_VisionPro_Teleop` | 공식 URDF(`robot_pybullet.urdf`) — 리타겟 IK 와 FK 교차검증의 실기 규약 기준 |
| `dex-urdf` | menagerie 모델의 출처. 관절 이름 계보 확인용 |

## 2. `leap-hand` 환경 — 텔레옵 / 디지털 트윈 / ROS2 (Python 3.10)

```bash
conda create -n leap-hand python=3.10 -y
conda activate leap-hand
pip install mujoco==3.11.0 pybullet==3.2.7 mediapipe==1.0.1 opencv-python numpy dynamixel-sdk
pip install -e .                                                      # leap_hand_mapping (ROS 노드가 import)
pip install empy==3.3.4 lark catkin_pkg colcon-common-extensions       # ROS2 빌드를 **conda 파이썬으로**
bash scripts/phase1/p1_0_fetch_mediapipe_model.sh                     # hand_landmarker.task (7.5 MB) -> models/
```

확인:
```bash
python -c "import mujoco, mediapipe, pybullet, cv2; print(mujoco.__version__, mediapipe.__version__)"
python -m leap_hand_mapping.joint_map | tail -1        # self_check 통과
which colcon                                            # .../envs/leap-hand/bin/colcon 이어야 한다
```

검증된 버전: mujoco 3.11.0, mediapipe 1.0.1, opencv 5.0, pybullet 3.2.7, numpy 2.2.6, dynamixel-sdk 4.0.5.
(`jax`/`brax` 가 이 환경에 남아 있어도 쓰지 않는다.)

### 2.1 ROS2 워크스페이스

```bash
conda activate leap-hand
source /opt/ros/humble/setup.bash
bash ros2_ws/setup_upstream.sh          # third_party/LEAP_Hand_API/ros2_module -> ros2_ws/src/leap_hand + 패치 1개
cd ros2_ws && colcon build --symlink-install && source install/setup.bash && cd ..
ros2 pkg executables leap_teleop        # tracker_node retarget_node sim_node hand_bridge_node fake_hand_node
```

**매 터미널**: `conda activate leap-hand && source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash`

주의 둘 — 둘 다 실제로 밟았다:
- `/usr/bin/colcon` 으로 빌드하면 실행 파일 셔뱅이 시스템 파이썬이 돼 노드가 `mediapipe`/`mujoco` 를 못 찾는다. `which colcon` 확인.
- conda **base**(py3.13)가 PATH 앞이면 `rclpy._rclpy_pybind11 ... cpython-313` 오류. 반드시 `leap-hand` 활성화.

동작 확인(카메라 없이): `ros2 launch leap_teleop sim.launch.py tracker:=false` → MuJoCo 창이 뜨면 된다.


## 3. 실기 (LEAP Hand Lite) — 🤖

절차 전체는 `docs/real_hand_bringup.md`. 여기선 **환경 설정** 부분만.

```bash
sudo usermod -aG dialout $USER            # 1회. 재로그인 필요
ls /dev/serial/by-id/                     # usb-FTDI_USB__-__Serial_Converter_<시리얼>-if00-port0  ← 이 경로를 쓴다 (ttyUSB 번호는 바뀐다)
```

FTDI 지연시간은 **재부팅마다 16 으로 돌아온다**(2026-08-24 확인). 매번 손으로 하지 말고 udev 규칙으로:

```bash
sudo tee /etc/udev/rules.d/99-u2d2-latency.rules >/dev/null <<'EOF'
ACTION=="add", SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer     # 1
```

Return Delay Time(모터 쪽) 0 은 Dynamixel Wizard 로 한 번 설정하면 모터에 남는다.
포트가 다르면 런치에 `port:=/dev/serial/by-id/...`. **`curr_lim` 350 은 올리지 않는다**(플라스틱 기어).

## 4. 전체 확인 (10분)

```bash
# leap-hand
conda activate leap-hand && source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
python scripts/phase0/p0_1_verify_mapping_fk.py | tail -3          # "통과 — 매핑 테이블 확정"
ros2 launch leap_teleop sim.launch.py                               # 웹캠 → MuJoCo 손
python scripts/phase1/p1_4_teleop_metrics.py --seconds 10           # 30 Hz, 촬영->명령 ~32 ms
```

## 5. 문제 해결

| 증상 | 원인 → 조치 |
|---|---|
| `ModuleNotFoundError: mediapipe` (노드에서만) | 시스템 colcon 으로 빌드됨 → conda colcon 으로 `colcon build` 다시 |
| `rclpy._rclpy_pybind11 ... cpython-313` | conda base 활성 → `conda activate leap-hand` |
| `can't open camera by index` | 다른 프로세스가 카메라를 쥠 → `fuser /dev/video0`, `pkill -9 -f "[l]ib/leap_teleop/"` |
| `Incorrect status packet` 3% | 배선/4 Mbps CRC, 무해(직전 값 대체). 케이블 재삽입 |
| 포트 못 열음 | `dialout` 그룹 재로그인, Dynamixel Wizard 종료, `by-id` 경로 |

## 6. 디렉터리 한눈에

```
leap_hand_mapping/     코어 (pip -e). 매핑·추적·리타겟
ros2_ws/src/leap_teleop/   ROS2 노드·런치.  ros2_ws/src/leap_hand/ 는 setup_upstream.sh 가 만든다 (gitignore)
scripts/phase0|1/      측정 스크립트 (전부 재현 가능)
models/                hand_landmarker.task (내려받음)
third_party/           참조 저장소 클론 (gitignore, 1장 커밋 고정)
docs/  course/         문서 / 교육 과정
```
