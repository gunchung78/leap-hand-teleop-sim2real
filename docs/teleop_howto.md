# 웹캠 텔레오퍼레이션 실습

Phase 1 파이프라인을 실제로 돌려 보는 순서. **0단계부터 차례대로.**

각 단계는 앞 단계가 통과했다는 전제로 쓰였다. 건너뛰면 문제가 생겼을 때
원인이 추적인지, 리타겟팅인지, 실기인지 가릴 수 없다. 그러라고 나눠 둔 것이다.

> 실기(`--real`)는 **5단계**다. 그 전까지는 모터가 전혀 움직이지 않는다.
> 실기를 붙이기 전에 [`real_hand_bringup.md`](real_hand_bringup.md) 를 먼저 끝내야 한다.

두 경로가 있다. 아래 0~6단계는 **단일 스크립트 경로**(`p1_3_teleop_mujoco.py`)고, 같은 코어를
ROS2 노드로 쪼갠 **ROS2 경로**는 맨 아래 "ROS2 로 돌리기"다. 처음이면 단일 스크립트로 추적과
리타겟팅을 확인하고, 그다음 ROS2 로 간다. 알고리즘은 둘이 같다.

---

## 0단계 — 준비 (최초 1회)

```bash
conda activate leap-hand
cd ~/Project/leap-hand-teleop-sim2real

bash scripts/phase1/p1_0_fetch_mediapipe_model.sh    # 7.5MB. 저장소에 없다
ls /dev/video*                           # 웹캠 확인
```

MediaPipe 1.x 는 legacy `mp.solutions.hands` 를 없앴고 tasks API 만 남았는데,
tasks API 는 모델을 패키지에 넣어 두지 않는다. 그래서 따로 받는다.
`models/hand_landmarker.task` 가 생기면 된다.

이 머신에는 `/dev/video0`, `/dev/video1` 두 개가 잡힌다. 기본은 0번이고,
엉뚱한 카메라가 열리면 `--camera 1`.

### 촬영 환경

| 항목 | 권장 |
|---|---|
| 거리 | 카메라에서 40~60 cm. 손이 화면에 다 들어오되 너무 작지 않게 |
| 배경 | 손과 대비되는 단색이 좋다. 얼굴이 같이 잡혀도 상관없다 |
| 조명 | 정면광. 역광이면 검출률이 급락한다 |
| 손 | 손 전체가 화면에 들어오게. 손가락을 카메라 쪽으로 굽히면 서로 가려져 깊이가 흔들린다 |

---

## 1단계 — 추적만 확인 (로봇 없음)

```bash
python scripts/phase1/p1_1_check_hand_tracking.py
```

창이 뜨고 손 위에 골격이 그려진다. 빨간 점 8개가 리타겟팅에 실제로 쓰는 점이다
(각 손가락의 DIP·TIP, 엄지는 IP·TIP). `q` 로 종료.

### 확인할 것

- [ ] 좌상단에 **`Right`** 로 잡히는가 (오른손을 들었을 때)
- [ ] 손바닥 폭(검지 MCP ↔ 약지 MCP)이 **흔들리지 않는가** — 값 자체보다 표준편차가 중요하다.
      성인 손이면 40~60 mm 범위에 들어오고, 표준편차가 3 mm 를 넘으면 조명·거리를 손본다
- [ ] 검출률이 90% 이상인가
- [ ] 손가락을 하나씩 굽힐 때 빨간 점이 따라오는가

### 오른손인데 `Right` 가 아니라면 — `--mirror`

```bash
python scripts/phase1/p1_1_check_hand_tracking.py --mirror
```

MediaPipe 의 handedness 판정은 **입력이 거울상이 아니라고** 가정한다. 셀피처럼
좌우가 뒤집힌 영상을 넣으면 라벨이 반대로 나오고, 동시에 world 랜드마크의
손대칭(chirality)도 뒤집힌다. 그대로 리타겟팅하면 로봇 손이 **손등 쪽으로 굽는
거울상**이 된다.

라벨이 맞으면 좌표계도 맞다. 그래서 이 확인이 곧 좌표계 정합 검사다.
여기서 `--mirror` 가 필요했다면 이후 모든 명령에도 붙여야 한다.

### 종료 시 출력

```
프레임 320, 검출 305 (95%)
손바닥 폭 평균 47.3 mm, 표준편차 1.8 mm
LEAP 손바닥 폭이 90.9 mm 이므로 스케일은 약 1.92 배가 된다.
```

(숫자는 예시다. 손 크기와 카메라에 따라 달라진다.)

이 스케일 값을 기억해 둔다. 3단계에서 로봇이 과하게/모자라게 굽으면
`--scale` 로 이 값을 직접 조정하게 된다.

---

## 2단계 — 기하 검증 (카메라 없음)

```bash
python scripts/phase1/p1_2_test_retarget_roundtrip.py
```

로봇을 알려진 자세에 두고, 그 자세에서 가짜 사람 랜드마크를 만들어(사람 손
크기로 축소 + **임의 회전·평행이동**) 리타겟터에 넣는다. 좌표계 구성이 옳다면
준 변환이 전부 상쇄되어 원래 자세가 돌아와야 한다.

기대 출력:

```
손끝 잔차     평균   0.09 mm    최대   0.98 mm
판정: 통과 — 좌표 변환과 IK 가 일관적이다
```

**언제 돌리나.** 리타겟팅 코드를 건드린 뒤. 손끝 잔차가 mm 대로 올라가면
좌표계나 IK 를 깨뜨린 것이다. 관절각 오차는 최대 60° 넘게 나와도 정상이다
(IK 중복성 — pip/dip 조합이 달라도 같은 손끝 위치가 나온다).

이 시험은 **좌표 변환과 IK 가 서로 맞물리는지**만 본다. 사람 손과 LEAP 의 비율
차이에서 오는 리타겟팅 품질은 3단계에서 눈으로 봐야 한다.

---

## 3단계 — 시뮬만 텔레오퍼레이션

```bash
python scripts/phase1/p1_3_teleop_mujoco.py
```

창이 둘 뜬다. 카메라 창과 MuJoCo 뷰어. 손을 카메라에 보이면 로봇이 따라간다.
`q`(카메라 창 포커스) 또는 Ctrl-C 로 종료.

### 확인할 것

- [ ] **손가락 대응이 맞는가.** 검지를 굽히면 로봇 검지가 굽는가
- [ ] **굽힘/벌림이 안 바뀌었는가.** 손가락을 벌리면 로봇도 벌어지는가
- [ ] **거울상이 아닌가.** 손바닥 쪽으로 굽어야 한다. 손등 쪽으로 굽으면 `--mirror`
- [ ] 엄지가 대충이라도 따라가는가 (가장 부정확한 부분이다)
- [ ] 손을 화면 밖으로 빼면 **자세를 유지**하다가 1.5초 뒤 천천히 펴지는가

### 출력 읽는 법

```
 26.8 fps  검출 142/150  IK 잔차  0.31 mm  처리 12.4 ms  접촉  0
```

| 항목 | 의미 | 정상 |
|---|---|---|
| fps | 전체 루프 | 25~30 (카메라 상한) |
| 검출 | 손이 잡힌 프레임 / 전체 | 90% 이상 |
| IK 잔차 | 목표 손끝과 실제 손끝의 거리 | 1 mm 이하 |
| 처리 | 추적 + 리타겟 + IK | 12~15 ms |
| 접촉 | MuJoCo 충돌 접촉점 수 | 손가락끼리 안 부딪히면 0 |

**IK 잔차가 크면** 사람 손 자세가 LEAP 이 도달할 수 없는 곳을 가리키는 것이다
(예: 손가락을 옆으로 크게 벌림 — LEAP 벌림 범위는 ±1.047 rad 뿐이다).
자세 탓이지 코드 탓이 아니다.

**접촉이 계속 잡히면** 손가락끼리 부딪히고 있다. Phase 0 에서 확인했듯 LEAP 은
편 자세에서 ±7.7° 밖에 못 벌린다. 실기를 붙이기 전에 이 자세를 피하는 게 좋다.

---

## 4단계 — IK 목표점 보기 (진단용)

```bash
python scripts/phase1/p1_3_teleop_mujoco.py --pybullet-gui
```

PyBullet 창이 하나 더 뜨고, IK 가 쫓고 있는 **목표점 8개**가 색 구슬로 보인다.
손가락마다 옅은 색이 DIP, 진한 색이 TIP (검지 빨강, 중지 초록, 약지 파랑, 엄지 노랑).

로봇 손끝이 구슬을 못 따라가면 도달 불가능한 목표이고, 구슬 자체가 엉뚱한 데
있으면 좌표 변환 문제다. **증상을 이 둘로 가르는 게 이 모드의 목적이다.**

무거우니 진단할 때만 켠다.

---

## 5단계 — 튜닝

기본값으로 잘 되면 건너뛴다.

| 증상 | 조정 |
|---|---|
| 로봇이 덜덜 떨림 | `--smoothing 0.2` (기본 0.4, 작을수록 부드럽고 느리다) |
| (ROS2) 로봇이 떨림 | 기본값이 이미 `kP 400 smoothing 0.2 deadband 1.0 restart_mm 50 pip_target true tip_mode axis` (2026-08-23 라이브 확인, 하나만으로는 안 됐다). `p1_4_teleop_metrics.py` 의 정지 떨림 블록으로 출처 확인 |
| 반응이 굼뜸 | `--smoothing 0.7` |
| 손이 확확 튐 | `--max-speed 4` (기본 8 rad/s) |
| 로봇이 과하게 굽음 | `--scale 2.0` 처럼 1단계 값보다 작게 |
| 로봇이 덜 굽음 | `--scale 2.5` 처럼 크게 |
| 검출이 자꾸 끊김 | 조명·거리 먼저. 그래도면 `--width 1280 --height 720` |

`--distal-mode scaled` 는 말단 마디 길이 보정을 끄고 사람 손에서 그대로 스케일한다.
기본값(`leap`)과 비교해 볼 때만 쓴다. 대개 기본이 낫다.

손을 놓쳤을 때 자세를 계속 유지시키려면 `--release-after 0`.

---

## 6단계 — 실기 붙이기

> **전제.** [`real_hand_bringup.md`](real_hand_bringup.md) 의 3~5단계를 통과했어야 한다.
> 매핑이 실기에서 확인되지 않은 상태로 텔레오퍼레이션을 붙이면 안 된다.

### 6-1. 사전 점검

```bash
# Dynamixel Wizard 가 떠 있으면 종료할 것
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
python scripts/phase0/p0_2_preflight_real_hand.py
```

토크를 켜지 않고 연결만 확인한다. 손은 힘없이 늘어져 있어야 정상이다.

### 6-2. 손을 편 자세로 시작

```bash
python scripts/phase1/p1_3_teleop_mujoco.py --real
```

**카메라 앞에 손을 보이기 전에** 로봇이 영점(편 자세)에 있는지 확인한다.
시작하자마자 굽힌 손을 들이대면 로봇이 급하게 따라간다. 속도 상한 8 rad/s 가
걸려 있지만 천천히 시작하는 게 낫다.

처음에는 **손가락 하나만** 굽혀 본다. 검지부터. 그 다음 중지, 약지, 엄지.
전부 확인한 뒤에 주먹을 쥔다.

### 6-3. 지켜볼 것

전류가 임계(300 mA)를 넘으면 출력에 이렇게 뜬다.

```
 26.1 fps  검출 150/150  IK 잔차  0.42 mm  처리 12.1 ms  접촉  2  ! 전류 초과 [(0, 318.0)]
```

이때 스크립트는 **명령을 그 자리에 얼린다.** 스톨 상태에서 계속 밀면 Lite 의
플라스틱 기어가 상하기 때문이다. 손 자세를 풀면 자동으로 다시 따라간다.

`접촉` 이 0 이 아닌 채로 전류 초과가 같이 뜨면 손가락끼리 부딪히는 것이다.
시뮬에서 이미 보이므로 실기 전에 3단계에서 걸러내는 게 좋다.

### 6-4. 안전 수칙

| 상황 | 대처 |
|---|---|
| 빨간 LED 점멸 | 과부하. **전원 사이클** 외에는 복구 안 됨 |
| 전류 초과가 계속 뜸 | Ctrl-C. 자세를 바꾸거나 `--kp 400` |
| 진동 / 떨림 | `--kp 400` 과 `--smoothing 0.2` |
| 그립 자세 장시간 유지 | **하지 말 것.** 연속 정격 0.10 N·m 는 스톨의 1/5 |

Ctrl-C 로 나가면 스크립트가 영점으로 되돌린 뒤 토크를 끈다.

---

## 문제 해결

**`카메라 0 를 열 수 없다`**
다른 프로그램이 웹캠을 쓰고 있다. `fuser /dev/video0` 로 확인.
`--camera 1` 로 다른 장치를 시도.

**`MediaPipe 모델이 없다`**
`bash scripts/phase1/p1_0_fetch_mediapipe_model.sh`

**손이 한 번도 안 잡힘**
조명(역광 아닌지), 거리(40~60 cm), 손이 화면에 다 들어오는지. `--hand Left` 를
쓰고 있지 않은지. 그래도 안 되면 `--mirror`.

**로봇 손이 손등 쪽으로 굽음 (거울상)**
`--mirror` 를 토글한다. 1단계에서 `Right` 로 잡히는지부터 확인.

**손가락 대응이 어긋남 (검지를 굽혔는데 중지가 움직임)**
리타겟팅이 아니라 매핑 문제다. `python scripts/phase0/p0_3_sweep_joints.py --joints 0` 으로
Phase 0 매핑부터 다시 확인. 시뮬에서 이미 어긋나면 코드 회귀다.

**로봇이 계속 덜덜 떨림**
`--smoothing 0.2`. 실기라면 `--kp 400` 도 같이.

**fps 가 10 아래**
`--no-window` 로 카메라 창을 끄거나 `--no-viewer` 로 MuJoCo 뷰어를 끈다.
`--pybullet-gui` 를 켜 뒀다면 그것부터 끈다.

**IK 잔차가 계속 5 mm 이상**
사람 손 자세가 LEAP 범위 밖이다. `--pybullet-gui` 로 목표 구슬 위치를 본다.
구슬 자체가 로봇에서 멀리 떨어져 있으면 `--scale` 이 잘못된 것이다.

---

## ROS2 로 돌리기

단일 스크립트와 같은 코어(`leap_hand_mapping`)를 노드 넷으로 쪼갠 것이다. 아키텍처와 노드 표는
README "Phase 1 — ROS2 통합".

### 준비 (최초 1회)

```bash
conda activate leap-hand
pip install -e .
pip install empy==3.3.4 lark catkin_pkg colcon-common-extensions
bash ros2_ws/setup_upstream.sh                  # third_party/LEAP_Hand_API 가 있어야 한다
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install                  # conda 의 colcon 이어야 한다 (which colcon 로 확인)
source install/setup.bash
```

매 터미널마다: `conda activate leap-hand && source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash`.
`ros2 run` 은 PATH 의 `python3` 을 쓰므로 conda base(3.13)가 앞에 오면 `rclpy` 가 안 뜬다.

### 시뮬만

```bash
ros2 launch leap_teleop sim.launch.py                  # 카메라 창(손 위치 틀) + MuJoCo 뷰어
ros2 launch leap_teleop sim.launch.py mirror:=true     # 오른손이 Right 로 안 잡히면
ros2 launch leap_teleop sim.launch.py show:=false viewer:=false   # 헤드리스
```

다른 터미널에서:

```bash
ros2 topic hz /hand/landmarks          # 손이 보이는 동안 ~30
ros2 topic hz /leap/joint_cmd
ros2 topic echo --once /leap/joint_cmd # 이름 16개가 MuJoCo 순서인지
python scripts/phase1/p1_4_teleop_metrics.py --seconds 20   # Hz / 종단 지연 / 추종 오차 표
```

### 실기 (데드맨이 있다)

```bash
ros2 launch leap_teleop real.launch.py fake:=true      # 먼저 가짜 실기로 배선 확인
ros2 launch leap_teleop real.launch.py                 # 실기. kP 400 / curr_lim 350 / port by-id 가 기본
ros2 topic pub --once /teleop/enable std_msgs/msg/Bool "data: true"     # 이걸 줘야 움직인다
ros2 topic pub --once /teleop/enable std_msgs/msg/Bool "data: false"    # 그 자리에서 멈춘다
```

- **SPACE** (카메라 창 포커스) = 데드맨 토글. 창 오른쪽 위 `ROBOT ON/OFF`. CLI 로 바꿔도 창에 반영된다.
- enable 하면 브리지가 먼저 실기 현재 자세를 읽고 **거기서부터** 1 rad/s 로 합류한 뒤 8 rad/s 로 바뀐다.
  켜자마자 점프하지 않는다. 런치 시 토크가 켜질 때도 현재 자세를 유지한다(`hold_on_start`).
- 어떤 모터든 |전류| > 400 (리더 단위, 한계 350 은 ~469) 이 3표본(100 ms) 연속이면 얼린다. 얼릴 때
  실기의 **현재 자세를 한 번 명령해 힘을 뺀다** (막히던 목표를 그대로 두면 모터가 계속 밀어 영영 안 풀린다 —
  2026-08-23 `rf_rot` 468 교착 실측). 전부 300 밑이면 풀고 1 rad/s 로 다시 합류. 손이 같은 자세면 또 얼린다 —
  그 관절이 어디에 걸리는지 눈으로 볼 것. 움직이는 순간의 300~380 과도 전류는 얼리지 않는다.
- 포트가 다르면 `port:=/dev/serial/by-id/...`. `ls /dev/serial/by-id` 로 확인.
- `curr_lim` 은 350 이다. **올리지 말 것** (Lite 플라스틱 기어).

실기 첫 확인 순서: 데드맨 OFF 상태에서 `/real/joint_states` 가 30 Hz 로 오는지 → enable →
`python scripts/phase1/p1_5_step_response.py --source real --joints if_mcp` (관절 하나, 20°) → 전체.
계단 응답은 카메라 없이 `tracker:=false` 로 띄우고 잰다(손이 보이면 명령이 섞인다).

### 문제 해결 (ROS2)

| 증상 | 확인 |
|---|---|
| `ModuleNotFoundError: rclpy._rclpy_pybind11 ... cpython-313` | conda base 가 PATH 앞. `conda activate leap-hand` |
| 노드가 `leap_hand_mapping` 을 못 찾음 | `/usr/bin/colcon` 으로 빌드됨. conda 의 colcon 으로 다시 `colcon build` |
| `No module named 'em'` (빌드) | `pip install empy==3.3.4 lark catkin_pkg` |
| 브리지가 "/leap_position 서비스를 기다리는 중" | leaphand_node 가 안 떴다. 포트/전원/Dynamixel Wizard 점유 확인 |
| 실기가 안 움직임 | 데드맨. `ros2 topic echo /teleop/enable` |

