# Phase 1 구현 계획 — 웹캠 텔레오퍼레이션 + 디지털 트윈 (ROS2)

> 작성일 2026-08-20. Phase 0(관절 매핑 확정) 완료 후.
> 이전 Phase 1 구현은 백지화하고 다시 짠다. 근거는 1장.

> **2026-08-21 갱신 — 방향 전환.** 이 계획은 "백지 재작성 + dex-retargeting 기본" 전제로
> 썼다. 그 뒤 실측과 라이브 확인을 거쳐 전제가 바뀌었다:
> - 리타겟터는 **직접 구현한 손끝 IK(`leap_hand_mapping/retarget.py`, 커밋 `7cccfdd` 상태)** 를
>   쓴다. dex 는 PIP 를 안 쓰고(갈고리), 오늘 시도한 개선들은 녹화 지표로는 이겼지만
>   라이브에서 졌다. geon 의 라이브 판정: "이게 가장 잘된다." 근거는 README 머리와 `bak/`.
> - 그래서 **코어를 다시 쓰지 않는다.** 1장·S0(archive)·S2(코어 재작성)·S2.5(설정 선정)·5.2 의
>   dex 설명·9장의 "직접 IK 를 다시 쓰지 않는다" 는 **폐기**. 남는 것은 3장 아키텍처,
>   5.1/5.3/5.4/5.5 노드 사양, 6장 지표, S1/S3~S7 순서다.
> - 코어는 `pip install -e .` 로 설치하고(`pyproject.toml`) 노드는 그것을 import 만 한다.
> - 업스트림 `leap_hand` 는 저장소에 복사하지 않는다(CC BY-NC). `ros2_ws/setup_upstream.sh` 가
>   `third_party/` 에서 가져와 `patches/leap_hand_port_param.patch` 를 얹는다.

---

## 1. 왜 다시 짜는가

이전 구현(`archive/phase1_v1/`)은 동작은 했지만 두 가지가 어긋나 있었다.

**아키텍처가 인수인계 문서 7장과 다르다.** 문서가 그린 것은 토픽으로 연결된
ROS2 노드 그래프이고, 이력서 문구도 *"ROS2로 통합해 단일 명령으로 동시 제어"* 다.
실제로 만든 것은 카메라·IK·MuJoCo·실기를 한 프로세스에 넣은 329줄짜리 스크립트였다.
로드맵 Day 4(ROS2 래핑)와 Day 5(정량 지표)가 통째로 비어 있다.

**직접 구현한 리타겟터에 시간을 다 썼다.** `retarget.py` 903줄을 쓰고 나서
같은 녹화 데이터로 재 보니 upstream `dex-retargeting` 이 이겼다 —
핀치가 되고(30mm vs 85mm), 더 빠르고(3.7 vs 6.5 ms), 캘리브레이션이 필요 없다.
기본값도 이미 dex 로 되돌린 상태다. 그 코드를 계속 들고 갈 이유가 없다.

그래서 이번 원칙은 하나다. **업스트림이 하는 일은 업스트림에게 맡기고,
우리는 연결과 안전과 측정만 쓴다.**

남기는 것은 Phase 0 산출물인 `joint_map.py` 뿐이다.

---

## 2. 사전 검증 — 이미 확인한 것

계획을 세우기 전에 위험한 가정 네 개를 실제로 돌려서 확인했다.

| 확인 항목 | 결과 |
|---|---|
| conda `leap-hand`(py3.10)에서 rclpy import + 퍼블리시 | **동작** (`PYTHONPATH`에 `/opt/ros/humble` 이 이미 잡혀 있음) |
| 업스트림 `LEAP_Hand_API/ros2_module` colcon 빌드 | **성공** — 단, conda 환경에 `empy==3.3.4 lark catkin_pkg` 필요 |
| 빌드된 `leap_hand.srv` 를 conda python 에서 import | **성공** (`ros2 pkg executables leap_hand` 도 정상) |
| `dex-retargeting` LEAP dexpilot 설정 로드 + 최적화 | **성공**, 1.8 ms/frame, `scaling=1.6`, 관절 이름 `['1','0','2',...]` |

확인된 걸림돌 세 가지도 같이 적어 둔다.

1. **mediapipe 1.0.1 에는 `mp.solutions.hands` 가 없다.** legacy API 가 제거됐다.
   그래서 `dex-retargeting` 의 예제 `SingleHandDetector` 를 그대로 못 쓴다.
   → tasks API 로 추적기를 얇게 다시 쓰되, **MANO 프레임 추정 수식은 예제에서 그대로 가져온다**
   (`estimate_frame_from_hand_points`, `OPERATOR2MANO_RIGHT`). 알고리즘을 새로 만들지 않는다.
2. **업스트림 `leaphand_node.py` 는 포트를 `/dev/ttyUSB0→1→2` 로 하드코딩한다.**
   지금 이 머신에는 FTDI 장치가 3개 붙어 있다.
   ```
   usb-FTDI_USB__-__Serial_Converter_FTBIN91W-if00-port0 -> ttyUSB0   (U2D2)
   usb-FTDI_Dual_RS232-HS-if00-port0                     -> ttyUSB1
   usb-FTDI_Dual_RS232-HS-if01-port0                     -> ttyUSB2
   ```
   오늘은 U2D2 가 `ttyUSB0` 라서 우연히 맞지만, 열거 순서는 부팅마다 바뀔 수 있다.
   → **`port`/`baudrate` 를 ROS 파라미터로 받는 최소 패치**를 `patches/` 에 커밋하고
   런치에서 `/dev/serial/by-id/...` 를 넘긴다 (문서 4.3).
3. **업스트림 런치 파일이 `curr_lim: 500.0` 을 준다.** Lite 에 그대로 쓰면 기어가 상한다
   (문서 4.1). 노드 자체의 기본값은 350 이므로 **우리 런치에서 350 을 명시**한다.
   `kP` 도 업스트림 기본 800 대신 600 으로 낮춰 준다 (문서 4.2, `main.py` 값).

---

## 3. 아키텍처

인수인계 문서 7장 그대로다.

```
 [웹캠]
    │
    ▼
 tracker_node ──── /hand/landmarks ────────────────┐   sensor_msgs/PointCloud2
    │              (21점, MediaPipe world, m)      │   header.stamp = 촬영 시각
    │                                              │
    ▼                                              ▼
 retarget_node                                  (rviz2 로 바로 보임)
    │  dex-retargeting dexpilot, scaling 1.6
    │
    ├──── /leap/joint_cmd ─────────────────────────┐   sensor_msgs/JointState
    │     16-DoF, MuJoCo 관절 순서/규약            │   header.stamp = 촬영 시각 그대로
    │                                              │   (전파해서 종단 지연을 잰다)
    ▼                                              ▼
 sim_node                                      hand_bridge_node   ← 안전 래퍼
  MuJoCo 디지털 트윈                              클립 + 속도제한 + 전류 감시 + 데드맨
  ctrl 에 명령 → 물리 스텝                            │
    │                                                ▼
    └── /sim/joint_states                        /cmd_leap   sensor_msgs/JointState
                                                     │       (LEAPhand 규약, 모터 ID 순서)
                                                     ▼
                                              leaphand_node.py   ← 업스트림 그대로
                                                     │
                                          /leap_pos_vel_eff 서비스
                                                     │
                                                     ▼
                                              /real/joint_states  (MuJoCo 규약으로 환산,
                                                                   effort = 전류 mA)
```

설계에서 지키는 것:

- **명령 토픽 하나가 시뮬과 실기를 동시에 먹인다.** 이게 "디지털 트윈"이라 부를 수 있는 근거다.
  `--real` 플래그가 아니라 런치 파일 선택으로 갈린다.
- **`header.stamp` 는 카메라 촬영 시각을 끝까지 전파한다.** ROS 규약상 stamp 는
  "데이터가 취득된 시각"이므로 이게 맞고, 덕분에 어느 노드에서든 `now - stamp` 로
  종단 지연을 잴 수 있다. README 가 지적한 "명령 직후 바로 읽어 통신 지연이 섞인다"는
  문제가 여기서 해소된다.
- **알고리즘은 ROS 를 모른다.** 추적·리타겟팅은 `leap_hand_mapping/` 에 순수 파이썬으로 두고
  노드는 그걸 호출만 한다. 카메라·ROS 없이 테스트할 수 있어야 하고,
  Phase 2 에서 학습한 정책을 배포할 때도 같은 코어를 쓴다.
- **QoS 는 sensor 계열 BEST_EFFORT, depth 1.** 텔레오퍼레이션에서 밀린 프레임은
  쓰레기다. 큐에 쌓아 지연을 만들지 않고 최신 것만 쓴다.

---

## 4. 저장소 구조 (완료 후 모습)

```
leap_hand_mapping/            순수 파이썬 코어. pip install -e . 로 설치
  joint_map.py                Phase 0 산출물. 손대지 않는다
  tracking.py                 [신규] MediaPipe tasks API -> 21 랜드마크
  retargeting.py              [신규] dex-retargeting 어댑터 (관절 순서 재배열 + 클립)
pyproject.toml                [신규] 코어를 설치 가능하게

ros2_ws/src/
  leap_hand/                  업스트림 ros2_module 복사본 + port 파라미터 패치
  leap_teleop/                [신규] 우리 ament_python 패키지
    leap_teleop/
      tracker_node.py
      retarget_node.py
      sim_node.py
      hand_bridge_node.py
    launch/
      sim.launch.py           카메라 + 리타겟 + MuJoCo (실기 없음)
      real.launch.py          위 + 브리지 + 업스트림 leaphand_node
patches/
  leap_hand_port_param.patch  업스트림에 가한 유일한 변경. 재현 가능하게 커밋

scripts/
  phase0/                     그대로
  phase1/                     [신규] 검증 · 지표 스크립트 (5장)
archive/phase1_v1/            이전 구현 전체 (retarget.py 903줄 포함)
docs/
  phase1_plan.md              이 문서
  teleop_howto.md             다시 씀 (ROS2 절차로)
```

---

## 5. 노드별 사양

### 5.1 `tracker_node`
- 입력: `/dev/videoN`. 출력: `/hand/landmarks` (PointCloud2, 21점 xyz32).
- MediaPipe tasks API `HandLandmarker`, VIDEO 모드, `num_hands=2` 로 잡고 handedness 로 고른다.
- 거울상 함정: 좌우 반전 입력은 handedness 라벨과 world 랜드마크의 손대칭을 **동시에** 뒤집는다.
  오른손이 `Right` 로 안 잡히면 영상이 뒤집힌 것 → `mirror` 파라미터.
  **라벨 필터가 곧 좌표계 정합 검사다.**
- 손을 못 잡으면 **아무것도 publish 하지 않는다.** 판단은 하류가 한다.
- 파라미터: `camera`, `width`, `height`, `hand`, `mirror`, `model_path`.
- PointCloud2 를 고른 이유: Header 가 있어 지연을 잴 수 있고, rviz2 에서 공짜로 보인다.
  커스텀 메시지를 만들지 않으므로 `leap_teleop` 은 순수 ament_python 으로 남는다.

### 5.2 `retarget_node`
- 입력 `/hand/landmarks` → 출력 `/leap/joint_cmd` (JointState, `name=MUJOCO_JOINT_NAMES`).
- `dex-retargeting` `leap_hand_right_dexpilot.yml` 을 **덮어쓰지 않고** 쓴다.
  `scaling_factor=1.6` 은 upstream 값이고 두 팀이 독립적으로 수렴한 값이다.
  바꿔 보고 싶으면 파라미터로 임시로만.
- 관절 순서는 **인덱스가 아니라 이름으로** 맞춘다 (dex-retargeting FAQ).
  dex 의 관절 이름이 실기 모터 ID 문자열이므로 `MUJOCO_TO_MOTOR` 가 그대로 쓰인다.
- 출력 전에 `jm.clip_mujoco()`.
- 손 유실 처리: `hold_timeout`(기본 1.5 s) 까지는 **직전 자세 유지**,
  넘어가면 1초에 걸쳐 영점으로 램프. 프레임 밖으로 나갈 때마다 손이 펴지면 물건을 놓친다.

### 5.3 `sim_node`
- 입력 `/leap/joint_cmd` → MuJoCo `data.ctrl`. 출력 `/sim/joint_states`.
- `qpos` 를 직접 넣지 않는다. **액추에이터에 명령을 주고 물리를 돌린다.**
  충돌과 추종 지연이 보여야 트윈 구실을 한다 (Phase 0 의 벌림 충돌이 이 방식으로 재현됐다).
- 뷰어는 `mujoco.viewer.launch_passive`, 물리 스텝은 ROS 타이머.
- 모델: `third_party/mujoco_menagerie/leap_hand/scene_right.xml`.

### 5.4 `hand_bridge_node` — 안전 래퍼
로드맵 Day 4 의 핵심. 실기로 나가는 유일한 경로이므로 여기 다 모은다.

| 장치 | 내용 |
|---|---|
| 데드맨 | `/teleop/enable` (Bool) 가 true 일 때만 `/cmd_leap` 을 낸다. **기본 false** |
| 범위 클립 | `LIMITS_INTERSECTION_MJ` — MJCF 와 실기 규약의 교집합 |
| 속도 제한 | 8 rad/s. 60 Hz 타이머로 목표를 향해 램프 |
| 전류 동결 | `/leap_pos_vel_eff` 를 30 Hz 폴링, `|effort| > 300 mA` 인 모터가 있으면 **명령을 그 자리에 얼린다.** 스톨 상태에서 계속 밀면 Lite 기어가 상한다 |
| 규약 변환 | `jm.safe_leaphand_command()` — 순서 치환 + π 오프셋 |
| 피드백 | 읽은 값을 MuJoCo 규약으로 되돌려 `/real/joint_states` 로 publish |

- 명령 60 Hz / 조회 30 Hz. 업스트림 readme 가 "90 samples/s 아래로" 권고한다.
- 구현 주의: 서비스 호출은 `call_async` + `MultiThreadedExecutor`.
  단일 스레드에서 콜백 안에 동기 호출을 넣으면 데드락이다.

### 5.5 업스트림 `leaphand_node`
우리가 실기 드라이버를 쓰지 않는다는 뜻이다. 런치에서 넘기는 값만 우리 것:

```python
parameters=[{'kP': 600.0, 'kI': 0.0, 'kD': 200.0,
             'curr_lim': 350.0,                       # Lite. 올리지 말 것
             'port': '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBIN91W-if00-port0'}]
```

---

## 5.6 ★ 최대 쟁점 — "로봇이 손처럼 움직이지 않는다"

이 프로젝트의 실질적 문제다. 파이프라인은 돌아갔지만 로봇 손 모양이 사람 손과 달랐다.
녹화해 둔 `thumb_capture.npz`(사람 손 4자세, 프레임 360여 개)로 원인을 측정했다.

### 무엇이 잘못되고 있었나

사람 굽힘각(인접 뼈 벡터 사이 각)과 로봇 관절각을 나란히 놓고,
**편 손 → 주먹으로 갈 때의 변화량**을 본다. 각도 정의가 서로 달라도 변화량은 비교된다.

| 관절 | 사람 | vector | dexpilot (현 기본) | position |
|---|---:|---:|---:|---:|
| **mcp (손허리, 주먹 쥘 때 가장 크게 굽는 곳)** | **+58.7°** | **-22.0°** | **-19.6°** | **+33.8°** |
| pip | +74.9° | +89.1° | +45.0° | +35.3° |
| dip | +66.0° | +90.8° | +102.6° | +26.5° |

사람이 주먹을 쥘 때 MCP 를 59° 굽히는데, **현재 기본값(dexpilot)은 MCP 를 오히려 20° 편다.**
굽힘을 전부 PIP/DIP 로 몰아넣는다. 즉 로봇은 주먹이 아니라 **갈고리**를 만든다.
"손처럼 안 움직인다"의 정체가 이것이다.

### 왜 그런가

teleop 설정(`vector`, `dexpilot`)은 **손끝 4점만** 목표로 삼는다.

```yaml
target_task_link_names: [ thumb_tip_head, index_tip_head, middle_tip_head, ring_tip_head ]
target_link_human_indices: [ [0,0,0,0], [4,8,12,16] ]
```

손끝 위치는 (MCP 굽힘 + 적당한 말림) 으로도, (MCP 편 채 + PIP/DIP 최대 말림) 으로도
거의 같게 나온다. 손가락 **모양**을 정하는 자유도가 목표에 아예 없는 것이다.
그 널스페이스 안에서 최적화는 이전 프레임에 가까운 해를 고르므로(`norm_delta` 정규화),
MCP 는 영영 굽지 않는다. dexpilot 은 여기에 더해 손끝끼리의 거리를 우선하므로
**모양 유사성을 설계상 희생한다** — 집기를 위한 선택이고, 그건 그것대로 옳다.

### 고칠 수 있는가 — 그렇다. 업스트림 안에 레버가 있다

`dex-retargeting` 은 중간 마디까지 묶는 설정을 이미 배포한다 (`configs/offline/leap_hand_right.yml`):

```yaml
type: position
target_link_names: [ thumb_tip_head, index_tip_head, middle_tip_head, ring_tip_head,
                     thumb_dip, dip, dip_2, dip_3 ]     # 손끝 4 + 중간마디 4
target_link_human_indices: [ 4, 8, 12, 16, 2, 6, 10, 14 ]
```

같은 데이터로 재면 **주먹에서 MCP 79.9° (사람 73.7°)** 로 유일하게 맞는다.
덤으로 **지터가 오히려 줄었다** — 주먹 자세 관절각 표준편차 2.30° vs vector 12.75° / dexpilot 9.99°.
제약이 늘면 널스페이스가 줄어 해가 프레임마다 갈아타지 않는다.
속도는 셋 다 1.1 ms/frame 로 차이가 없다.

> 이전 세션이 "앞마디 목표를 넣으면 지터가 는다"고 기록해 둔 것은
> 자작 IK(`ours`)의 성질이었다. 도달 불가능한 절대 위치 목표 + 재시도 시드 구조라서
> 그랬던 것이고, dex 의 nlopt 최소자승에는 그 실패 모드가 없다.

### 아직 남은 것

`position` 설정은 `scaling_factor` 가 없어(기본 1.0) **편 손에서 MCP 가 30° 과굽힘**된다.
사람 손이 LEAP 보다 작은데 절대 위치를 그대로 맞추려니 더 말아야 하는 것이다.
MANO 스케일 오프라인 데이터용 설정이라 그렇다.

| 자세 | 사람 mcp | position mcp |
|---|---:|---:|
| 편 손 | 15.0° | 46.1° |
| 주먹 | 73.7° | 79.9° |

즉 **방향은 맞고 원점이 어긋나 있다.** 배율을 주면 풀릴 가능성이 높다.

### S2.5 — 리타겟팅 설정 선정 (신규 단계, 0.5일)

느낌으로 정하지 않는다. 후보 4개를 같은 녹화 데이터로 같은 자에 올린다.

| 후보 | 목표점 | 근거 |
|---|---|---|
| A `dexpilot` | 손끝 4 + 손끝 간 거리 6 | 현 기본값. 대조군 |
| B `vector` | 손끝 4 | 손끝 사전지식 없음 |
| C **`vector` + 중간마디** | 손끝 4 + 중간마디 4 | **YAML 만 수정.** upstream 스키마 그대로, `scaling_factor` 유지 |
| D `position` + 배율 | 손끝 4 + 중간마디 4 (절대 위치) | upstream offline 설정 + 배율 실험 |

C 가 가장 유망하다 — 모양을 잡는 제약(중간마디)과 크기를 맞추는 기구(배율)를
둘 다 가진 유일한 후보이고, **알고리즘을 한 줄도 새로 쓰지 않는다.**

재는 것:
1. **자세 충실도** — 위 표의 변화량 비교. 자세를 늘려 다시 녹화 (개별 손가락 굽힘, 벌림, 손목 회전 포함)
2. **기능** — 엄지-검지 손끝 간격 (집을 수 있는가). 기존 `compare_retargeters` 지표
3. **지터** — 정지 자세 관절각 표준편차
4. **눈** — MuJoCo 렌더와 웹캠 영상을 나란히 붙인 영상. 최종 판정은 사람 눈이 한다

산출물은 `scripts/phase1/p1_2_compare_retargeting.py` 와 README 의 표.

### 정직하게, 이 단계로도 안 풀리는 것

- **엄지.** 사람과 LEAP 은 엄지 축 배치가 근본적으로 다르다(`th_axl` 은 회전축이
  손끝을 거의 지난다). 각도 대응 자체가 성립하지 않아 위 측정에서도 뺐다.
  엄지는 "모양"이 아니라 **"검지와 만나는가"**로만 평가하는 게 맞다.
- **단안 깊이.** MediaPipe world 랜드마크의 z 는 단일 RGB 에서 학습으로 추정한 값이라
  약하다. 손가락을 카메라 쪽으로 굽히는 동작이 정확히 그 z 축이다.
  리타겟터로는 못 고친다. 이 데이터(전부 손을 정면으로 든 자세)로는 영향을 가를 수 없다.
  → **판별 시험**: 손 모양을 고정한 채 손목만 돌린다. 관절각이 흔들리면 센서·프레임 문제,
  안 흔들리면 리타겟팅 문제. S2.5 녹화에 이 자세를 반드시 넣는다.
  센서가 상한이라고 나오면 그때 선택지는 웹캠 2대 삼각측량이나 더 무거운 손 복원 모델이고,
  그건 Phase 1 범위 밖으로 뺀다.

---

## 6. 검증과 지표

"손 흔들어 보니 따라오더라"는 검증이 아니다. 전부 재실행 가능한 스크립트로 만든다.

| 스크립트 | 카메라 | 실기 | 무엇을 재는가 |
|---|:--:|:--:|---|
| `p1_1_check_tracking.py` | O | X | 추적이 되는지, 손바닥 폭 실측, handedness/거울 확인 |
| `p1_2_test_retarget_offline.py` | X | X | 녹화된 bag 을 리타겟터에 다시 넣는다. **회귀 시험** |
| `p1_3_step_response.py` | X | O | 계단 명령 → 관절별 **지연·상승시간·정상상태오차**. 카메라 없이 |
| `p1_4_teleop_metrics.py` | O | 선택 | 라이브 지표: 각 토픽 Hz, 종단 지연, sim-real 추종 오차 |
| `p1_5_pinch_test.py` | O | 선택 | **기능 지표**: 엄지-검지 손끝 간격. 집을 수 있는가 |

- `p1_3` 이 README 가 지적한 문제("명령 직후 바로 읽어 통신 지연이 섞인다")를 정면으로 푼다.
  계단 응답은 지연과 추종을 분리해서 준다.
- `p1_5` 는 내가 만든 내부 지표가 아니라 **기능**을 잰다. 지표가 사람 눈과 갈리면 지표를 의심한다.
- 데이터 로깅은 공짜로 딸려 온다:
  ```bash
  ros2 bag record /hand/landmarks /leap/joint_cmd /sim/joint_states /real/joint_states
  ```
  인수인계 문서 9장의 "임팩트 강화 옵션 2 — imitation learning 데모 수집 파이프라인"이 이것이다.

**README 에 넣을 숫자**: 제어 주파수(추적/명령 Hz), 종단 지연(ms),
관절별 계단 응답 지연·정상상태오차(deg), sim-real 추종 오차(deg RMS), 핀치 최소 간격(mm).

---

## 7. 작업 순서

| 단계 | 내용 | 산출물 | 예상 |
|---|---|---|---|
| **S0** | 정리 — 이전 구현을 `archive/phase1_v1/` 로, `pyproject.toml` 추가, `joint_map` 만 남기고 코어 비우기 | 커밋 1 | 0.5 h |
| **S1** | `ros2_ws` 구성. 업스트림 `leap_hand` 복사 + port 패치 + 빌드. `empy` 등 의존성을 README 에 명시 | 빌드 통과, `ros2 pkg executables` 확인 | 2 h |
| **S2** | 코어 재작성 — `tracking.py`, `retargeting.py`. ROS 없이 단위 확인 | `p1_1`, `p1_2` 통과 | 0.5 일 |
| **S2.5** | **리타겟팅 설정 선정 (5.6장).** 자세를 늘려 재녹화 → 후보 4개 측정 → 기본값 확정 | 비교표, 기본 설정 | 0.5 일 |
| **S3** | `tracker_node` + `retarget_node`. 실기·시뮬 없이 `ros2 topic hz` / `ros2 topic echo` 로만 확인 | 토픽 30 Hz | 0.5 일 |
| **S4** | `sim_node`. 여기까지가 로드맵 Day 3 (시뮬 전용) | `ros2 launch sim.launch.py` 로 손이 따라옴 | 0.5 일 |
| **S5** | `hand_bridge_node` + 업스트림 노드 연결. 데드맨 → 관절 하나 → 전체 순으로. **로드맵 Day 4** | `real.launch.py` | 1 일 |
| **S6** | 지표 스크립트 4개 + 측정. **로드맵 Day 5** | 숫자 표 | 0.5 일 |
| **S7** | `docs/teleop_howto.md` 재작성, README 갱신, 데모 영상/GIF | 커밋 | 0.5 일 |

합계 약 **4.5일**. 인수인계 문서의 Phase 1 예상(4~5일, Day 1~2 는 Phase 0 에서 선행)과 맞는다.

**진행 (2026-08-21):** S0 `5e2cb53` · S1 `5e2cb53` · S2/S2.5 폐기 · S3 `8a996a8` · S4 `28e5013` ·
S5 `7fd2d16`(fake 실기로 확인, 실기 전원 확인은 사용자) · S6 `2122027`(지표 2종 + fake/시뮬 측정) ·
S7 문서 커밋. 남은 것: 사용자의 라이브 확인(sim.launch), 실기 데드맨→관절 하나→전체, 데모 영상.

**완료 (2026-08-23):** 라이브 확인 전부 통과 — sim.launch 동일 동작, 실기 계단 16관절, 웹캠+실기 라이브
32 ms / 트윈 RMS 1.95° / 정지 떨림 <1°. 그 과정에서 추가된 것: 데드맨 SPACE 토글, hold_on_start, 합류 램프,
전류 동결(디바운스 + 힘 빼기), 떨림 기본값(kP 400 / smoothing 0.2 / deadband 1° / restart 50 mm / pip_target /
tip_mode axis), 벌림 관절 텔레옵 제한. 남은 것: 데모 영상, 엄지 IK 떨림(th_cmc 3°).

각 단계는 독립적으로 커밋한다. S5 전까지는 실기에 전원을 넣지 않는다.

---

## 8. 위험 요소와 대응

| 위험 | 대응 |
|---|---|
| conda + colcon 빌드 실패 | **확인 완료.** `empy==3.3.4 lark catkin_pkg` 를 conda env 에 설치. README 에 명시 |
| U2D2 포트가 부팅마다 바뀜 | `port` 파라미터 패치 + `/dev/serial/by-id/` 사용. `dialout` 그룹 소속이라 chmod 불필요 |
| Lite 기어 파손 | `curr_lim=350` 고정, 전류 동결, 데드맨, 속도 제한. S5 는 관절 하나부터 |
| 서비스 호출 데드락 | `MultiThreadedExecutor` + `call_async` |
| MuJoCo 뷰어와 ROS 스핀 충돌 | 뷰어는 passive, 물리는 ROS 타이머. 한 프로세스 안에서 GIL 로 직렬화 |
| mediapipe 1.x 로 예제를 못 씀 | tasks API 로 얇게 감싸되 좌표계 수식은 예제에서 그대로 인용 (2장 걸림돌 1) |
| 설정을 바꿔도 모양이 안 잡힘 | 원인이 단안 깊이라는 뜻. S2.5 의 손목 회전 시험이 가른다. 그 경우 Phase 1 범위를 '기능적 유사성'으로 좁히고 명시한다 |
| 실기 없이 진행이 막힘 | S4 까지 실기가 필요 없다. `p1_2`·`p1_3` 도 카메라/실기를 분리해 뒀다 |

---

## 9. 하지 않기로 한 것

- **직접 IK 를 다시 쓰지 않는다.** `retarget.py` 는 `archive/` 로. README 의 비교표는
  근거로 남기고, `p1_2` 가 같은 bag 으로 재현할 수 있게 해 둔다.
- **`scaling_factor` 를 기본값에서 바꾸지 않는다.** 자체 지표로 upstream 상수를 덮어썼다가
  되돌린 적이 있다(`d88ae8c`). 비교는 파라미터로 임시로만.
- **커스텀 메시지 타입을 만들지 않는다.** 표준 `sensor_msgs` 로 충분하고,
  그래야 `leap_teleop` 이 순수 ament_python 으로 남아 빌드가 가볍다.
- **리타겟팅 알고리즘을 새로 쓰지 않는다.** 5.6장의 해법은 전부 upstream YAML 설정 변경이다.
- **엄지 캘리브레이션을 넣지 않는다.** 사람마다 매번 해야 하는 절차는 재현성 비용이다.
  dex-retargeting 은 손목 프레임을 매 프레임 추정하므로 그 단계가 아예 없다.
