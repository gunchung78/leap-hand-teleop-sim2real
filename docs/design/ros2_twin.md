# ROS2 통합과 디지털 트윈 — 설계와 측정

> 성격: 설계 설명 + 실측 (Phase 1 ROS2). 절차는 `../teleop_howto.md` "ROS2 로 돌리기".


인수인계 문서 7장의 토픽 그래프 그대로다. 단일 스크립트(`p1_3_teleop_mujoco.py`)가 하던 일을
노드 넷으로 쪼갰고, **명령 토픽 하나(`/leap/joint_cmd`)가 시뮬과 실기를 동시에 먹인다.**
알고리즘은 전부 `leap_hand_mapping/`(순수 파이썬, `pip install -e .`)에 있고 노드는 토픽만 잇는다.

```
 [웹캠] -> tracker_node --/hand/landmarks (PointCloud2 21점, stamp=촬영시각)--> retarget_node
                                                                                  |
                                          /leap/joint_cmd (JointState, MuJoCo 순서, stamp 전파)
                                                  |                               |
                                                  v                               v
                                              sim_node                    hand_bridge_node (안전 래퍼)
                                        MuJoCo 디지털 트윈                  데드맨·클립·속도제한·전류동결
                                                  |                               |
                                          /sim/joint_states            /cmd_leap --> leaphand_node (업스트림)
                                                                                  |
                                                                      /leap_pos_vel_eff --> /real/joint_states
```

| 노드 | 하는 일 |
|---|---|
| `tracker_node` | 웹캠 → MediaPipe 21점 → `/hand/landmarks`. 손 없으면 publish 안 함. `show:=true` 면 카메라 창 + 손 위치 틀 |
| `retarget_node` | `LeapRetargeter`(7cccfdd 손끝 IK) 호출 → `/leap/joint_cmd`. 손 유실 1.5 s 까지 유지, 그 뒤 1 s 에 걸쳐 영점 |
| `sim_node` | `/leap/joint_cmd` → MuJoCo `ctrl` → 물리 스텝(60 Hz) → `/sim/joint_states`. qpos 를 직접 넣지 않는다 |
| `hand_bridge_node` | 실기로 나가는 **유일한 경로.** 데드맨 `/teleop/enable`(기본 false), 시작 시 실기 자세 동기, ON 직후 1 rad/s 로 **천천히 합류**한 뒤 8 rad/s, 클립, `|전류| > 400` 이 100 ms 연속이면 동결(현재 자세를 명령해 **힘을 뺀다**; 300 밑으로 내려오면 천천히 재합류), `safe_leaphand_command` 변환, `/real/joint_states` 발행 |
| `leaphand_node.py` | **업스트림 그대로**(+패치: 포트 파라미터, 시작 시 **현재 자세 유지** `hold_on_start`). 런치에서 `kP 400 / curr_lim 350 / port by-id` 만 넘긴다 |
| `fake_hand_node` | 업스트림 인터페이스만 흉내내는 더미. 실기 없이 배선·데드맨·전류 동결 시험 |

`header.stamp` 는 카메라 촬영 시각을 끝까지 전파한다. 어느 노드에서든 `now - stamp` 가 종단 지연이다.
QoS 는 sensor 계열 BEST_EFFORT depth 1 — 밀린 프레임은 쓰레기라 최신 것만 쓴다.

## 빌드와 실행

```bash
conda activate leap-hand                     # 반드시. ros2 run 은 PATH 의 python3 을 쓴다
pip install -e .                             # leap_hand_mapping
pip install empy==3.3.4 lark catkin_pkg colcon-common-extensions   # (최초 1회) 아래 참고
bash ros2_ws/setup_upstream.sh               # third_party 의 업스트림 leap_hand 복사 + 패치
cd ros2_ws && source /opt/ros/humble/setup.bash && colcon build --symlink-install && source install/setup.bash

ros2 launch leap_teleop sim.launch.py                        # 카메라 + 리타겟 + MuJoCo (실기 없음)
ros2 launch leap_teleop real.launch.py fake:=true            # + 브리지 + 가짜 실기 (배선 시험)
ros2 launch leap_teleop real.launch.py                       # + 브리지 + 실기
ros2 topic pub --once /teleop/enable std_msgs/msg/Bool "data: true"    # 데드맨 ON — 이걸 줘야 움직인다
```

**데드맨 버튼:** 카메라 창에 포커스를 두고 **SPACE** 를 누르면 ON/OFF 가 토글되고 창 오른쪽 위에
`ROBOT ON` / `ROBOT OFF` 가 뜬다. OFF 는 그 자리에서 정지(영점으로 튀지 않는다). 창을 닫아도 OFF 를 보낸다.

**시작할 때 확 움직이지 않게 한 두 가지:**
1. 업스트림 `leaphand_node` 는 토크를 켜자마자 편 손(π)으로 가서, 손이 굽어 있으면 튀었다.
   패치 `hold_on_start`(기본 true)가 토크를 켜기 **전에** 현재 자세를 목표로 써 넣어 그 자리를 유지한다.
2. 데드맨을 켤 때 카메라가 이미 다른 자세를 보고 있으면 8 rad/s 로 달려갔다. 이제 ON 직후에는
   `engage_speed` 1 rad/s 로 합류하고(`real.launch.py engage_speed:=`), 모든 관절이 목표에 든 뒤에야
   8 rad/s 로 바뀐다(로그 `합류 완료`).

`conda activate leap-hand` 뒤에 **같은 환경의 `colcon`** 으로 빌드해야 한다. ament_python 은
colcon 을 돌리는 파이썬을 console_scripts 의 shebang 에 박는다 — `/usr/bin/colcon` 으로 빌드하면
`/usr/bin/python3` 이 박혀 mediapipe/mujoco/우리 코어가 없다. `empy==3.3.4 lark catkin_pkg` 는
conda 파이썬이 rosidl 코드 생성에 쓰이기 때문에 필요하다(`leap_hand.srv`).

## 측정 (`scripts/phase1/p1_4_teleop_metrics.py`, `p1_5_step_response.py`)

fake 실기 + MuJoCo, 카메라 없이(`tracker:=false`), 스텝 20° — 재현 가능:

| | 지연 | 상승 (10→90%) | 정상상태 오차 |
|---|---:|---:|---:|
| 시뮬 (`/sim/joint_states`) | 33 ms | ~17 ms | 평균 1.17° |
| fake 실기 (`/real/joint_states`) | 66~79 ms | 167 ms | 0° |
| **실기** 16관절 전부 (kP 600, 4 Mbps) | **67~115 ms, 평균 83** | **31~101 ms, 평균 68** | **0.05~2.25°, 평균 0.81** |

- 시뮬 지연 33 ms 는 60 Hz 물리 타이머 + ROS 전송. fake 실기 지연은 브리지 램프(8 rad/s,
  20° = 44 ms) + 30 Hz 폴링이다.
- 시뮬에서 **`if_rot`/`mf_rot` 만 정상상태 오차 5.9°** (상승 없음). 영점에서 검지/중지를 20°
  벌리면 옆 손가락과 충돌한다 — Phase 0 의 벌림 충돌이 트윈에서 그대로 재현된 것이다.
  `qpos` 를 직접 넣었으면 안 보였을 사실이다.
- 시뮬-실기(fake) 추종 오차 RMS 2.68°.
- 헤드리스 확인(옛 기본값)에서 촬영→시뮬 반영 76 ms, 리타겟 38 ms/frame(매 프레임 재시도 5회). 라이브
  수치는 아래 "라이브 지표".

geon 라이브 확인(2026-08-21): `sim.launch.py` 가 단일 스크립트와 **똑같이 움직인다.**

**떨림 (2026-08-23 라이브 확인).** 실기가 kP 600 에서 정지 상태로 떨었다. 원인이 하나가 아니라
네 개가 겹쳐 있었고 — IK 재시도(`restart_mm` 1 mm 는 도달 불가라 매 프레임 시드 5개가 번갈아 이김),
랜드마크 잡음(`smoothing` 0.4), 2점 목표의 PIP 널스페이스, 모터 게인 — **하나씩 바꿔서는 안 보였고
전부 넣어야 멎었다.** 그래서 아래 조합이 런치와 노드의 **기본값**이다:

| 인자 | 전 | 후 | 뜻 |
|---|---|---|---|
| `kP` | 600 | **400** | 모터 P 게인 (실기만) |
| `smoothing` | 0.4 | **0.2** | 리타겟 지수 평활 |
| `deadband` | 0.5° | **1.0°** | 출력 변화가 이보다 작으면 직전 명령 재송 |
| `restart_mm` | 1 | **50** | IK 재시도 임계. 엄지 잔차가 30 mm+ 라 50 이어야 재시도 0 (35 → 6 ms/frame) |
| `pip_target` | false | **true** | PIP 관절점을 목표에 추가 (손가락당 3점) |
| `tip_mode` | realtip | **axis** | 로봇 손끝점을 패드(축에서 20° 이탈) 대신 손가락 축 위 점으로 |

예전 동작으로 돌리려면 `kP:=600 smoothing:=0.4 deadband:=0.5 restart_mm:=1 pip_target:=false tip_mode:=realtip`.

**벌림(rot) 관절 범위 제한 (2026-08-23).** 라이브에서 `rf_rot`(약지 벌림, 모터 8)이 옆 손가락에 걸려
전류가 한계(468)에 붙은 채 동결됐다. 모델 한계 안이라도 실기에서는 손가락끼리 닿는다. `joint_map.clip_mujoco`
가 쓰는 표(`LIMITS_TELEOP_MJ_*`)를 좁혔다 — IK 해·시뮬 ctrl·실기 명령이 전부 같은 범위를 받는다:

| 관절 | 전 (교집합) | 후 | 뜻 |
|---|---|---|---|
| `if_rot` | ±60° | **−60° ~ +3°** | 중지 쪽(+)으로 3° 까지만, 바깥쪽은 유지 |
| `mf_rot` | ±60° | **±3°** | 좌우 3° |
| `rf_rot` | ±60° | **−3° ~ +60°** | 중지 쪽(−)으로 3° 까지만, 바깥쪽은 유지 |

부호는 MJCF 로 확인했다 (`if_rot` + 면 검지–중지 손끝 거리 45→30 mm, `rf_rot` − 면 약지–중지 45→32 mm).
`python -m leap_hand_mapping.joint_map` 의 마지막 열(텔레옵)이 이 표다. 교집합 표 자체는 그대로 둔다.
`python scripts/phase1/p1_4_teleop_metrics.py` 의 **정지 떨림** 블록(관절 명령 / 실기 위치 / 시뮬 위치 /
랜드마크 간격 표준편차)이 떨림의 출처를 가른다 — 명령이 조용한데 실기가 떨면 모터, 명령이 떨면 리타겟/센서.

## 라이브 지표 (2026-08-23, geon, 웹캠 + MuJoCo + 실기, 위 기본값, `p1_4 --seconds 20`)

15초 자유 동작 + 마지막 5초 손을 편 채 정지.

| 토픽 | Hz | 촬영→수신 지연 평균 / 95% / 최대 (ms) |
|---|---:|---:|
| `/hand/landmarks` | 30.0 | 21.5 / 23.3 / 26.1 |
| `/leap/joint_cmd` | 30.0 | **32.1** / 33.9 / 37.8 |
| `/sim/joint_states` | 60.0 | – |
| `/real/joint_states` | 28.9 | – |

- **종단 지연 촬영→관절명령 32 ms** = 추적 21.5 + 리타겟·전송 10.6 (옛 기본값 restart 1 mm 로는 리타겟만 35~38 ms).
- **시뮬–실기 추종 오차 RMS 1.95°** (578 쌍, 최대 관절 `rf_pip` 3.4°). fake 실기 2.68° 보다 낮다 —
  실기는 kP 400 위치 제어, 트윈은 MuJoCo 액추에이터라 둘 다 같은 명령을 비슷한 속도로 따른다.
- **정지 떨림** (마지막 2초 std 평균): 관절 명령 0.75°, 실기 0.96°, 시뮬 0.77°, 랜드마크 간격 0.85 mm.
  엄지 `th_cmc` 만 명령 3.0° / 실기 4.8° 로 튄다 — 엄지 손끝 IK 잔차가 30 mm+ 라 해가 덜 고정되는 것.
  나머지 관절은 1° 안팎. 떨림 수정 전에는 실기가 눈에 띄게 떨렸다(수치 없음 — 그때는 분해 스크립트가 없었다).
- 전류 동결 0회 (벌림 관절 제한 뒤).

Phase 1 **완료.** 남은 개선 후보는 엄지 IK(잔차·떨림)와 데모 영상.
무슨 일이 있었고 무엇을 왜 고쳤는지는 **`docs/history/phase1_retrospective.md`** 에 전부 모았다.

실기 계단 응답 (2026-08-21, 데드맨 → `if_mcp` 한 관절 → 16관절 전체, 관절당 20°, 표본 41~44):

```
관절     지연ms 상승ms 정상오차deg      관절     지연ms 상승ms 정상오차deg
if_mcp    100    100    2.07          rf_mcp     99    101    1.95
if_rot    100     34    1.89          rf_rot     80     66    0.21
if_pip     68     65    0.05          rf_pip     73     65    0.14
if_dip     80     67    0.30          rf_dip     80     66    0.30
mf_mcp    100    100    1.54          th_cmc    115     64    1.37
mf_rot     73     65    2.25          th_axl     68     64    0.39
mf_pip     80     66    0.05          th_mcp     72     66    0.14
mf_dip     67     66    0.21          th_ipl     80     31    0.05
평균       83     68    0.81
```

- 지연 83 ms 중 ~45 ms 는 브리지의 8 rad/s 램프(20° 를 그 속도로 가는 시간)고 나머지가 통신 + 모터다.
  MCP(밑마디) 셋이 100 ms 로 가장 느리고 정상오차도 ~2° 로 가장 크다 — 손가락 전체 관성을 드는 관절이라 그렇다.
- **시뮬과 다른 점 하나.** 시뮬에서는 `if_rot`/`mf_rot` 가 영점에서 20° 벌리면 옆 손가락과 충돌해 5.9° 에서
  막혔는데, 실기는 1.9~2.3° 로 정상 추종했다. menagerie 충돌 박스가 실물보다 보수적이거나 실물 영점이
  조금 다르다는 뜻이다. 트윈의 충돌 기하는 실물 기준으로 손볼 여지가 있다.
- 계단 중 전류 동결은 한 번도 뜨지 않았다(정지 상태 20° 스텝, 부하 없음).
- 실기 읽기는 4 Mbps 에서 약 3% 가 CRC 로 깨져 직전 값으로 대체된다
  (`scripts/phase0/p0_4_read_reliability.py`, 방식·주기와 무관 → 배선/전기 쪽).

다음: 웹캠 + 시뮬 + 실기 동시(`real.launch.py`) → `p1_4_teleop_metrics.py` 로 종단 지연과 시뮬-실기 추종 오차.

---
