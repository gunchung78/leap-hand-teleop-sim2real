# Phase 1 회고 — 웹캠 텔레오퍼레이션 + 디지털 트윈 (2026-08-19 ~ 08-23)

Phase 1 동안 **무슨 일이 있었고, 무엇을 왜 고쳤고, 지금 어떤 상태인지**를 한 곳에 모은 기록이다.
사용 절차는 `teleop_howto.md`, 계획과 노드 사양은 `phase1_plan.md`, 수치 표는 README 에 있다.
이 문서는 "왜 지금 모양이 이런가"에 답한다. Phase 1 을 마치고 썼다.

---

## 0. 한 장 요약

| | 시작 (08-19) | 끝 (08-23) |
|---|---|---|
| 형태 | 단일 파이썬 스크립트 (`p1_3_teleop_mujoco.py`) | **ROS2 노드 5개** + 업스트림 `leaphand_node`, 런치 2개 |
| 대상 | MuJoCo 시뮬만 | **시뮬 + 실기 동시**, 같은 명령 토픽 |
| 리타겟터 | 직접 구현 손끝 IK (7cccfdd) | **같은 것** (+ 옵션 4개, 기본값 변경) |
| 안전 | 전류 한계 350 mA | + 데드맨(SPACE), 시작 자세 유지, 합류 램프, 전류 동결(힘 빼기), 벌림 관절 제한 |
| 라이브 수치 | 없음 | 촬영→명령 **32 ms**, 시뮬–실기 RMS **1.95°**, 정지 떨림 **<1°** |
| 상태 | — | **완료** (geon 라이브 "아주 잘된다") |

커밋 범위: `683e1d2`(Phase 1 시작) → `32985f3`(완료). 약 50 커밋.

---

## 1. 시간순으로 무슨 일이 있었나

### 08-19 — 파이프라인과 리타겟터 (단일 스크립트)
- `683e1d2` 웹캠 → MediaPipe 21점 → 리타겟 → MuJoCo. 처음엔 PyBullet 내장 IK 를 썼다가 버리고
  자코비안 DLS IK 를 직접 썼다(README "PyBullet 내장 IK를 버린 이유").
- `7cccfdd` **손가락별 길이 스케일링** (손바닥 폭 균일 스케일 폐기). → 이 커밋이 나중에 "가장 잘 되는 버전"이 된다.
- `08c010e` 손끝 앵커 + 엄지 정렬 캘리브레이션 + 앞마디 보조 목표. `f75a2c6` dex-retargeting 으로 교체.
  `eee227f` 다시 직접 IK 로. `0716f82` 엄지 roll 2자세 캘리브레이션. `f118b58` IK 재시도 기준 적응형.
  — 하루에 리타겟터를 네 번 갈아 끼웠다.

### 08-20 ~ 08-21 오전 — 리타겟터 비교와 엄지 (나중에 전부 보관함으로)
- `095b908` dex 로, `40d3b53` 다시 ours 로. 실측으로 확정: **ours 가 자세(주먹 MCP/PIP/DIP), dex 가 핀치**
  에서 이긴다. dex 는 손목→손끝 벡터만 맞추므로 PIP 를 안 쓰고 갈고리가 된다.
- `09d921a` `799c39b` 엄지 원인 규명·검지에 결합(A2). `e8b4042` `367ae1f` **사람 녹화 데이터가 오염**돼 있어
  (타이머 녹화 → 핀치의 2/3 가 손 펴는 중) 녹화기를 키 입력식으로 다시 만들고 모든 표를 다시 쟀다.
- `b352685` 엄지 관절각 매핑, `04fa5e4` `4e31ad7` 손 위치 틀(HandGuide)과 원인 분리 리포트.
- 시뮬 엄지 한계표가 출처 URDF 와 달라(th_mcp 하한 −27° vs −69°) 고쳤다. "LEAP 엄지는 기구적으로 안 접힌다"는
  내 주장은 틀렸음 — 실물은 접힌다(geon).

### 08-21 낮 — 되돌리기 결정
- geon: **"보정할수록 뭔가 이상해지네. 완전 처음으로 돌아가고 싶어."** → `141cd93` 리타겟터와 스크립트를
  **7cccfdd 로 되돌림.** 그 뒤의 작업(dex 어댑터, 엄지 A2/매핑, 진단 스크립트, 녹화기, 한계 통일)은
  `bak/2026-08-21_thumb_work/` 와 브랜치 `bak/2026-08-21-thumb-work` 에 보관.
- 라이브 확인: **"이게 가장 잘된다."** 녹화 데이터 지표로 이긴 버전들이 라이브에서는 전부 졌다.
  원인: 캘리브레이션이 배율·엄지 회전을 시작 시점 거리에 고정하는데 MediaPipe metric 배율은 거리에 따라
  변해서 고칠수록 조건 의존성이 늘었다. → **교훈 1** (아래 5장).

### 08-21 오후 — ROS2 통합 (S0~S7, 하루)
- `5e2cb53` S0/S1 코어 패키징(`pyproject.toml`), `ros2_ws` + 업스트림 `leap_hand` 복사 스크립트 + 포트 패치.
- `8a996a8` S3 `tracker_node`, `retarget_node`. `28e5013` S4 `sim_node` + `sim.launch.py`.
- `7fd2d16` S5 `hand_bridge_node`(안전 래퍼) + `fake_hand_node` + `real.launch.py`.
- `2122027` S6 지표 스크립트 `p1_4`(Hz/지연/추종), `p1_5`(계단 응답). `d0f2ea7` S7 문서.
- `65c5722` geon 라이브: **ROS2 `sim.launch` 가 단일 스크립트와 똑같이 움직인다.**

### 08-21 밤 — 실기 연결
- 첫 실행 로그: `Incorrect status packet` 반복 + 시작 직후 전류 초과. `5a128a7` poll_rate 인자,
  `8f2dfab` `p0_4_read_reliability.py` 로 오류율 측정 → fast/sync, 30/15 Hz 모두 ~3%: **배선/전기**, 무해(직전 값 대체).
- 데드맨 → `if_mcp` 한 관절(`3c951c3`) → 16관절(`5c458a4`) 계단 응답. 평균 지연 83 ms, 상승 68 ms, 정상오차 0.81°.
  시뮬에서 보이던 `if_rot`/`mf_rot` 충돌이 실기에선 없었다.
- `b93c464` **시작 시 확 움직임 제거 둘** + **SPACE 데드맨 버튼**. `8983e75` 전류 동결 디바운스.
- `1ff3040` `080a22e` `97bb7eb` `0bf3d1d` 떨림 대책 인자(deadband, restart_mm)와 런치 타입 버그 둘.
- `104668b` `bd011a1` 손끝 불일치 진단 → `pip_target`, `tip_mode` 옵션. `0abd5f9` 정지 떨림 분해 지표.

### 08-23 — 떨림 해결, 동결 교착, 벌림 제한, 완료
- `27c8833` 떨림: **하나씩은 안 됐고 전부 넣어야 멎었다** → 조합을 기본값으로.
- `09fe1c7` 전류 동결이 안 풀리는 교착 수정. `a33121e` 벌림 관절 텔레옵 제한.
- `32985f3` 라이브 지표 → README, **Phase 1 완료.**

---

## 2. 최종 구조

```
tracker_node ──/hand/landmarks──▶ retarget_node ──/leap/joint_cmd──┬──▶ sim_node ──▶ /sim/joint_states
 (웹캠, MediaPipe,                (7cccfdd 손끝 IK,                 │    (MuJoCo, ctrl, 60 Hz)
  SPACE 데드맨)                     평활·데드밴드·유실 램프)          │
                                                                    └──▶ hand_bridge_node ──/cmd_leap──▶ leaphand_node (업스트림+패치)
                                                                          (데드맨·동기·합류·클립·전류동결)      └──/leap_pos_vel_eff──▶ /real/joint_states
```

- 명령 토픽 하나가 시뮬과 실기를 같이 먹인다 — 이게 "디지털 트윈"의 실체다.
- `header.stamp` 는 카메라 촬영 시각을 끝까지 전파 → 어느 노드에서나 `now − stamp` 가 종단 지연.
- 실기로 나가는 길은 `hand_bridge_node` 하나뿐. 업스트림 `leaphand_node` 는 포트/보레이트 파라미터와
  `hold_on_start` 패치만 얹었다(`patches/leap_hand_port_param.patch`).
- 빌드는 conda 의 colcon (`/usr/bin/colcon` 이면 노드가 conda 패키지를 못 찾는다). 실행 전 `conda activate leap-hand`.

---

## 3. 문제 → 원인 → 고친 것 (전부)

| # | 증상 | 원인 | 고친 것 | 커밋 |
|---|---|---|---|---|
| 1 | 노드가 `leap_hand_mapping`/`mediapipe` 를 못 찾음 | `/usr/bin/colcon` 이 `#!/usr/bin/python3` 셔뱅을 박음 | conda 에 colcon 설치 후 재빌드. README 에 명시 | 5e2cb53 |
| 2 | `rclpy._rclpy_pybind11 ... cpython-313` | conda base(py3.13)가 PATH 앞 | `conda activate leap-hand` 필수 | 문서 |
| 3 | `Incorrect status packet` 초당 1~2회 | 4 Mbps CRC 깨짐, 읽기 방식/주기/latency_timer/return delay 무관 | 측정 스크립트 `p0_4`. 무해(직전 값 대체)로 판정, 그대로 둠 | 8f2dfab |
| 4 | 런치 직후 실기가 **확** 움직임 (1) | 업스트림 노드가 토크 켜며 편 손(π)으로 스냅 | 패치 `hold_on_start`: 현재 자세를 읽어 그 자세로 토크 ON | b93c464 |
| 5 | 데드맨 켜는 순간 **확** 움직임 (2) | 카메라가 이미 다른 자세를 보고 있어 8 rad/s 로 점프 | 브리지 합류 램프: 실기 자세 읽고 1 rad/s 로 목표에 합류 후 8 rad/s | b93c464 |
| 6 | 버튼으로 켜고 끄고 싶다 | — | 카메라 창 SPACE = `/teleop/enable` 토글, 창에 ROBOT ON/OFF | b93c464 |
| 7 | 전류 초과 경고가 계속 뜸 (th_cmc/if_pip 300~380) | 움직이는 순간 과도 전류를 한 표본으로 얼림 → 30 ms 마다 동결/해제 | 임계 300→400, 3표본(100 ms) 연속일 때만 | 8983e75 |
| 8 | **실기가 떨린다** | 네 원인이 겹침: IK 재시도(1 mm 는 도달 불가 → 시드 5개 번갈아 이김), 랜드마크 잡음, 2점 목표의 PIP 널스페이스, kP 600 | 인자 노출(deadband, restart_mm, pip_target, tip_mode) → 하나씩은 실패 → **조합을 기본값으로** (kP 400 / smoothing 0.2 / deadband 1° / restart 50 mm / pip_target / axis) | 1ff3040 080a22e 104668b bd011a1 27c8833 |
| 9 | `restart_mm:=15` 주니 시뮬이 안 움직임 | 정수 문자열 → INTEGER 파라미터 → 노드 사망 | Node 파라미터에 `ParameterValue(..., value_type=float)` | 97bb7eb |
| 10 | `'ParameterValue' object is not iterable` | 9 번을 include 의 launch_arguments 에도 넣음 | include 인자는 문자열 치환만, 타입은 Node 쪽에서 | 0bf3d1d |
| 11 | 손가락 끝점과 로봇 끝점이 다름 | `realtip` 이 손가락 축에서 20° 벗어난 패드점(74 mm 오프셋) + MediaPipe 검지 말단 편향 + 2점 목표 널스페이스 | `tip_mode axis`(축 위 점), `pip_target`(PIP 점 추가) | 104668b bd011a1 |
| 12 | 전류 동결이 한번 걸리면 **안 풀림** (rf_rot 468→445→425) | 동결이 명령만 멈춰 막히던 목표가 모터에 남음 → 계속 밀어 한계 전류 유지 | 얼릴 때 실기 현재 자세를 한 번 명령해 힘 빼기, 풀리면 1 rad/s 재합류 | 09fe1c7 |
| 13 | `rf_rot` 이 옆 손가락에 걸려 전류 한계 | 모델 한계(±60°) 안이라도 실기는 서로 닿음 | `clip_mujoco` 표 좁힘: mf_rot ±3°, if/rf_rot 중지 쪽 3° (바깥쪽 유지). 부호는 MJCF 손끝 거리로 확인 | a33121e |
| 14 | 카메라 "can't open camera by index" | 내 헤드리스 시험 런치가 죽지 않고 `/dev/video0` 을 쥠 | `pkill -9 -f "[l]ib/leap_teleop/"`, `fuser /dev/video0` 로 확인하는 습관 | 운영 |
| 15 | 종료 시 `RCLError`/`GLXBadContext` 트레이스 | launch 의 SIGINT 와 컨텍스트 종료 경합 | `if not rclpy.ok(): return` 가드. GLX 는 MuJoCo 뷰어 종료 잡음, 무시 | 2122027 |

---

## 4. 지금 기본값과 그 근거

| 항목 | 값 | 근거 |
|---|---|---|
| 리타겟터 | 7cccfdd 손끝 IK | 라이브에서 모든 변형보다 나았다 (08-21) |
| `smoothing` | 0.2 | 떨림 조합 (08-23) |
| `deadband` | 1.0° | 〃 |
| `restart_mm` | 50 | 〃. 엄지 잔차 30 mm+ 라 15 로는 재시도가 남았다. 리타겟 35→10 ms |
| `pip_target` | true | 〃. 녹화 실측 정지 떨림 1.11→0.33° |
| `tip_mode` | axis | 〃 + 손끝 불일치 |
| `kP / kD` | 400 / 200 | 〃 (600 은 떨림) |
| `curr_lim` | 350 mA | Lite 플라스틱 기어. **올리지 말 것** |
| 전류 동결 | >400 × 3표본, 해제 <300, 힘 빼기 | #7, #12 |
| 합류 속도 / 최대 속도 | 1 / 8 rad/s | #5 |
| 벌림 관절 | mf ±3°, if/rf 중지 쪽 3° | #13 |
| `poll_rate` | 30 Hz | 읽기 오류율이 주기와 무관해 낮출 이유 없음 |
| `hold_on_start` | true | #4 |

---

## 5. 배운 것 (다음 Phase 에서 지킬 규칙)

1. **오프라인 지표는 라이브 판정을 대신하지 못한다.** 녹화 npz 로 이긴 버전이 웹캠 앞에서 졌다.
   변경은 geon 이 직접 보고 "좋다/나쁘다"를 말한 뒤 기본값이 된다. 그 전까지는 인자(옵션)로만 둔다.
2. **한 번에 하나씩 — 단, 원인이 겹치면 예외.** 떨림은 하나씩 바꿔서는 안 보였고 조합으로 멎었다.
   순서는 "조합으로 멎게 한 뒤 하나씩 빼서 기여도 가르기"가 맞다 (아직 안 했음, 남은 일).
3. **사람 데이터부터 검증.** 타이머 녹화 하나가 모든 표를 오염시켰다. 시계열을 먼저 본다.
4. **시뮬 한계표를 믿고 "기구적 불가"라 하지 말 것.** 실물을 가진 사람에게 묻고 출처 URDF 와 대조한다.
5. **업스트림은 복사+패치로.** `leap_hand` 는 vendoring 하지 않고 스크립트로 가져와 한 패치만 얹는다. 재현 가능.
6. **측정 스크립트를 같이 커밋.** 숫자는 `p0_4`, `p1_4`, `p1_5` 로 누구나 다시 잴 수 있다.
7. **상태를 바꾸는 행동(동결·데드맨)은 풀리는 길도 같이 설계.** #12 는 "얼리기"만 설계하고 "풀리기"를 안 본 결과.
8. **내가 띄운 시험 프로세스는 내가 죽인다.** 카메라를 쥔 좀비가 사용자 화면에 에러로 나타났다.

---

## 6. 남은 것 / 알려진 한계

- **엄지 `th_cmc` 떨림 3°(명령)/4.8°(실기)** — 엄지 손끝 IK 잔차가 30 mm+ 라 해가 덜 고정. 나머지 관절은 1° 안팎.
- 떨림 조합의 **기여도 분해** 안 했음 (하나씩 빼 보기).
- `Incorrect status packet` ~3% — 배선/4 Mbps. 케이블 재삽입 또는 2 Mbps 로 내리면 줄 수 있다.
- 손목 회전 자세는 범위 밖(단안 깊이 한계). 핀치는 ours 가 dex 보다 넓다(54 vs 38 mm).
- 데모 영상 없음.
- `bak/2026-08-21_thumb_work/` 의 진단 스크립트는 현재 `retarget.py` API 와 안 맞는다 — 되살릴 땐 브랜치에서 같이.

---

## 7. 파일 지도 (Phase 1 에서 생기거나 바뀐 것)

```
pyproject.toml                               leap_hand_mapping 패키징 (pip install -e .)
leap_hand_mapping/retarget.py                7cccfdd + pip_target / tip_mode 옵션
leap_hand_mapping/hand_tracker.py            HandGuide (손 위치 틀)
leap_hand_mapping/joint_map.py               LIMITS_TELEOP_MJ_* (벌림 제한), clip_mujoco
patches/leap_hand_port_param.patch           업스트림 leaphand_node: port/baudrate 파라미터, hold_on_start
ros2_ws/setup_upstream.sh                    업스트림 복사 + 패치
ros2_ws/src/leap_teleop/
  leap_teleop/{tracker,retarget,sim,hand_bridge,fake_hand}_node.py
  launch/sim.launch.py  launch/real.launch.py
scripts/phase0/p0_4_read_reliability.py      읽기 오류율
scripts/phase1/p1_4_teleop_metrics.py        Hz / 지연 / 추종 / 정지 떨림
scripts/phase1/p1_5_step_response.py         계단 응답
docs/phase1_plan.md  docs/teleop_howto.md  docs/phase1_retrospective.md (이 문서)
bak/2026-08-21_thumb_work/                   되돌리기 전 작업 보관 (+ 브랜치 bak/2026-08-21-thumb-work)
```
