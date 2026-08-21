# LEAP Hand v1 Lite — 텔레오퍼레이션 / 디지털 트윈 / MJX 강화학습

> **2026-08-21 — 코드를 보정 이전(커밋 `7cccfdd`)으로 되돌렸다.**
> 엄지 정렬 캘리브레이션("보정")을 넣은 뒤로 고칠수록 라이브에서 이상해져서, 보정이 없던
> 마지막 상태에서 다시 시작한다. 그 이후의 모든 작업(dex-retargeting 비교, 엄지 결합,
> 엄지 관절각 매핑, 진단 스크립트, 녹화기, 실측 표)은 `bak/2026-08-21_thumb_work/` 와
> git 브랜치 `bak/2026-08-21-thumb-work` 에 그대로 있다. 그 과정에서 확인된 사실
> (dex 는 PIP 를 안 쓴다 / 엄지는 검지가 작업공간 밖이라 안 닿는다 / 손목 회전은 범위 밖)은
> `bak/2026-08-21_thumb_work/README.md` 의 Phase 1 절에 있다.
> 되돌린 뒤 웹캠으로 직접 확인: **이 버전이 오늘 시도한 모든 버전보다 잘 된다.** 녹화 데이터
> 지표로 이긴 버전들이 라이브에서는 졌다. 이후 변경은 라이브 확인 없이는 기본값을 바꾸지 않는다.

자작 LEAP Hand v1 **Lite**(XL330-M288-T ×16)에 MuJoCo 단일 스택으로 소프트웨어를 얹는 프로젝트.
전체 계획은 인수인계 문서(`leaphand.txt`, 개인 정보가 섞여 있어 저장소에는 넣지 않음) 참조.

---

## 현재 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| 환경 | Ubuntu 22.04 / conda `leap-hand` / JAX GPU | 완료 |
| Phase 0 | **관절 매핑 테이블 확정 + 기하 검증 + 실기 대조** | 완료 |
| Phase 1 | **웹캠 텔레오퍼레이션 + 디지털 트윈** | 파이프라인 완성, 실사용 검증 대기 |
| Phase 2 | MJX 강화학습 (`rotate_z`) | 미착수 |

---

## Phase 0 결과 — 관절 매핑

인수인계 문서 5장이 "최대 함정"으로 지목한 관절 순서 불일치를 해소했다.

### 무엇이 어긋나 있었나

MuJoCo 모델(`mujoco_menagerie/leap_hand`)은 leap-hand 공식 repo가 아니라
`dexsuite/dex-urdf`에서 파생되었다. 그 결과 **손가락마다 앞 두 관절의 순서가 실기와 반대**다.

| | 첫 번째 | 두 번째 | 세 번째 | 네 번째 |
|---|---|---|---|---|
| MuJoCo | `if_mcp` (굽힘) | `if_rot` (벌림) | `if_pip` | `if_dip` |
| 실기 모터 ID | `1` (MCP Forward) | `0` (MCP Side) | `2` | `3` |

엄지(12~15)는 순서가 그대로 대응된다.

확정된 매핑 — `mujoco_to_motor`:

```
[1, 0, 2, 3,  5, 4, 6, 7,  9, 8, 10, 11,  12, 13, 14, 15]
```

이 치환은 `(0 1)(4 5)(8 9)` 짝바꿈뿐이라 **자기 자신이 역치환**이다.

각도 규약은 실기의 0점이 모터 180도이므로 오프셋이 붙는다
(`leap_hand_utils.LEAPsim_to_LEAPhand`와 동일):

```
실기 각도 = MuJoCo 각도 + π      (순서 치환을 적용한 뒤)
```

### 어떻게 검증했나

문서는 육안 대조를 제안했지만, 먼저 **순기구학으로 정량 검증**했다.
실기와 같은 규약을 쓰는 공식 URDF를 PyBullet(headless)에 올리고,
MuJoCo와 같은 자세를 준 뒤 손끝 기하를 비교한다.

두 모델은 계보가 달라 베이스 프레임 규약이 다를 수 있으므로,
절대 좌표 대신 **좌표계에 무관한 네 손끝 사이의 쌍거리**를 비교한다.
매핑이 맞으면 자세를 바꿔도 오차가 요동치지 않으므로, 판정 지표는 평균이 아니라 **표준편차**다.

```bash
python scripts/phase0/p0_1_verify_mapping_fk.py
```

무작위 자세 200개에 대한 결과:

| 매핑 후보 | std (mm) | 변동폭 (mm) |
|---|---:|---:|
| **채택안 (체인 기반)** | **0.06** | **0.20** |
| 엄지 13-14 뒤바꿈 | 25.55 | 142.78 |
| 엄지 역순 | 30.25 | 159.45 |
| 손가락만 치환, 엄지 1칸 시프트 | 30.30 | 179.40 |
| 중지-약지 블록 교환 | 54.89 | 306.46 |
| 치환 없음 (항등) | 66.90 | 354.05 |

채택안이 차점 후보보다 **453배** 우수하다.
관절을 하나씩 단독 구동한 대조에서는 **16개 전부 0.0 mm 차이**로 일치했다.

독립적인 교차 확인: `LEAP_Hand_API/python/main.py`의 주석이
*"from MCP Side, MCP Forward, PIP, DIP for each finger"*라고 명시하여 같은 결론을 준다.

### 검증 과정에서 걸린 함정 두 가지

기록해 둘 가치가 있다. 둘 다 "검증이 통과했는데 사실은 검증이 안 된" 경우다.

1. **말단 링크의 원점을 기준점으로 쓰면 안 된다.**
   그 링크의 부모 관절(`dip`/`ipl`, 모터 3/7/11/15)이 움직여도 원점은 제자리라
   말단 4개 관절이 지표에서 통째로 빠진다. 처음엔 이 상태로 "통과"가 떴다.
2. **두 모델의 "손끝"은 서로 다른 점이다.**
   URDF `realtip`은 fingertip 프레임 + `(0.02, -0.07, 0.015)`인 실제 접촉점이고,
   MJCF `if_tip`은 tip 메시 geom의 중심이다. 그대로 비교하면 지렛대 길이가 달라
   매핑과 무관한 20 mm 오차가 섞인다. 양쪽에 동일한 오프셋을 적용해 통일했다.

`th_axl`(모터 13)은 엄지 축방향 회전이라 회전축이 손끝을 거의 지나가서
단독 구동 시 손끝이 0.1 mm밖에 안 움직인다. 이 지표로는 약하게만 검증되므로,
인접 관절과 뒤바뀐 후보(`엄지 13-14 뒤바꿈`)를 따로 두어 배제했다.

### 남은 한 칸 — 실기 대조

위 검증은 어디까지나 **모델끼리의 비교**다. 실물 배선과 모터 ID가 도면대로인지는
실기를 돌려 봐야 한다. 문서 5장의 절차를 그대로 자동화해 두었다.

> 작업대에서 따라갈 순서는 **[docs/real_hand_bringup.md](docs/real_hand_bringup.md)** 에 정리했다.
> 전원·권한·지연시간 설정부터 관절 하나씩 확인하는 절차까지 포함한다.

```bash
python scripts/phase0/p0_2_preflight_real_hand.py          # 먼저 이것부터. 모터를 움직이지 않는다
python scripts/phase0/p0_3_sweep_joints.py                 # 시뮬만 (실기 없이 확인)
python scripts/phase0/p0_3_sweep_joints.py --real --joints 0   # 관절 하나부터
python scripts/phase0/p0_3_sweep_joints.py --real          # 전체
```

관절을 하나씩 왕복시키며 MuJoCo 뷰어와 실기를 나란히 보여 주고,
실기 연결 시 관절별 추종 오차(deg)를 집계한다.

#### 실기 대조 결과

16개 관절 전부 육안 대조 통과. 검지 굽힘/벌림이 매핑대로 움직이는 것을 확인했다.
평균 추종 오차 **1.45°**, 최대 3.91°.

| 종류 | 모터 | 검지 | 중지 | 약지 |
|---|---|---:|---:|---:|
| rot (벌림) | 0/4/8 | 3.91 | 3.72 | 1.84 |
| mcp (굽힘) | 1/5/9 | 1.66 | 1.68 | 1.80 |
| pip | 2/6/10 | 0.75 | 0.81 | 0.93 |
| dip | 3/7/11 | 0.88 | 0.82 | 0.74 |

세 손가락은 기구학적으로 동일한데 **오차가 관절 종류별로 뭉친다**.
매핑이 어긋나 있으면 이렇게 정렬될 수 없으므로, FK 검증과 독립적인 확인이 된다.

#### 벌림 축 충돌 — 디지털 트윈으로 해결

첫 실행에서 **벌림 모터 0/4/8 만** 전류 임계(300 mA)를 넘겼다. 실기에서 손가락끼리
부딪히는 것이 원인이었고, MuJoCo 충돌 검출로 재 보니 그대로 재현됐다.

손을 편 자세에서 충돌 없이 벌릴 수 있는 각도는 **±0.14 rad(7.7°)뿐**인데
스크립트는 34.4°를 명령하고 있었다. 훑는 손가락 자신의 pip/dip 를 굽히면 범위가 열린다.

| pip/dip 굽힘 | 0.0 | 0.4 | 0.8 | 1.0 | 1.2 | **1.4** |
|---|---:|---:|---:|---:|---:|---:|
| 충돌 없는 최대 벌림 | 7.7° | 11.2° | 15.3° | 23.4° | 35.6° | **60.0° (전 범위)** |

1.4 rad 에서 전 범위가 열리므로 여유를 둬 **1.5** 를 쓴다.
가동범위를 깎지 않고 검증 자세만 바꾼 것이므로 손의 성능은 그대로다.
수정 후 16개 관절 전 구간에서 시뮬 충돌 0을 확인했다.

기준 자세로 넘어갈 때는 급격한 점프 대신 1초에 걸쳐 램프시킨다.

> **실기 주의 (문서 4.5)**
> - Dynamixel Wizard가 떠 있으면 포트를 점유해 연결되지 않는다. 먼저 종료할 것.
> - 과부하로 빨간 LED가 점멸하면 **전원을 껐다 켜야** 복구된다.
> - 진동하거나 과부하가 잦으면 `--kp 400`으로 낮출 것 (문서 4.2).

---

## Phase 1 — 웹캠 텔레오퍼레이션

```
웹캠 → MediaPipe 21 랜드마크 → 손바닥 좌표계 정규화 + LEAP 치수 스케일링
     → PyBullet IK (손끝 8점 → 16 관절각) → MuJoCo 디지털 트윈 → (선택) 실기
```

> 작업대에서 따라갈 순서는 **[docs/teleop_howto.md](docs/teleop_howto.md)** 에 정리했다.
> 촬영 환경, 단계별 확인 항목, 출력 숫자 읽는 법, 튜닝, 문제 해결까지 포함한다.

```bash
bash scripts/phase1/p1_0_fetch_mediapipe_model.sh          # 최초 1회 (7.5MB, 저장소에 없음)
python scripts/phase1/p1_1_check_hand_tracking.py          # 추적만 먼저 확인
python scripts/phase1/p1_3_teleop_mujoco.py                # 시뮬만
python scripts/phase1/p1_3_teleop_mujoco.py --pybullet-gui # IK 목표점까지 보면서
python scripts/phase1/p1_3_teleop_mujoco.py --real         # 실기까지
```

### 기하 — 사람 손과 로봇 손을 같은 방식으로 재기

두 손에 **동일한 규칙으로** 손바닥 좌표계를 세운다. 원점은 검지 MCP와 약지 MCP의
중점, y축은 손가락이 뻗는 방향, x축은 검지→약지, z축은 그 외적이다. 사람 쪽은
MediaPipe world 랜드마크에서, 로봇 쪽은 URDF 영점 자세에서 뽑는다.

이렇게 하면 카메라 앞에서 손을 어떻게 들고 있든 회전·평행이동이 상쇄된다.
크기는 손바닥 폭 비율로 매 프레임 맞춘다. LEAP 쪽은 90.9mm로 고정이고, 사람 쪽은
MediaPipe가 추정한 값을 쓴다 — `check_hand_tracking.py`가 실측값과 그로부터 나온
배율을 출력한다. 사람 손이 훨씬 작아 배율은 2배 안팎이 된다.

관절각을 그대로 베끼지 않고 손끝을 목표로 IK를 푸는 이유는 두 손의 링크 길이와
축 배치가 다르기 때문이다. 조작에서 중요한 건 관절각이 아니라 손끝 위치다.

### 말단 마디 길이 보정

사람 손을 스케일해도 DIP→TIP 길이는 LEAP과 맞지 않는다. LEAP의 `realtip`은
fingertip 프레임에서 74mm 떨어진 접촉점이라(Phase 0에서 확인한 그 오프셋),
스케일된 사람 손끝보다 훨씬 멀다. 두 목표를 동시에 만족할 수 없어 IK가
어중간하게 수렴한다.

그래서 TIP 목표는 **방향만 사람에게서 받고 길이는 LEAP 자신의 값**을 쓴다.
두 목표가 항상 도달 가능한 조합이 되므로 IK가 깨끗하게 풀린다.

### PyBullet 내장 IK를 버린 이유

공식 `avp_leap.py`는 `calculateInverseKinematics2`를 쓴다. 그대로 따라가다 막혔다.

| 방식 | 손끝 잔차 |
|---|---:|
| 내장 IK + 널스페이스 인자 (범위/restPose) | 150 mm |
| 내장 IK, 인자 없이 1회 | 10 mm |
| 내장 IK, 30회 반복 | 1.25 mm |
| **자코비안 DLS 직접 구현** | **0.09 mm** |

널스페이스 인자를 주면 해가 시드와 무관하게 고정돼 버린다. 빼면 동작은 하지만
수렴이 선형이라 반복해도 mm 대에 머문다. `IK_SDLS`는 이 모델에서 세그폴트한다.

직접 푸는 쪽이 빠르고(연속 추적 시 1.2ms), 매 반복 관절 범위를 강제할 수 있어
결과도 안전하다. 목표 24차원 / 자유도 16의 과결정계라 16×16 정규방정식으로 푼다.
자코비안은 유한차분과 대조해 확인했다(상대오차 0.00%).

### 국소최소 — 사람 손 각도를 시드로

DLS를 영점에서 출발시키면 크게 굽힌 손에서 갇힌다. 중지·약지가 관절 한계에 붙은
채 **손끝 잔차 158mm**로 수렴하는 자세가 나왔다. 같은 자세에 정답을 시드로 주면
0.000mm로 풀리므로, 해가 없는 게 아니라 출발점 문제다.

사람 손 각도는 정확한 답은 아니지만 **올바른 골짜기**에 있다. 랜드마크에서 각
마디의 굽힘·벌림 각을 직접 재서 시드로 쓴다(부호 규약은 URDF에서 실측 —
굽힘 +는 손바닥 쪽, 벌림 +는 약지 쪽).

엄지는 이게 안 통한다. LEAP 엄지는 축 배치가 사람과 달라(`th_axl`은 회전축이
손끝을 거의 지난다) 각도를 옮겨 봐야 엉뚱한 골짜기다. 실제로 남은 실패의 2/3가
엄지였다. 굽힘량과 축회전을 달리한 자세 4개를 재시도 시드로 추가했다.

재시도는 **실패한 손가락만** 다시 푼다. 손가락끼리 자유도가 겹치지 않아
자코비안이 블록대각이므로, 전체를 다시 풀고 해당 블록만 가져와도 손가락 단위로
따로 푸는 것과 수학적으로 같다.

### 검증 — 로봇을 사람이라고 치고 왕복

"손 흔들어 보니 따라오더라"는 검증이 아니다. 정답이 있는 시험을 만들었다.

```bash
python scripts/phase1/p1_2_test_retarget_roundtrip.py
```

LEAP을 알려진 자세에 두고, 그 자세에서 MediaPipe 배치의 가짜 사람 랜드마크를
뽑아 사람 손 크기로 줄이고 **임의 회전·평행이동**까지 준 뒤 리타겟터에 넣는다.
좌표계 구성이 옳다면 준 변환이 전부 상쇄되어 원래 자세가 돌아와야 한다.

무작위 자세 60개:

| 지표 | 평균 | 최대 |
|---|---:|---:|
| 손끝 잔차 | **0.09 mm** | 0.98 mm |
| 관절각 오차 | 1.42° | 62.14° |

관절각 최대 오차가 큰 것은 IK의 중복성 때문이다 — pip/dip 조합이 달라도 같은
손끝 위치가 나온다. 손끝이 맞으면 목적은 달성된 것이므로 판정은 잔차로 한다.

이 시험은 **좌표 변환과 IK가 서로 맞물리는지**를 보는 것이지, 사람 손과 LEAP의
비율 차이에서 오는 리타겟팅 품질을 보증하지는 않는다. 그쪽은 실제 웹캠으로만
확인할 수 있다.

### 성능

| 항목 | 값 |
|---|---:|
| 전체 루프 | 27 fps (카메라 상한) |
| MediaPipe 추론 | 12 ms |
| 리타겟 + IK (연속 추적) | 1.2 ms |
| 리타겟 + IK (재시도 전부 도는 최악) | 14.5 ms |

### 안전 설계

- 손을 놓치면 **직전 자세를 유지**한다. 프레임 밖으로 나갈 때마다 손이 펴지면
  물건을 놓치기 때문이다. `--release-after` 초를 넘기면 그때 천천히 영점으로.
- 지수 평활(랜드마크가 프레임마다 수 mm 떨림) + 관절 속도 상한 8 rad/s.
- 출력은 Phase 0의 교집합 범위로 클립한 뒤에만 실기로 나간다.
- 실기 전류가 임계를 넘으면 **명령을 그 자리에 얼린다**. 스톨 상태에서 계속 밀면
  Lite 기어가 상한다.
- 디지털 트윈은 `qpos`를 직접 넣지 않고 액추에이터에 명령을 주고 물리를 돌린다.
  충돌과 추종 지연이 보여야 트윈 구실을 한다(Phase 0의 벌림 충돌이 이 방식으로
  재현됐다).

### 거울상 함정

MediaPipe의 handedness 판정은 입력이 거울상이 아니라고 가정한다. 셀피처럼 좌우를
뒤집어 넣으면 라벨이 반대로 나오고 **world 랜드마크의 손대칭도 같이 뒤집힌다**.
그대로 리타겟팅하면 로봇 손이 손등 쪽으로 굽는 거울상이 된다.

그래서 추적기는 handedness 라벨로 손을 고른다. 오른손을 들었는데 `Right`로 안
잡히면 영상이 뒤집힌 것이므로 `--mirror`를 켜면 라벨과 좌표가 동시에 제자리로
돌아온다. **라벨 필터가 곧 좌표계 정합 검사**다.

### 남은 것

- 실사용 검증: 실제 손으로 돌려 보고 스케일·평활 계수를 조정.
- 엄지 리타겟 품질. 기하가 가장 많이 달라 예상 오차가 가장 크다.
- 추종 오차 측정 방식. 현재는 명령 직후 바로 읽어 통신 지연이 섞인다.
  Day 5의 정량 지표로 쓰려면 측정 시점을 제대로 잡아야 한다.

---

## 구조

```
leap_hand_mapping/
  joint_map.py      매핑 테이블 + 좌표 변환 (순수 numpy, OS 무관)
  real_hand.py      실기 드라이버. 전류 제한 350mA 고정, 전류 상시 감시
  hand_tracker.py   웹캠 -> MediaPipe 21 랜드마크 (로봇을 모른다)
  retarget.py       랜드마크 -> 16 관절각. 손바닥 좌표계 + 자코비안 DLS IK
scripts/
  phase0/
    p0_1_verify_mapping_fk.py      PyBullet(URDF) vs MuJoCo 순기구학 교차검증
    p0_2_preflight_real_hand.py    실기 사전 점검 (토크를 켜지 않는다)
    p0_3_sweep_joints.py           관절 순차 구동 (시뮬 / 실기)
  phase1/
    p1_0_fetch_mediapipe_model.sh  MediaPipe 모델 받기
    p1_1_check_hand_tracking.py    웹캠 추적만 확인 (로봇 없이)
    p1_2_test_retarget_roundtrip.py 리타겟팅 왕복 검증 (카메라 없이)
    p1_3_teleop_mujoco.py          텔레오퍼레이션 본체
bak/
  2026-08-21_thumb_work/           되돌리기 전 작업 보관본 (아래 참고)
```

바깥 코드는 **MuJoCo 관절 순서만** 알면 된다. 모터 ID 순서와 π 오프셋 변환은
`LeapHandDriver`가 내부에서 처리한다.

```python
from leap_hand_mapping import joint_map as jm
from leap_hand_mapping.real_hand import LeapHandDriver, find_port

with LeapHandDriver(port=find_port()) as hand:
    hand.command_mujoco(q)        # MuJoCo qpos 를 그대로 명령
    q_now = hand.read_mujoco()    # 실기 상태를 MuJoCo 규약으로 회수
    over = hand.check_current()   # 전류 임계 초과 모터 목록
```

`mujoco_playground`의 LEAP 모델도 menagerie 계열이라
(`leap_hand_constants.JOINT_NAMES`가 동일함을 확인) 이 매핑은 **Phase 2에 그대로 재사용된다.**

---

## 환경 구성

```bash
conda create -n leap-hand python=3.10 -y     # ROS2 Humble rclpy 와 같은 버전
conda activate leap-hand
pip install "jax[cuda12]" mujoco mujoco-mjx playground pybullet \
            mediapipe opencv-python numpy dynamixel-sdk

bash scripts/phase1/p1_0_fetch_mediapipe_model.sh   # MediaPipe 1.x 는 모델을 패키지에 넣지 않는다
```

GPU 확인 (문서 6장 — CPU 빌드가 깔리면 에러 없이 조용히 느려지기만 한다):

```bash
python -c "import jax; print(jax.devices())"   # [CudaDevice(id=0)] 이어야 함
```

### 참조 저장소

```bash
mkdir -p third_party && cd third_party
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
git clone --depth 1 https://github.com/leap-hand/Bidex_VisionPro_Teleop.git
git clone --depth 1 https://github.com/leap-hand/LEAP_Hand_API.git
git clone --depth 1 https://github.com/google-deepmind/mujoco_playground.git
```

라이선스: 코드 MIT / CAD는 CC BY-NC-SA(비상업).
