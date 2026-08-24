# 실기 붙이기와 안전: 모터를 움직이기 전에 확인할 것

> 원문: `../real_hand_bringup.md`, `scripts/phase0/p0_2_preflight_real_hand.py`, `p0_3_sweep_joints.py`, `p0_4_read_reliability.py`,
> `../design/joint_mapping.md` "실기 대조", `../history/phase1_retrospective.md` 3장

## 요약

- 실기 LEAP Lite 를 **기어를 상하게 하지 않고** 처음 켜는 순서를 정리한다.
- "통신은 되는데 값이 이상하다"를 세 가지로 가른다: 권한/포트, 지연시간 설정, 배선(CRC).
- 관절 하나 → 전체 순으로 검증하고 **추종 오차를 숫자로** 남긴다.

## 배경 — Lite 가 무엇이 다른가

| | Full | **Lite (우리 것)** |
|---|---|---|
| 기어 | 금속 | **플라스틱** |
| 전류 한계 | 550 mA | **350 mA** (코드 고정, 인자로 노출하지 않음) |
| 스톨 토크 / 연속 정격 | | 0.52 / 0.10 N·m |

스톨(막힌 채 계속 미는 상태)에서 350 mA 를 넘기면 이빨이 깨진다. 그래서 이 저장소의 실기 경로는 **전류를 상시 읽고**,
넘으면 명령을 얼린다. 350 은 올리지 않는다 — 어느 경로에서도.

### 통신 구조

PC ─USB─ U2D2(FTDI) ─4 Mbps 직렬─ 모터 16개 데이지체인. 한 번의 "읽기"는 16개에 동시 질의(FastSyncRead) → 16개 응답.
FTDI 의 `latency_timer`(기본 16 ms)를 1 로 내려야 30 Hz 폴링이 지연 없이 돈다.

## 실습 — 순서를 지킨다 (건너뛰면 모터가 상한다)

```bash
# 0. 전원 5V 30A ON, U2D2 연결, Dynamixel Wizard 종료(포트 점유), 손 주변 비우기
# 1. 권한 (1회): sudo usermod -aG dialout $USER → 재로그인
# 2. 지연시간: echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer   (docs/real_hand_bringup.md 2단계)
ls /dev/serial/by-id/                                  # 항상 by-id 경로. ttyUSB 번호는 바뀐다

# 3. 사전 점검 — 모터를 움직이지 않는다
python scripts/phase0/p0_2_preflight_real_hand.py
# 4. 관절 하나만
python scripts/phase0/p0_3_sweep_joints.py --real --joints 0     # MuJoCo 0 = if_mcp = 모터 1
# 5. 전체
python scripts/phase0/p0_3_sweep_joints.py --real
# 6. 읽기 오류율 (런치/스크립트를 다 끈 상태에서)
python scripts/phase0/p0_4_read_reliability.py
```

### 예시 출력 — 전체 순차 구동 (`../design/joint_mapping.md`)

16관절 육안 대조 통과. 추종 오차 평균 **1.45°**, 최대 3.91°:

| 종류 | 모터 | 검지 | 중지 | 약지 |
|---|---|---:|---:|---:|
| rot (벌림) | 0/4/8 | 3.91 | 3.72 | 1.84 |
| mcp (굽힘) | 1/5/9 | 1.66 | 1.68 | 1.80 |
| pip | 2/6/10 | 0.75 | 0.81 | 0.93 |
| dip | 3/7/11 | 0.88 | 0.82 | 0.74 |

### 예시 출력 — 읽기 오류율 (`p0_4`, 2026-08-21)

```
방식   읽기    Hz   시도  실패  실패율
fast   pvc     30   300     9   3.0%
fast   pvc     15   300     8   2.7%
sync   pvc     30   300    10   3.3%
```

## 결과 읽는 법

- 추종 오차가 **관절 종류별로 뭉친다**(벌림 > 굽힘 > pip ≈ dip). 세 손가락은 기구가 같으니 매핑이 맞을 때만 이렇게 정렬된다 —
  `joint_mapping.md` 의 FK 와 독립적인 두 번째 검증.
- 벌림 모터만 첫 실행에서 전류 초과가 떴다. 편 손에서 옆 손가락과 닿기 때문(±7.7° 만 여유). pip/dip 를 1.5 rad 굽힌
  자세에서 훑으면 전 범위가 열린다. **가동범위를 깎지 않고 검증 자세를 바꾼 것.**
- `Incorrect status packet` 3% — fast/sync, 30/15 Hz 모두 같다 → 소프트웨어가 아니라 **배선/4 Mbps** 문제. 업스트림
  드라이버가 실패한 읽기를 직전 값으로 대체하므로 무해. 줄이려면 케이블·커넥터 재삽입 또는 2 Mbps.

## 안전 규칙 (이 저장소의 실기 경로에 전부 코드로 들어 있다)

| 규칙 | 어디에 |
|---|---|
| 전류 350 mA 고정 | `real_hand.py`, 런치 `curr_lim` (올리지 말 것) |
| 전류 >400(리더 단위) 3표본 연속이면 명령 동결, 얼릴 때 현재 자세를 명령해 힘 빼기 | `hand_bridge_node` (`debugging_casebook.md` 사례 6·10) |
| 토크 켤 때 현재 자세 유지 (편 손으로 스냅하지 않음) | 업스트림 패치 `hold_on_start` |
| 데드맨 켜면 실기 자세를 읽고 1 rad/s 로 합류 | `hand_bridge_node` engage 램프 |
| 관절 범위 클립 후에만 송신 | `clip_mujoco` |
| 과부하 LED 점멸 → 전원 껐다 켜기 | 사람 |
