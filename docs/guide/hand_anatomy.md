# 로봇 손 이해: 16개 모터, 두 가지 이름, 한 가지 규약

> 원문: `leap_hand_mapping/joint_map.py`, `../design/joint_mapping.md`, `../real_hand_bringup.md`

## 요약

- LEAP Hand v1 의 모터 16개가 **어디에 붙어 있고 무엇을 움직이는지** 손으로 가리킬 수 있게 표로 정리한다.
- 같은 관절을 부르는 두 이름 체계(MuJoCo 이름 / 실기 모터 ID)를 서로 바꿔 보인다.
- 이 저장소의 모든 벡터가 **어느 순서·어느 단위·어느 0점**인지 안다 — 이후 문서의 전제다.

## 배경

LEAP Hand 는 손가락 4개(검지·중지·약지·엄지, **새끼손가락 없음**) × 모터 4개 = 16 자유도다. 모터는 Dynamixel
XL330-M288, 데이지체인 한 줄로 U2D2 에 물린다. 우리 것은 **Lite** — 기어가 플라스틱이라 전류 한계 350 mA 를 넘기면
이빨이 깨진다(`real_hand_safety.md`).

### 손가락 하나의 구조 (손바닥 → 손끝)

```
손바닥 ── [벌림 rot] ── [굽힘 mcp] ── [중간 pip] ── [끝 dip] ── 손끝
          ↑ 한 덩어리(같은 하우징) ↑
```

앞의 두 모터는 한 덩어리에 들어 있어 헷갈리기 쉽다. **벌림(rot)** 은 손가락을 옆으로 벌리고, **굽힘(mcp)** 은
손가락 전체를 손바닥 쪽으로 접는다. 엄지는 뿌리 회전(cmc) → 축 회전(axl) → 굽힘(mcp) → 끝 굽힘(ipl).

### 두 이름 체계

| 손가락 | 위치 | MuJoCo 이름 | 모터 ID | 움직임 |
|---|---|---|---|---|
| 검지 | 뿌리 | `if_rot` | **0** | 벌림 (+ 가 중지 쪽) |
| | 뿌리 | `if_mcp` | **1** | 굽힘 |
| | 중간 | `if_pip` | 2 | |
| | 끝 | `if_dip` | 3 | |
| 중지 | 뿌리 | `mf_rot` | **4** | 벌림 (+ 약지 쪽, − 검지 쪽) |
| | 뿌리 | `mf_mcp` | **5** | 굽힘 |
| | 중간 | `mf_pip` | 6 | |
| | 끝 | `mf_dip` | 7 | |
| 약지 | 뿌리 | `rf_rot` | **8** | 벌림 (− 가 중지 쪽) |
| | 뿌리 | `rf_mcp` | **9** | 굽힘 |
| | 중간 | `rf_pip` | 10 | |
| | 끝 | `rf_dip` | 11 | |
| 엄지 | 뿌리 | `th_cmc` | 12 | 손바닥 가로지르는 회전 |
| | | `th_axl` | 13 | 축 회전 (마주보기) |
| | | `th_mcp` | 14 | 굽힘 |
| | 끝 | `th_ipl` | 15 | 끝 굽힘 |

굵은 숫자를 보라: **손가락마다 벌림·굽힘의 순서가 두 체계에서 반대**다. 이게 `joint_mapping.md` 의 주제다.

### 이 저장소의 규약 (외울 것 세 줄)

1. **순서**: 코드 안의 16차원 벡터는 특별한 말이 없으면 **MuJoCo 순서** (`MUJOCO_JOINT_NAMES`). 실기 모터 순서로 바꾸는
   건 `joint_map` 이 한다.
2. **단위**: 라디안. 로그에만 도(deg)를 쓴다.
3. **0점**: 편 손 = 0 (MuJoCo). 실기는 편 손 = π(모터 180°). 변환은 `mujoco_to_leaphand()`.

## 실행

```bash
conda activate leap-hand
python -m leap_hand_mapping.joint_map
```

```
MuJoCo 관절이름  손가락  모터ID       MJCF 범위          실기 범위           교집합            텔레옵
  0 if_mcp   검지     1  [-0.314,+2.230]  [-0.314,+2.230]  [-0.314,+2.230]  [-0.314,+2.230]
  1 if_rot   검지     0  [-1.047,+1.047]  [-1.047,+1.047]  [-1.047,+1.047]  [-1.047,+0.052]
  ...
self_check 통과
```

그리고 파이썬에서 직접:

```python
from leap_hand_mapping import joint_map as jm
import numpy as np
q = np.zeros(16); q[jm.MUJOCO_JOINT_NAMES.index("if_mcp")] = 0.5   # 검지 굽힘 0.5 rad
print(jm.mujoco_to_leaphand(q).round(3))     # 실기 명령: 모터 1 이 π+0.5, 나머지 π
print(jm.MUJOCO_TO_MOTOR)                    # [1 0 2 3 5 4 6 7 9 8 10 11 12 13 14 15]
```

시뮬로 눈에 익히기(실기 없이):

```bash
python scripts/phase0/p0_3_sweep_joints.py          # MuJoCo 뷰어에서 관절을 하나씩 왕복
```

## 결과 읽는 법

- `mujoco_to_leaphand(q)` 결과에서 **0.5 가 들어간 자리가 인덱스 1** 이다(모터 ID 1). 이름은 `if_mcp` 인데 자리는 두 번째.
- 범위 열 네 개가 다 같지 않다. `th_axl`/`th_mcp` 는 두 모델이 다르고(`joint_mapping.md`), `*_rot` 은 텔레옵 열이 훨씬 좁다(`debugging_casebook.md` 사례 13).
- `p0_3` 에서 벌림 모터(0/4/8)가 움직일 때 옆 손가락과의 간격을 보라. 편 손에서는 ±7.7° 만 벌려도 닿는다(`../design/joint_mapping.md`).
