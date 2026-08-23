"""LEAP Hand v1 관절 매핑 테이블 (MuJoCo <-> 실기).

인수인계 문서 5장("최대 함정: 관절 이름/순서 불일치")의 산출물.

배경
----
MuJoCo 모델(mujoco_menagerie/leap_hand)은 leap-hand 공식 repo가 아니라
dexsuite/dex-urdf에서 파생되었고, 그 과정에서 body/joint 이름이 바뀌었다.
그 결과 MuJoCo의 16개 관절 순서와 LEAP_Hand_API의 모터 ID 순서가 일치하지 않는다.

두 순서의 관계는 실제 모델 파일에서 다음과 같이 확인했다.

1. URDF(Bidex_VisionPro_Teleop/leap_hand_mesh_right/robot_pybullet.urdf)의
   joint 이름 "0".."15" 는 그대로 실기 모터 ID 다.
   근거: URDF의 관절 범위 16개가 LEAP_Hand_API 의
   leap_hand_utils.LEAPsim_limits() 배열과 순서까지 완전히 일치한다.

2. 두 모델의 운동학 체인을 비교하면 손가락(검지/중지/약지)은 앞 두 관절이 서로 뒤바뀐다.

       MuJoCo                          URDF / 모터 ID
       palm  -> if_bs : if_mcp         palm_lower -> mcp_joint : 1
       if_bs -> if_px : if_rot         mcp_joint  -> pip       : 0
       if_px -> if_md : if_pip         pip        -> dip       : 2
       if_md -> if_ds : if_dip         dip        -> fingertip : 3

   즉 MuJoCo는 [굽힘(mcp), 벌림(rot), pip, dip] 순서이고
   실기는   [벌림(0),   굽힘(1),   pip, dip] 순서다.

3. 엄지는 체인 순서가 그대로 대응된다(치환 없음).

       palm  -> th_mp : th_cmc         palm_lower -> pip_4           : 12
       th_mp -> th_bs : th_axl         pip_4      -> thumb_pip       : 13
       th_bs -> th_px : th_mcp         thumb_pip  -> thumb_dip       : 14
       th_px -> th_ds : th_ipl         thumb_dip  -> thumb_fingertip : 15

각도 규약
--------
MuJoCo qpos 는 LEAP_Hand_API 가 "LEAPsim" 이라 부르는 규약과 같은 0점을 쓴다.
실기("LEAPhand")는 모터의 180도가 0점이므로 pi 만큼 오프셋이 있다.
(leap_hand_utils.LEAPsim_to_LEAPhand 참조)

    실기 각도 = MuJoCo 각도 + pi        (순서 치환을 적용한 뒤)

주의
----
관절 범위는 두 모델이 완전히 같지 않다. 엄지 th_axl / th_mcp 에서 다르다.
같은 로봇이지만 MJCF 는 dexsuite URDF, 실기 규약은 leap-hand 공식 URDF 에서
왔기 때문이다. 안전하게 쓰려면 LIMITS_INTERSECTION_MJ 로 클립할 것.
"""

from __future__ import annotations

import numpy as np

NUM_JOINTS = 16

# MuJoCo(menagerie / mujoco_playground) 의 qpos = ctrl 순서.
# mujoco_playground 의 leap_hand_constants.JOINT_NAMES 와 동일하므로
# 이 매핑 테이블은 Phase 2(MJX 학습)에서도 그대로 재사용된다.
MUJOCO_JOINT_NAMES = [
    "if_mcp", "if_rot", "if_pip", "if_dip",   # 검지
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",   # 중지
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",   # 약지
    "th_cmc", "th_axl", "th_mcp", "th_ipl",   # 엄지
]

# MUJOCO_TO_MOTOR[i] = MuJoCo i 번째 관절에 대응하는 실기 모터 ID.
MUJOCO_TO_MOTOR = np.array(
    [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15], dtype=np.int64
)

# MOTOR_TO_MUJOCO[i] = 실기 모터 ID i 에 대응하는 MuJoCo 관절 인덱스.
# 이 치환은 (0 1)(4 5)(8 9) 짝바꿈뿐이라 자기 자신이 역치환이다.
MOTOR_TO_MUJOCO = np.argsort(MUJOCO_TO_MOTOR)

# 실기(LEAPhand) 0점과 시뮬(LEAPsim) 0점의 차이. leap_hand_utils 와 같은 값을 쓴다.
SIM_TO_REAL_OFFSET = 3.14159

# MJCF(right_hand.xml)에서 읽은 관절 범위. MuJoCo 순서.
LIMITS_MJCF_LOWER = np.array(
    [-0.314, -1.047, -0.506, -0.366,
     -0.314, -1.047, -0.506, -0.366,
     -0.314, -1.047, -0.506, -0.366,
     -0.349, -0.349, -0.470, -1.340]
)
LIMITS_MJCF_UPPER = np.array(
    [2.230, 1.047, 1.885, 2.042,
     2.230, 1.047, 1.885, 2.042,
     2.230, 1.047, 1.885, 2.042,
     2.094, 2.094, 2.443, 1.880]
)

# LEAP_Hand_API 의 leap_hand_utils.LEAPsim_limits(). 실기 모터 ID 순서.
LIMITS_LEAPSIM_LOWER = np.array(
    [-1.047, -0.314, -0.506, -0.366,
     -1.047, -0.314, -0.506, -0.366,
     -1.047, -0.314, -0.506, -0.366,
     -0.349, -0.470, -1.200, -1.340]
)
LIMITS_LEAPSIM_UPPER = np.array(
    [1.047, 2.230, 1.885, 2.042,
     1.047, 2.230, 1.885, 2.042,
     1.047, 2.230, 1.885, 2.042,
     2.094, 2.443, 1.900, 1.880]
)

# 두 모델 모두가 허용하는 교집합. MuJoCo 순서.
LIMITS_INTERSECTION_MJ_LOWER = np.maximum(
    LIMITS_MJCF_LOWER, LIMITS_LEAPSIM_LOWER[MUJOCO_TO_MOTOR]
)
LIMITS_INTERSECTION_MJ_UPPER = np.minimum(
    LIMITS_MJCF_UPPER, LIMITS_LEAPSIM_UPPER[MUJOCO_TO_MOTOR]
)

# 텔레오퍼레이션용 추가 제한 (geon, 2026-08-23). 벌림(rot) 관절이 옆 손가락에 걸려 전류가
# 한계에 붙는 일이 있었다(라이브 rf_rot 468 동결). 모델 한계 안이라도 실기에서는 서로 닿는다.
#   부호는 MJCF 로 확인: if_rot + = 중지 쪽, rf_rot - = 중지 쪽, mf_rot + = 약지 쪽 / - = 검지 쪽.
#   mf_rot  : 좌우 3도
#   if_rot  : 중지 쪽으로 3도까지만, 반대쪽(벌리기)은 모델 한계 유지
#   rf_rot  : 중지 쪽으로 3도까지만, 반대쪽 유지
# clip_mujoco 가 이 표를 쓰므로 IK 해, 시뮬 ctrl, 실기 명령이 전부 같은 범위를 받는다.
ROT_TOWARD_MIDDLE = np.radians(3.0)
LIMITS_TELEOP_MJ_LOWER = LIMITS_INTERSECTION_MJ_LOWER.copy()
LIMITS_TELEOP_MJ_UPPER = LIMITS_INTERSECTION_MJ_UPPER.copy()
LIMITS_TELEOP_MJ_UPPER[MUJOCO_JOINT_NAMES.index("if_rot")] = +ROT_TOWARD_MIDDLE
LIMITS_TELEOP_MJ_LOWER[MUJOCO_JOINT_NAMES.index("mf_rot")] = -ROT_TOWARD_MIDDLE
LIMITS_TELEOP_MJ_UPPER[MUJOCO_JOINT_NAMES.index("mf_rot")] = +ROT_TOWARD_MIDDLE
LIMITS_TELEOP_MJ_LOWER[MUJOCO_JOINT_NAMES.index("rf_rot")] = -ROT_TOWARD_MIDDLE


def _as_vec(joints) -> np.ndarray:
    arr = np.asarray(joints, dtype=np.float64)
    if arr.shape[-1] != NUM_JOINTS:
        raise ValueError(f"16차원 벡터를 기대했으나 shape={arr.shape} 를 받았다")
    return arr


def mujoco_to_motor_order(q_mujoco) -> np.ndarray:
    """MuJoCo 관절 순서 -> 실기 모터 ID 순서. (각도 오프셋은 적용하지 않는다)"""
    return _as_vec(q_mujoco)[..., MOTOR_TO_MUJOCO]


def motor_to_mujoco_order(q_motor) -> np.ndarray:
    """실기 모터 ID 순서 -> MuJoCo 관절 순서. (각도 오프셋은 적용하지 않는다)"""
    return _as_vec(q_motor)[..., MUJOCO_TO_MOTOR]


def mujoco_to_leaphand(q_mujoco) -> np.ndarray:
    """MuJoCo qpos -> 실기에 그대로 명령할 수 있는 16차원 각도(rad)."""
    return mujoco_to_motor_order(q_mujoco) + SIM_TO_REAL_OFFSET


def leaphand_to_mujoco(q_real) -> np.ndarray:
    """실기 관절각(rad) -> MuJoCo qpos / ctrl."""
    return motor_to_mujoco_order(_as_vec(q_real) - SIM_TO_REAL_OFFSET)


LIMIT_TABLES = {
    "teleop": (LIMITS_TELEOP_MJ_LOWER, LIMITS_TELEOP_MJ_UPPER),          # 텔레옵 기본: 벌림 관절 3도 제한
    "model": (LIMITS_INTERSECTION_MJ_LOWER, LIMITS_INTERSECTION_MJ_UPPER),  # 두 모델 교집합 (학습 정책 경로용)
}


def clip_mujoco(q_mujoco, limits: str = "teleop") -> np.ndarray:
    """관절 범위로 클립. MuJoCo 순서 입출력.

    limits="teleop" (기본): 교집합 + 벌림 관절 텔레옵 제한. 사람 손 리타겟 경로.
    limits="model": 두 모델의 교집합만. 학습 정책(rotate_z)은 벌림을 ±20도 넘게 쓰므로 이걸 쓴다.
    """
    lo, hi = LIMIT_TABLES[limits]
    return np.clip(_as_vec(q_mujoco), lo, hi)


def safe_leaphand_command(q_mujoco) -> np.ndarray:
    """MuJoCo qpos 를 클립까지 거쳐 실기 명령으로 변환. 실기 전송 전에 이걸 쓸 것."""
    return mujoco_to_leaphand(clip_mujoco(q_mujoco))


def describe() -> str:
    """매핑 테이블을 사람이 읽는 표로 출력."""
    finger = ["검지"] * 4 + ["중지"] * 4 + ["약지"] * 4 + ["엄지"] * 4
    lines = [
        f"{'MuJoCo':>3s} {'관절이름':<8s} {'손가락':<5s} {'모터ID':>5s}"
        f" {'MJCF 범위':>18s} {'실기 범위':>18s} {'교집합':>18s} {'텔레옵':>18s}",
        "-" * 96,
    ]
    for i in range(NUM_JOINTS):
        mid = MUJOCO_TO_MOTOR[i]
        lines.append(
            f"{i:>3d} {MUJOCO_JOINT_NAMES[i]:<8s} {finger[i]:<5s} {mid:>5d}"
            f"  [{LIMITS_MJCF_LOWER[i]:+.3f},{LIMITS_MJCF_UPPER[i]:+.3f}]"
            f"  [{LIMITS_LEAPSIM_LOWER[mid]:+.3f},{LIMITS_LEAPSIM_UPPER[mid]:+.3f}]"
            f"  [{LIMITS_INTERSECTION_MJ_LOWER[i]:+.3f},{LIMITS_INTERSECTION_MJ_UPPER[i]:+.3f}]"
            f"  [{LIMITS_TELEOP_MJ_LOWER[i]:+.3f},{LIMITS_TELEOP_MJ_UPPER[i]:+.3f}]"
        )
    return "\n".join(lines)


def self_check() -> None:
    """치환이 서로 역이고 왕복 변환이 항등인지 확인."""
    assert np.array_equal(MUJOCO_TO_MOTOR[MOTOR_TO_MUJOCO], np.arange(NUM_JOINTS))
    assert np.array_equal(MOTOR_TO_MUJOCO[MUJOCO_TO_MOTOR], np.arange(NUM_JOINTS))
    rng = np.random.default_rng(0)
    q = rng.uniform(LIMITS_MJCF_LOWER, LIMITS_MJCF_UPPER)
    assert np.allclose(leaphand_to_mujoco(mujoco_to_leaphand(q)), q)
    assert np.all(LIMITS_INTERSECTION_MJ_LOWER < LIMITS_INTERSECTION_MJ_UPPER)
    assert np.all(LIMITS_TELEOP_MJ_LOWER < LIMITS_TELEOP_MJ_UPPER)
    assert np.all(LIMITS_TELEOP_MJ_LOWER >= LIMITS_INTERSECTION_MJ_LOWER)
    assert np.all(LIMITS_TELEOP_MJ_UPPER <= LIMITS_INTERSECTION_MJ_UPPER)


def to_dict() -> dict:
    """언어/OS 에 무관하게 재사용할 수 있도록 매핑을 순수 자료구조로 내보낸다."""
    return {
        "description": "LEAP Hand v1 관절 매핑 (MuJoCo menagerie/playground <-> LEAP_Hand_API 모터 ID)",
        "num_joints": NUM_JOINTS,
        "mujoco_joint_names": MUJOCO_JOINT_NAMES,
        "mujoco_to_motor": MUJOCO_TO_MOTOR.tolist(),
        "motor_to_mujoco": MOTOR_TO_MUJOCO.tolist(),
        "sim_to_real_offset_rad": SIM_TO_REAL_OFFSET,
        "limits_mjcf_mujoco_order": {
            "lower": LIMITS_MJCF_LOWER.tolist(),
            "upper": LIMITS_MJCF_UPPER.tolist(),
        },
        "limits_leapsim_motor_order": {
            "lower": LIMITS_LEAPSIM_LOWER.tolist(),
            "upper": LIMITS_LEAPSIM_UPPER.tolist(),
        },
        "limits_teleop_mujoco_order": {
            "lower": LIMITS_TELEOP_MJ_LOWER.tolist(),
            "upper": LIMITS_TELEOP_MJ_UPPER.tolist(),
        },
        "limits_intersection_mujoco_order": {
            "lower": LIMITS_INTERSECTION_MJ_LOWER.tolist(),
            "upper": LIMITS_INTERSECTION_MJ_UPPER.tolist(),
        },
    }


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH", help="매핑을 JSON 파일로 내보낸다")
    ns = ap.parse_args()

    self_check()
    if ns.json:
        with open(ns.json, "w", encoding="utf-8") as f:
            json.dump(to_dict(), f, indent=2, ensure_ascii=False)
        print(f"{ns.json} 에 기록했다")
    else:
        print(describe())
        print("\nself_check 통과")
