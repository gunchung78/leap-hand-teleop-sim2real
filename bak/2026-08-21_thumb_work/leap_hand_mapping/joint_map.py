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

관절 범위
--------
기준은 하나다: LEAP_Hand_API 의 `LEAPsim_limits()` (= 출처 URDF dex-urdf 의 limit 와 동일).
MuJoCo 순서로 옮긴 것이 LIMITS_MJ_LOWER/UPPER 이고, 클립·IK·트윈·실기 전부 이걸 쓴다.
menagerie MJCF 의 범위는 엄지 th_axl / th_mcp 에서 이와 어긋나므로(아래 주석), MuJoCo
모델을 로드한 직후 apply_model_limits() 로 모델 쪽을 공식값에 맞춘다.
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

# 공식 관절 범위 (기준). MuJoCo 순서.
# LEAP_Hand_API 의 LEAPsim_limits() 를 MuJoCo 순서로 옮긴 것. 출처 URDF(dex-urdf) 의
# limit 와 16개 전부 일치한다. 클립(clip_mujoco), IK 범위, 디지털 트윈, 실기 명령이 전부
# 이 하나를 쓴다.
LIMITS_MJ_LOWER = LIMITS_LEAPSIM_LOWER[MUJOCO_TO_MOTOR]
LIMITS_MJ_UPPER = LIMITS_LEAPSIM_UPPER[MUJOCO_TO_MOTOR]

# menagerie MJCF 와 공식의 불일치 (기록용)
# ------------------------------------
# right_hand.xml 의 엄지 관절 범위 두 개가 출처 URDF(dexsuite/dex-urdf, 해시 2ee2f70)와
# 어긋난다. URDF 에는 joint 12/13/14/15 가 차례로
#     [-0.349, 2.094] [-0.47, 2.443] [-1.20, 1.90] [-1.34, 1.88]
# 인데 MJCF 는 th_axl 에 joint 12 의 값을, th_mcp 에 joint 13 의 값을 넣었다(한 칸씩 밀림).
# LEAPsim_limits() 는 URDF 와 같으므로 공식 두 출처가 일치하고 menagerie 만 다르다.
# (CHANGELOG 의 "Fixed Left Leap Hand's Thumb CMC range" 와 같은 부류. 오른손은 못 잡은 듯.)
#
# 한때 "MJCF ∩ LEAPsim" 교집합으로 클립했는데, 그러면 th_mcp 하한이 -26.9도(공식 -68.8도)
# 에서 잘려 엄지가 실물보다 42도 덜 젖혀졌다. 실물 LEAP 은 공식 범위를 쓴다(실기 확인).
# 공식 범위로 넓혀도 MuJoCo 자기충돌은 없다(th_mcp 단독 -68.8도, th_cmc 86~115도 조합 모두
# 접촉 0). 그래서 교집합을 버리고 공식 하나로 통일했다. 아래는 어긋난 관절의 목록이며
# describe() 의 * 표시와 apply_model_limits() 가 이걸 근거로 모델을 고친다.
MJCF_MISMATCH = [
    MUJOCO_JOINT_NAMES[i] for i in range(NUM_JOINTS)
    if not (np.isclose(LIMITS_MJCF_LOWER[i], LIMITS_MJ_LOWER[i])
            and np.isclose(LIMITS_MJCF_UPPER[i], LIMITS_MJ_UPPER[i]))
]   # == ["th_axl", "th_mcp"]


def apply_model_limits(model) -> list[str]:
    """로드한 MuJoCo 모델의 관절 범위를 공식 한계(LIMITS_MJ)로 맞춘다. MJCF 로드 직후 부를 것.

    MJCF_MISMATCH 의 관절(엄지 th_axl/th_mcp)만 실제로 바뀐다 — 나머지 14개는 MJCF 가
    이미 공식과 같다. 바뀐 관절 이름 목록을 돌려준다. jnt_range 와 해당 액추에이터의
    ctrlrange 를 함께 바꾼다(position 액추에이터는 ctrlrange 로도 막히기 때문).
    """
    import mujoco

    changed = []
    for i, name in enumerate(MUJOCO_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        lo, hi = LIMITS_MJ_LOWER[i], LIMITS_MJ_UPPER[i]
        if np.allclose(model.jnt_range[jid], (lo, hi)):
            continue
        model.jnt_range[jid] = (lo, hi)
        for aid in np.where(model.actuator_trnid[:, 0] == jid)[0]:
            model.actuator_ctrlrange[aid] = (lo, hi)
        changed.append(name)
    return changed


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


def clip_mujoco(q_mujoco) -> np.ndarray:
    """공식 관절 범위(LEAPsim_limits)로 클립. MuJoCo 순서 입출력."""
    return np.clip(
        _as_vec(q_mujoco), LIMITS_MJ_LOWER, LIMITS_MJ_UPPER
    )


def safe_leaphand_command(q_mujoco) -> np.ndarray:
    """MuJoCo qpos 를 클립까지 거쳐 실기 명령으로 변환. 실기 전송 전에 이걸 쓸 것."""
    return mujoco_to_leaphand(clip_mujoco(q_mujoco))


def describe() -> str:
    """매핑 테이블을 사람이 읽는 표로 출력."""
    finger = ["검지"] * 4 + ["중지"] * 4 + ["약지"] * 4 + ["엄지"] * 4
    lines = [
        f"{'MuJoCo':>3s} {'관절이름':<8s} {'손가락':<5s} {'모터ID':>5s}"
        f" {'MJCF 범위':>18s} {'공식(LEAPsim)':>18s} {'적용(LIMITS_MJ)':>18s}",
        "-" * 96,
    ]
    for i in range(NUM_JOINTS):
        mid = MUJOCO_TO_MOTOR[i]
        lines.append(
            f"{i:>3d} {MUJOCO_JOINT_NAMES[i]:<8s} {finger[i]:<5s} {mid:>5d}"
            f"  [{LIMITS_MJCF_LOWER[i]:+.3f},{LIMITS_MJCF_UPPER[i]:+.3f}]"
            f"  [{LIMITS_LEAPSIM_LOWER[mid]:+.3f},{LIMITS_LEAPSIM_UPPER[mid]:+.3f}]"
            f"  [{LIMITS_MJ_LOWER[i]:+.3f},{LIMITS_MJ_UPPER[i]:+.3f}]"
            + ("  *" if MUJOCO_JOINT_NAMES[i] in MJCF_MISMATCH else "")
        )
    lines.append("* menagerie MJCF 범위가 공식(출처 URDF = LEAPsim_limits)과 어긋난다. 적용값은 공식이다"
                 " (MJCF_MISMATCH, apply_model_limits 가 모델을 고친다)")
    return "\n".join(lines)


def self_check() -> None:
    """치환이 서로 역이고 왕복 변환이 항등인지 확인."""
    assert np.array_equal(MUJOCO_TO_MOTOR[MOTOR_TO_MUJOCO], np.arange(NUM_JOINTS))
    assert np.array_equal(MOTOR_TO_MUJOCO[MUJOCO_TO_MOTOR], np.arange(NUM_JOINTS))
    rng = np.random.default_rng(0)
    q = rng.uniform(LIMITS_MJCF_LOWER, LIMITS_MJCF_UPPER)
    assert np.allclose(leaphand_to_mujoco(mujoco_to_leaphand(q)), q)
    assert np.all(LIMITS_MJ_LOWER < LIMITS_MJ_UPPER)
    # 적용 한계는 공식(LEAPsim) 그 자체다. MJCF 와의 불일치는 엄지 두 관절뿐이어야 한다.
    assert MJCF_MISMATCH == ["th_axl", "th_mcp"], MJCF_MISMATCH


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
        "limits_mujoco_order": {
            "note": "적용 한계. LEAPsim_limits() 를 MuJoCo 순서로 옮긴 것(출처 URDF 와 동일). "
                    "menagerie MJCF 는 mjcf_mismatch 의 관절에서 이와 어긋나므로 MuJoCo 로드 후 "
                    "apply_model_limits() 로 jnt_range/ctrlrange 를 이 값에 맞출 것.",
            "lower": LIMITS_MJ_LOWER.tolist(),
            "upper": LIMITS_MJ_UPPER.tolist(),
        },
        "mjcf_mismatch": MJCF_MISMATCH,
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
