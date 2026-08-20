"""관절 매핑 테이블을 순기구학(FK)으로 교차검증한다.

인수인계 문서 5장은 "실기와 MuJoCo에 같은 16차원 벡터를 넣고 한 관절씩 순차 구동하여
육안 대조"를 검증 방법으로 제시한다. 이 스크립트는 그 절차를 자동화한 것으로,
실기 대신 실기와 같은 규약을 쓰는 URDF(PyBullet, headless)를 기준으로 삼는다.

  MuJoCo  : mujoco_menagerie/leap_hand/right_hand.xml   (dexsuite 파생)
  PyBullet: Bidex_VisionPro_Teleop/.../robot_pybullet.urdf (leap-hand 공식, 모터 ID 순서)

두 모델은 계보가 달라 베이스 프레임과 링크 로컬 원점 규약이 서로 다를 수 있다.
그래서 손끝의 절대 좌표를 직접 비교하지 않고, 좌표계에 무관한 양인
**네 손끝 사이의 쌍거리 6개**를 비교한다.

  - 매핑이 맞으면: 자세를 바꿔도 오차가 거의 변하지 않는다(상수 오프셋만 남음).
    따라서 판정 지표는 평균이 아니라 **오차의 표준편차**다.
  - 매핑이 틀리면: 자세에 따라 오차가 크게 요동친다.

사용법:
    python scripts/phase0/p0_1_verify_mapping_fk.py            # 전체 검증
    python scripts/phase0/p0_1_verify_mapping_fk.py --samples 500
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from leap_hand_mapping import joint_map as jm  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MJCF = os.path.join(REPO, "third_party/mujoco_menagerie/leap_hand/right_hand.xml")
URDF = os.path.join(
    REPO, "third_party/Bidex_VisionPro_Teleop/leap_hand_mesh_right/robot_pybullet.urdf"
)

# 비교에 쓸 기준점. [검지, 중지, 약지, 엄지] 순으로 맞춰 둔다.
#
# 말단 링크의 '원점'을 쓰면 안 된다. 그 링크의 부모 관절(dip/ipl, 모터 3/7/11/15)이
# 움직여도 원점은 제자리라서 말단 4개 관절이 검증에서 빠져 버린다.
# 실제 손끝에 해당하는 점을 써야 16개 관절 전부가 지표에 반영된다.
#
# 단, 두 모델이 말하는 '손끝'은 서로 다른 점이다.
#   URDF: realtip = fingertip 프레임 + (0.02, -0.07, 0.015)  <- 실제 접촉점
#   MJCF: if_tip geom 중심 = (-0.0013, -0.0336, 0.0145)      <- 메시 중심
# 그대로 비교하면 지렛대 길이가 달라 매핑과 무관한 오차가 섞인다.
# 그래서 양쪽 모두 URDF 의 realtip 오프셋을 말단 링크 프레임에 적용해 기준점을 통일한다.
# (공식 IK 코드 avp_leap.py 가 타깃으로 삼는 점도 realtip 이다)
MJ_TIP_BODIES = ["if_ds", "mf_ds", "rf_ds", "th_ds"]
PB_TIP_PARENTS = ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]
TIP_OFFSETS = np.array(
    [
        [0.02, -0.07, 0.015],
        [0.02, -0.07, 0.015],
        [0.02, -0.07, 0.015],
        [0.00, -0.07, -0.015],
    ]
)

PAIRS = list(itertools.combinations(range(4), 2))  # 6개 쌍

# 검증할 매핑 후보들. 실제 채택안이 다른 후보를 이겨야 의미가 있다.
CANDIDATES = {
    "채택안(체인기반)": jm.MUJOCO_TO_MOTOR,
    "치환없음(항등)": np.arange(16),
    "손가락만치환,엄지1칸시프트": np.array(
        [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 13, 14, 15, 12]
    ),
    "엄지역순": np.array([1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 15, 14, 13, 12]),
    # th_axl 은 축방향 회전이라 단독 구동 시 손끝이 0.1mm 밖에 안 움직인다.
    # 그만큼 지표에 약하게 잡히므로, 인접한 th_mcp 와 뒤바뀐 경우를 따로 배제해 둔다.
    "엄지13-14뒤바꿈": np.array([1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 14, 13, 15]),
    # 손가락 배정 자체가 뒤바뀐 경우(중지 <-> 약지)도 배제해 둔다.
    "중지-약지 블록교환": np.array([1, 0, 2, 3, 9, 8, 10, 11, 5, 4, 6, 7, 12, 13, 14, 15]),
}


class MujocoFK:
    def __init__(self) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(MJCF)
        self.data = mujoco.MjData(self.model)
        names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        if names != jm.MUJOCO_JOINT_NAMES:
            raise RuntimeError(
                "MJCF 관절 순서가 joint_map.MUJOCO_JOINT_NAMES 와 다르다.\n"
                f"  모델: {names}\n  테이블: {jm.MUJOCO_JOINT_NAMES}"
            )
        self.tip_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            for b in MJ_TIP_BODIES
        ]
        if -1 in self.tip_ids:
            raise RuntimeError(f"MJCF 에서 말단 body {MJ_TIP_BODIES} 를 못 찾았다")

    def tips(self, q_mujoco: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = q_mujoco
        self._mujoco.mj_kinematics(self.model, self.data)
        out = []
        for k, bid in enumerate(self.tip_ids):
            rot = self.data.xmat[bid].reshape(3, 3)
            out.append(self.data.xpos[bid] + rot @ TIP_OFFSETS[k])
        return np.array(out)


class PybulletFK:
    def __init__(self) -> None:
        import pybullet as pb

        self._pb = pb
        self.cid = pb.connect(pb.DIRECT)
        self.body = pb.loadURDF(URDF, useFixedBase=True, physicsClientId=self.cid)
        self.joint_by_motor_id: dict[int, int] = {}
        link_index: dict[str, int] = {}
        for i in range(pb.getNumJoints(self.body, physicsClientId=self.cid)):
            info = pb.getJointInfo(self.body, i, physicsClientId=self.cid)
            name, jtype, link = info[1].decode(), info[2], info[12].decode()
            link_index[link] = i
            if jtype == pb.JOINT_REVOLUTE:
                self.joint_by_motor_id[int(name)] = i
        missing = set(range(16)) - set(self.joint_by_motor_id)
        if missing:
            raise RuntimeError(f"URDF 에서 모터 ID {sorted(missing)} 관절을 못 찾았다")
        self.tip_ids = [link_index[n] for n in PB_TIP_PARENTS]

    def tips(self, q_motor: np.ndarray) -> np.ndarray:
        for motor_id, angle in enumerate(q_motor):
            self._pb.resetJointState(
                self.body,
                self.joint_by_motor_id[motor_id],
                float(angle),
                physicsClientId=self.cid,
            )
        out = []
        for k, lid in enumerate(self.tip_ids):
            st = self._pb.getLinkState(
                self.body, lid, computeForwardKinematics=True, physicsClientId=self.cid
            )
            pos, orn = np.array(st[4]), st[5]  # worldLinkFramePosition / Orientation
            rot = np.array(self._pb.getMatrixFromQuaternion(orn)).reshape(3, 3)
            out.append(pos + rot @ TIP_OFFSETS[k])
        return np.array(out)


def pair_distances(tips: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(tips[a] - tips[b]) for a, b in PAIRS])


def evaluate(mj: MujocoFK, pbf: PybulletFK, perm: np.ndarray, qs: np.ndarray) -> dict:
    """perm[i] = MuJoCo i 번 관절 <-> 모터 ID. 쌍거리 오차 통계를 낸다."""
    errs = []
    for q_mj in qs:
        q_motor = np.empty(16)
        q_motor[perm] = q_mj  # MuJoCo i 번 값을 모터 perm[i] 로 보낸다
        errs.append(pair_distances(mj.tips(q_mj)) - pair_distances(pbf.tips(q_motor)))
    errs = np.array(errs)
    return {
        "std_mm": float(errs.std(axis=0).mean() * 1000.0),
        "ptp_mm": float(np.ptp(errs, axis=0).mean() * 1000.0),
        "bias_mm": float(np.abs(errs.mean(axis=0)).mean() * 1000.0),
    }


def sweep_report(mj: MujocoFK, pbf: PybulletFK, delta: float = 0.5) -> np.ndarray:
    """문서 5장의 '한 관절씩 순차 구동'을 자동화.

    0 자세에서 관절 하나만 delta 만큼 움직였을 때 네 손끝이 얼마나 움직이는지를
    두 모델에서 각각 재고 비교한다. 매핑이 맞으면 같은 손가락이 비슷한 크기로 움직인다.
    """
    zero = np.zeros(16)
    mj_base, pb_base = mj.tips(zero), pbf.tips(zero)
    rows = []
    print(
        f"\n[관절 단독 구동 대조]  각 관절을 0 -> {delta} rad 로 움직였을 때 "
        "손끝 이동량(mm), 검지/중지/약지/엄지"
    )
    print(f"{'MuJoCo 관절':<10s} {'모터':>4s}  {'MuJoCo 이동량':<28s} {'URDF 이동량':<28s} {'차이':>8s}")
    print("-" * 88)
    for i in range(16):
        q_mj = zero.copy()
        q_mj[i] = delta
        q_motor = zero.copy()
        q_motor[jm.MUJOCO_TO_MOTOR[i]] = delta
        d_mj = np.linalg.norm(mj.tips(q_mj) - mj_base, axis=1) * 1000.0
        d_pb = np.linalg.norm(pbf.tips(q_motor) - pb_base, axis=1) * 1000.0
        diff = np.abs(d_mj - d_pb).max()
        rows.append(diff)
        fmt = lambda v: " ".join(f"{x:6.1f}" for x in v)  # noqa: E731
        print(
            f"{jm.MUJOCO_JOINT_NAMES[i]:<10s} {jm.MUJOCO_TO_MOTOR[i]:>4d}  "
            f"{fmt(d_mj):<28s} {fmt(d_pb):<28s} {diff:>7.1f}"
        )
    return np.array(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    jm.self_check()
    print(jm.describe())

    mj, pbf = MujocoFK(), PybulletFK()
    rng = np.random.default_rng(args.seed)
    qs = rng.uniform(
        jm.LIMITS_INTERSECTION_MJ_LOWER,
        jm.LIMITS_INTERSECTION_MJ_UPPER,
        size=(args.samples, 16),
    )

    print(f"\n[매핑 후보 비교]  무작위 자세 {args.samples}개, 손끝 쌍거리 오차")
    print("판정 지표는 std(자세에 따른 요동). 매핑이 맞으면 작아야 한다.")
    print(f"\n{'후보':<28s} {'std(mm)':>9s} {'변동폭(mm)':>11s} {'상수편향(mm)':>13s}")
    print("-" * 66)
    results = {}
    for name, perm in CANDIDATES.items():
        r = evaluate(mj, pbf, np.asarray(perm), qs)
        results[name] = r
        print(f"{name:<28s} {r['std_mm']:>9.2f} {r['ptp_mm']:>11.2f} {r['bias_mm']:>13.2f}")

    sweep_diff = sweep_report(mj, pbf)

    best = min(results, key=lambda k: results[k]["std_mm"])
    adopted = "채택안(체인기반)"
    runner_up = min((k for k in results if k != adopted), key=lambda k: results[k]["std_mm"])
    ratio = results[runner_up]["std_mm"] / max(results[adopted]["std_mm"], 1e-9)

    print("\n[판정]")
    print(f"  최소 std 후보 : {best}")
    print(f"  채택안 std    : {results[adopted]['std_mm']:.2f} mm")
    print(f"  차점 후보 std : {results[runner_up]['std_mm']:.2f} mm ({runner_up}) -> {ratio:.1f}배")
    print(f"  단독 구동 최대 불일치 : {sweep_diff.max():.1f} mm ({jm.MUJOCO_JOINT_NAMES[int(sweep_diff.argmax())]})")

    ok = best == adopted and ratio > 3.0
    print(f"\n  결과: {'통과 — 매핑 테이블 확정' if ok else '실패 — 매핑 재검토 필요'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
