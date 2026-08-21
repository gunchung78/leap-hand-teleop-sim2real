"""리타겟팅 기하를 카메라 없이 검증한다.

문제
----
"웹캠 들고 손 흔들어 보니 대충 따라오더라"는 검증이 아니다. Phase 0 에서 겪었듯이
좌표계가 뒤집혀 있거나 축이 하나 어긋나 있어도 눈으로는 그럴듯해 보인다.
사람 손이 개입하지 않는, 정답이 있는 시험이 필요하다.

방법
----
로봇 자신을 사람이라고 치고 왕복시킨다.

    1. LEAP 을 알려진 자세 q 로 둔다.
    2. 그 자세에서 MediaPipe 21 랜드마크 배치에 대응하는 점들을 뽑아
       "가짜 사람 손"을 만든다. 사람 손 크기로 줄이고, 카메라 앞에서
       손을 아무렇게나 든 상황을 흉내내려 임의 회전/평행이동까지 준다.
    3. 그 랜드마크를 리타겟터에 넣는다.
    4. 돌아온 관절각이 q 와 같은가?

리타겟터는 손바닥 좌표계를 스스로 세우고 스케일을 스스로 정하므로, 2번에서 준
회전/평행이동/크기는 전부 상쇄되어야 한다. 상쇄되지 않으면 좌표계 구성이 틀린 것이다.

한계
----
가짜 사람 손은 LEAP 의 기하를 그대로 쓴다. 따라서 이 시험은 **좌표 변환과 IK 가
서로 맞물리는지**를 보는 것이지, 사람 손 비율과 LEAP 비율의 차이에서 오는
리타겟팅 품질을 보증하지는 않는다. 그쪽은 실제 웹캠으로만 확인할 수 있다.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from leap_hand_mapping import hand_tracker as ht  # noqa: E402
from leap_hand_mapping import joint_map as jm  # noqa: E402
from leap_hand_mapping.retarget import (  # noqa: E402
    T_THUMB_TIP as R_T_THUMB_TIP,
    EE_LINKS,
    FINGERS,
    MCP_LINKS,
    ROOT_LINKS,
    LeapRetargeter,
)

# 사람 손 크기로 줄일 때 쓰는 값. 리타겟터가 배율을 스스로 다시 계산하므로
# 구체적인 값 자체는 결과에 영향을 주지 않는다. 상쇄되는지 보는 게 목적이다.
HUMAN_PALM_WIDTH = 0.040

# 사람 손가락에서 MCP->PIP 와 PIP->DIP 의 길이 비. 근위지골이 중위지골보다 길다.
PHALANX_RATIO = 0.6


def middle_joint(root: np.ndarray, end: np.ndarray, total: float,
                 bend_dir: np.ndarray) -> np.ndarray:
    """뿌리와 끝 사이에 마디 두 개짜리 사슬의 중간 관절을 놓는다.

    사람 손가락은 강체 마디로 이어져 있어서 굽혀도 마디 길이 합이 변하지 않는다.
    가짜 사람 손도 그래야 리타겟터가 계산하는 길이 배율이 자세에 무관해진다.
    (LEAP 링크 원점을 그대로 PIP 로 쓰면 그 사이에 관절이 둘 끼어 있어서
     길이가 자세에 따라 변한다. 그러면 왕복이 정확할 수 없다.)

    total 을 뿌리->끝 도달거리로 잡아 두면 배율이 정확히 1/축소배율로 나와
    시험이 닫힌다. 대신 이 시험은 **길이 배율 자체의 타당성은 보증하지 않는다** —
    그쪽은 실제 손을 재서 확인할 문제다(README 의 비율 표).
    """
    u = end - root
    d = float(np.linalg.norm(u))
    l1 = total * PHALANX_RATIO
    l2 = total - l1
    if d < 1e-9 or d >= l1 + l2:
        # 완전히 뻗은 (또는 도달거리를 넘는) 경우. 직선 위에 놓는다.
        direction = u / d if d > 1e-9 else np.array([1.0, 0.0, 0.0])
        return root + direction * min(l1, d)

    u_hat = u / d
    along = (d * d + l1 * l1 - l2 * l2) / (2.0 * d)
    perp = bend_dir - np.dot(bend_dir, u_hat) * u_hat
    n = np.linalg.norm(perp)
    perp = perp / n if n > 1e-9 else np.zeros(3)
    height = np.sqrt(max(l1 * l1 - along * along, 0.0))
    return root + along * u_hat + height * perp


def fake_human_landmarks(rt: LeapRetargeter, q: np.ndarray, rng,
                         palm_width: float = HUMAN_PALM_WIDTH) -> np.ndarray:
    """자세 q 의 LEAP 에서 MediaPipe 배치의 가짜 사람 랜드마크 21개를 만든다."""
    p = rt.p
    for i, joint in enumerate(rt.dof_indices):
        p.resetJointState(rt.uid, joint, float(q[i]), physicsClientId=rt.client)

    root = {f: rt._link_origin(idx) for f, idx in ROOT_LINKS.items()}
    mcp = {f: rt._link_origin(idx) for f, idx in MCP_LINKS.items()}
    dip = {f: rt._link_origin(EE_LINKS[f][0]) for f in EE_LINKS}
    tip = {f: rt._link_origin(EE_LINKS[f][1]) for f in EE_LINKS}

    # 중간 관절은 손바닥 쪽으로 굽힌다. 사람 손이 그렇게 굽는다.
    bend = -rt.reference.rotation[:, 2]
    # 가짜 사람 손의 뿌리->손끝 사슬 길이가 LEAP 도달거리와 같아야 배율이 닫힌다.
    # 말단 마디(DIP->TIP)는 강체라 이미 고정이므로 나머지를 중간 관절로 채운다.
    mid = {
        f: middle_joint(root[f], dip[f],
                        rt.reference.reach[f] - rt.reference.distal_length[f], bend)
        for f in FINGERS
    }

    # 손목은 랜드마크 배치를 채우기 위한 점이다. 리타겟터는 이 점을 손가락 방향
    # (손목 -> 중지 MCP)에만 쓰므로, 영점 자세의 손가락 방향 반대편에 둔다.
    ref = rt.reference
    finger_dir = ref.rotation[:, 1]
    wrist = mcp["middle"] - 0.09 * finger_dir

    lm = np.zeros((ht.NUM_LANDMARKS, 3))
    lm[ht.WRIST] = wrist
    for f, i_mcp, i_pip, i_dip, i_tip in [
        ("index", ht.INDEX_MCP, ht.INDEX_PIP, ht.INDEX_DIP, ht.INDEX_TIP),
        ("middle", ht.MIDDLE_MCP, ht.MIDDLE_PIP, ht.MIDDLE_DIP, ht.MIDDLE_TIP),
        ("ring", ht.RING_MCP, ht.RING_PIP, ht.RING_DIP, ht.RING_TIP),
    ]:
        lm[i_mcp] = mcp[f]
        lm[i_pip] = mid[f]
        lm[i_dip] = dip[f]
        lm[i_tip] = tip[f]
    lm[ht.THUMB_CMC] = root["thumb"]
    lm[ht.THUMB_MCP] = mid["thumb"]
    lm[ht.THUMB_IP] = dip["thumb"]
    lm[ht.THUMB_TIP] = tip["thumb"]

    # 사람 손 크기로 축소 + 임의 강체변환. 리타겟터가 전부 상쇄해야 한다.
    scale = palm_width / ref.palm_width
    lm = scale * lm

    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0, 2 * np.pi)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return lm @ R.T + rng.uniform(-0.3, 0.3, size=3)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amplitude", type=float, default=0.6,
                    help="시험 자세의 관절각 범위 비율 (0~1 에 가까울수록 크게 굽힘)")
    args = ap.parse_args()

    jm.self_check()
    rng = np.random.default_rng(args.seed)

    # 평활화/속도제한은 시간축 필터라 1회 변환 정확도를 가린다. 여기서는 끈다.
    #
    # 엄지 결합(_couple_thumb)도 끈다. 이 시험은 **배율이 상쇄되는가**를 보는데,
    # 결합의 project_dist/escape_dist 는 절대 길이(30/50mm)라 정의상 상쇄되지 않는다.
    # 켜 두면 여기서 재는 것이 좌표계 정합성이 아니라 가짜 손 크기가 되어 버린다.
    # 결합의 크기 민감도는 아래 hand_size_sensitivity 에서 따로 잰다.
    # 엄지 각도 매핑(thumb_mode="map")도 끈다. 그쪽은 손끝 위치를 맞추는 경로가 아니라
    # 이 시험의 판정 기준(손끝 잔차)과 무관하다. 여기서는 IK 경로만 본다.
    rt = LeapRetargeter(smoothing=1.0, max_speed=1e9, distal_mode="leap",
                        thumb_couple=False, thumb_mode="ik")

    lo = jm.LIMITS_INTERSECTION_MJ_LOWER
    hi = jm.LIMITS_INTERSECTION_MJ_UPPER
    mid = 0.5 * (lo + hi)

    joint_err = []
    tip_err = []
    for _ in range(args.trials):
        span = args.amplitude * 0.5 * (hi - lo)
        q = np.clip(mid + rng.uniform(-1, 1, size=jm.NUM_JOINTS) * span, lo, hi)

        lm = fake_human_landmarks(rt, q, rng)
        rt.reset()
        q_hat = rt.retarget(lm)

        joint_err.append(np.abs(q_hat - q))
        tip_err.append(rt.tip_error())

    joint_err = np.degrees(np.array(joint_err))
    tip_err = np.array(tip_err) * 1000.0
    # 목표 8개 중 홀수 번째가 손끝, 짝수 번째가 앞마디다.
    dip_err = tip_err[:, 0::2]
    tip_only = tip_err[:, 1::2]

    print(f"시행 {args.trials}회, 자세 진폭 {args.amplitude}")
    print()
    print(f"관절각 오차   평균 {joint_err.mean():6.2f} deg   최대 {joint_err.max():6.2f} deg")
    print(f"손끝 잔차     평균 {tip_only.mean():6.2f} mm    최대 {tip_only.max():6.2f} mm   <- 판정 기준")
    print(f"앞마디 잔차   평균 {dip_err.mean():6.2f} mm    최대 {dip_err.max():6.2f} mm   (보조 목표)")
    print()
    print(f"{'관절':>3} {'이름':<8} {'평균(deg)':>10} {'최대(deg)':>10}")
    for i in range(jm.NUM_JOINTS):
        print(f"{i:>3} {jm.MUJOCO_JOINT_NAMES[i]:<8} "
              f"{joint_err[:, i].mean():>10.2f} {joint_err[:, i].max():>10.2f}")

    rt.close()

    # 손끝이 본 목표다. 앞마디는 가중치를 낮춘 보조 목표라 판정에서 뺀다.
    # (자세를 잡아 주는 역할이라 남는 잔차는 정상이다. retarget.py 의 dip_weight 참조)
    ok = tip_only.mean() < 1.0
    print()
    print("판정:", "통과 — 좌표 변환과 IK 가 일관적이다" if ok
          else "실패 — 손끝 잔차가 크다. 좌표계 구성을 의심할 것")

    hand_size_sensitivity(args, rng)
    return 0 if ok else 1


def hand_size_sensitivity(args, rng) -> None:
    """엄지 결합이 사람 손 크기에 얼마나 민감한지 잰다.

    나머지 파이프라인은 배율을 스스로 재서 상쇄한다. 그런데 엄지 결합의
    project_dist(30mm)/escape_dist(50mm)는 **절대 길이**라 상쇄되지 않는다.
    손이 작으면 벌린 자세도 "붙이려는 중"으로 읽혀 엄지가 검지 쪽으로 딸려 간다.

    두 상수는 DexPilot(dex-retargeting leap_hand_right_dexpilot.yml) 값 그대로다.
    upstream 도 실제 MediaPipe 랜드마크(진짜 미터)에 같은 절대값을 쓴다. 그래서
    바꾸지 않고, 대신 얼마나 흔들리는지를 여기에 기록해 둔다.

    '엄지 목표 이동'은 결합이 엄지 손끝 목표를 원래 자리에서 얼마나 옮겼는지다.
    0 이면 결합이 아무것도 안 한 것이고, 크면 크게 끌어당긴 것이다.
    """
    print()
    print("엄지 결합의 손 크기 민감도")
    print("  가짜 손을 여러 크기로 만들어 결합이 목표를 얼마나 옮기는지 본다.")
    print(f"  {'손바닥폭mm':>10}{'엄지목표이동mm':>15}{'손끝잔차mm':>12}")

    rt = LeapRetargeter(smoothing=1.0, max_speed=1e9, distal_mode="leap", thumb_mode="ik")
    lo, hi = jm.LIMITS_INTERSECTION_MJ_LOWER, jm.LIMITS_INTERSECTION_MJ_UPPER
    mid = 0.5 * (lo + hi)
    for pw in (0.030, 0.040, 0.050, 0.060, 0.070):
        shift, resid = [], []
        for _ in range(max(10, args.trials // 5)):
            span = args.amplitude * 0.5 * (hi - lo)
            q = np.clip(mid + rng.uniform(-1, 1, size=jm.NUM_JOINTS) * span, lo, hi)
            lm = fake_human_landmarks(rt, q, rng, palm_width=pw)
            rt.thumb_couple = False
            base = rt.compute_targets(lm)
            rt.thumb_couple = True
            with_c = rt.compute_targets(lm)
            shift.append(np.linalg.norm(with_c[R_T_THUMB_TIP] - base[R_T_THUMB_TIP]))
            rt.reset()
            rt.retarget(lm)
            resid.append(rt.tip_error()[1::2].mean())
        print(f"  {pw*1000:10.0f}{np.mean(shift)*1000:15.1f}{np.mean(resid)*1000:12.2f}")
    rt.close()
    print("  실제 사람 손바닥 폭은 실측 49.5mm 였다 (thumb_capture.npz).")


if __name__ == "__main__":
    raise SystemExit(main())
