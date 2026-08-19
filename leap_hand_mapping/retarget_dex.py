"""dex-retargeting 을 이 프로젝트의 규약에 맞춰 감싼다.

왜 직접 만든 retarget.py 를 두고 이걸 쓰는가
------------------------------------------
직접 만든 쪽은 사람 손끝의 **절대 위치**를 목표로 IK 를 푼다. 그런데 사람과 LEAP 은
비율이 반대라(손가락/손바닥 폭이 사람 1.2~1.4 : LEAP 0.98, 말단 마디가 사람 35% :
LEAP 83~89%) 목표가 자주 도달 불가능해진다. 그러면 IK 가 관절 한계에 붙고, 재시도
시드를 전부 소진한 뒤 덜 나쁜 해를 고르는데 그 선택이 프레임마다 갈아타며 손이 떨린다.
손가락별 스케일링, 손끝 앵커, 엄지 정렬 캘리브레이션으로 많이 줄였지만 구조적 한계다.

dex-retargeting 은 절대 위치가 아니라 **벡터**를 맞춘다.

    vector    손목 -> 각 손끝 벡터 4개를 scaling_factor 배해서 목표로 삼고,
              Huber loss 로 관절각을 직접 최적화한다(nlopt).
    dexpilot  거기에 더해 **손끝끼리의 거리**(엄지-검지 등 6쌍)를 목표에 넣는다.
              집기/핀치에서 중요한 것이 절대 위치가 아니라 손끝 사이 거리라는
              DexPilot 의 관찰을 따른다.

벡터 집합의 최소자승 해는 언제나 존재하므로 "도달 불가능한 목표"라는 실패 모드가
구조적으로 생기지 않는다. 이전 프레임 해에 대한 정규화 항과 low-pass 필터도 안에
들어 있어 지터도 자체적으로 억제한다.

출처: AnyTeleop (Qin et al., RSS 2023), https://github.com/dexsuite/dex-retargeting (MIT).
로봇 모델은 dex-urdf 에서 온다 — Phase 0 에서 MuJoCo menagerie 모델의 출처로
확인했던 바로 그 저장소다.

관절 순서
--------
dex-retargeting 이 내놓는 관절 순서는 URDF 파싱 순서라 우리 MuJoCo 순서와 다르다.

    dex     ['1','0','2','3', '12','13','14','15', '5','4','6','7', '9','8','10','11']
            검지            엄지               중지            약지
    MuJoCo  검지            중지               약지            엄지

이름(= 실기 모터 ID)으로 대응시켜 재배열한다. dex-retargeting 의 FAQ 도 인덱스가
아니라 **관절 이름으로** 맞추라고 명시한다. Phase 0 의 MUJOCO_TO_MOTOR 가 그대로 쓰인다.
"""

from __future__ import annotations

import os

import numpy as np

from leap_hand_mapping import hand_tracker as ht
from leap_hand_mapping import joint_map as jm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URDF_DIR = os.path.join(REPO, "third_party/dex-urdf/robots/hands")

# MediaPipe world 랜드마크 좌표계 -> MANO 규약. 공식 예제와 동일한 상수다.
OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
OPERATOR2MANO_LEFT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=float)


def estimate_wrist_frame(keypoints: np.ndarray) -> np.ndarray:
    """손목 기준 좌표계를 랜드마크에서 추정한다. 공식 예제의 구현 그대로.

    손목/검지 MCP/중지 MCP 세 점으로 손바닥 평면을 SVD 로 맞추고, 그 법선과
    손바닥 방향으로 정규직교 프레임을 만든다.
    """
    points = keypoints[[0, 5, 9], :]
    x_vector = points[0] - points[2]

    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, v = np.linalg.svd(centered)
    normal = v[2, :]

    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # 새끼->검지 방향이 MANO 의 z 축과 같은 쪽을 보게 부호를 맞춘다.
    if np.sum(z * (centered[1] - centered[2])) < 0:
        normal = -normal
        z = -z
    return np.stack([x, normal, z], axis=1)


class DexRetargeter:
    """MediaPipe 21 랜드마크 -> LEAP 16 관절각(MuJoCo 순서).

    LeapRetargeter 와 같은 자리에 끼울 수 있도록 인터페이스를 맞춰 두었다.
    엄지 캘리브레이션은 필요 없다 — 손목 프레임을 매 프레임 랜드마크에서 추정하고
    벡터를 맞추므로 사람마다 다른 엄지 안착 각도가 문제되지 않는다.
    """

    def __init__(
        self,
        hand_type: str = "Right",
        retargeting_type: str = "dexpilot",
        urdf_dir: str = DEFAULT_URDF_DIR,
        scaling_factor: float | None = None,
        low_pass_alpha: float | None = None,
        max_speed: float = 8.0,
    ) -> None:
        from dex_retargeting.constants import (
            HandType,
            RetargetingType,
            RobotName,
            get_default_config_path,
        )
        from dex_retargeting.retargeting_config import RetargetingConfig

        if not os.path.isdir(urdf_dir):
            raise FileNotFoundError(
                f"dex-urdf 자산이 없다: {urdf_dir}\n"
                "  git clone --depth 1 https://github.com/dexsuite/dex-urdf.git third_party/dex-urdf"
            )
        RetargetingConfig.set_default_urdf_dir(urdf_dir)

        config_path = get_default_config_path(
            RobotName.leap,
            RetargetingType[retargeting_type],
            HandType[hand_type.lower()],
        )
        override = {}
        if scaling_factor is not None:
            override["scaling_factor"] = scaling_factor
        if low_pass_alpha is not None:
            override["low_pass_alpha"] = low_pass_alpha
        config = RetargetingConfig.load_from_file(config_path, override or None)

        self.retargeting_type = retargeting_type
        self.config_path = config_path
        self.retargeting = config.build()
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )
        self.max_speed = max_speed

        # 관절 이름(= 실기 모터 ID)으로 MuJoCo 순서를 만든다.
        names = list(self.retargeting.joint_names)
        try:
            self.dex_to_mujoco = np.array(
                [names.index(str(m)) for m in jm.MUJOCO_TO_MOTOR], dtype=int
            )
        except ValueError as exc:
            raise RuntimeError(
                f"dex-retargeting 관절 이름을 모터 ID 로 해석할 수 없다: {names}"
            ) from exc

        indices = self.retargeting.optimizer.target_link_human_indices
        self.origin_indices = np.asarray(indices[0, :], dtype=int)
        self.task_indices = np.asarray(indices[1, :], dtype=int)

        self._q = np.zeros(jm.NUM_JOINTS)
        self.last_objective = float("nan")
        self.last_restarts = 0   # 이 구현에는 재시도 개념이 없다. 지표 호환용.

    # ------------------------------------------------------------------ 변환

    def to_mano_frame(self, world: np.ndarray) -> np.ndarray:
        """MediaPipe world 랜드마크 -> 손목 원점 MANO 프레임."""
        keypoints = np.asarray(world, dtype=float)
        keypoints = keypoints - keypoints[0:1, :]
        return keypoints @ estimate_wrist_frame(keypoints) @ self.operator2mano

    def reference_vectors(self, world: np.ndarray) -> np.ndarray:
        """최적화가 맞출 목표 벡터. vector 는 4개, dexpilot 은 10개."""
        joint_pos = self.to_mano_frame(world)
        return joint_pos[self.task_indices, :] - joint_pos[self.origin_indices, :]

    def retarget(self, world: np.ndarray, dt: float = 0.02) -> np.ndarray:
        """21 랜드마크 -> 16 관절각(MuJoCo 순서)."""
        qpos = self.retargeting.retarget(self.reference_vectors(world))
        q = jm.clip_mujoco(np.asarray(qpos)[self.dex_to_mujoco])

        # nlopt 가 도달한 목적함수 값. 벡터 오차의 Huber loss 라 mm 단위는 아니지만
        # 추종이 무너지면 같이 커지므로 감시 지표로 쓸 수 있다.
        try:
            self.last_objective = float(
                self.retargeting.optimizer.opt.last_optimum_value()
            )
        except Exception:
            self.last_objective = float("nan")

        # 속도 제한만 우리 쪽에서 건다. 평활은 dex-retargeting 안의 low-pass 가 한다.
        # 실기 안전장치라 최적화가 무엇을 내놓든 마지막에 걸려 있어야 한다.
        step = np.clip(q - self._q, -self.max_speed * dt, self.max_speed * dt)
        self._q = jm.clip_mujoco(self._q + step)
        return self._q.copy()

    # -------------------------------------------------------- 인터페이스 호환

    def tip_error(self) -> np.ndarray:
        """LeapRetargeter 와 자리를 맞추기 위한 것. 여기서는 목적함수 값을 돌려준다.

        dex-retargeting 은 절대 위치가 아니라 벡터를 맞추므로 '손끝 잔차 mm' 라는
        수치가 같은 뜻을 갖지 않는다. 지표를 섞어 읽지 않도록 이름만 맞춰 둔다.
        """
        return np.full(8, self.last_objective)

    def set_pose(self, q) -> None:
        self._q = jm.clip_mujoco(q)

    def reset(self) -> None:
        self._q = np.zeros(jm.NUM_JOINTS)
        self.retargeting.reset()

    def observe_calibration(self, world: np.ndarray) -> None:
        """캘리브레이션이 필요 없다. 호출되어도 아무것도 하지 않는다."""

    def finish_calibration(self) -> bool:
        return False

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
