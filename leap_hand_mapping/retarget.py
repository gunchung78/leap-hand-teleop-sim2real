"""사람 손 랜드마크 -> LEAP Hand 16 관절각 (리타겟팅 + IK).

Phase 1 의 핵심. hand_tracker 가 준 21개 랜드마크를 로봇 관절각으로 바꾼다.
공식 Bidex_VisionPro_Teleop/avp_leap.py 의 PyBullet IK 방식을 그대로 따르되,
입력을 Apple Vision Pro 대신 MediaPipe 로 바꾸고 스케일링을 명시적으로 만들었다.

왜 IK 인가
---------
사람 손과 LEAP 은 관절 구조가 다르다(사람 손가락은 MCP 2자유도 + PIP + DIP,
LEAP 도 4자유도지만 링크 길이와 축 배치가 다르다). 관절각을 그대로 베끼면
손끝이 엉뚱한 데로 간다. 조작에서 중요한 것은 관절각이 아니라 **손끝 위치**이므로
손끝을 목표로 IK 를 푼다. 공식 구현의 코멘트도 같은 말을 한다.

    "Note how the fingertip positions are matching, but the joint angles between
     the two hands are not due to the IK solution."

관절 순서 (Phase 0 과의 연결)
---------------------------
PyBullet 이 URDF 에서 읽은 revolute 관절의 순서는

    [1, 0, 2, 3,  5, 4, 6, 7,  9, 8, 10, 11,  12, 13, 14, 15]   (관절 이름 = 모터 ID)

로, joint_map.MUJOCO_TO_MOTOR 와 **정확히 같다**. 둘 다 운동학 체인 순서를 따르기
때문이다. 즉 calculateInverseKinematics2 의 출력은 이미 MuJoCo 관절 순서다.
실기로 보낼 때만 joint_map.safe_leaphand_command() 를 거치면 된다.
(avp_leap.py 가 결과에 [::-1] 짝바꿈을 하는 이유가 이것이다. 그쪽은 모터 ID
 순서로 내보내려고 치환하는 것이고, 우리는 MuJoCo 순서를 쓰므로 치환하지 않는다.)

기하 변환
--------
1. 사람 손과 로봇 손에 **같은 방식으로** 손바닥 좌표계를 만든다.
       원점  검지 MCP 와 약지 MCP 의 중점
       y축   손가락이 뻗는 방향
       x축   검지 MCP -> 약지 MCP
       z축   x * y (손바닥 법선)
2. 손바닥 폭(검지 MCP <-> 약지 MCP) 비율로 균일 스케일한다. LEAP 이 사람 손보다
   두 배 넘게 크다.
3. 손가락마다 두 점(DIP, TIP)을 목표로 준다. 네 손가락 x 2 = 8개 목표.

말단 마디 길이 보정
-----------------
사람 손을 스케일해도 말단 마디(DIP->TIP) 길이는 LEAP 과 맞지 않는다. LEAP 의
'realtip' 은 fingertip 프레임에서 (0.02, -0.07, 0.015) 떨어진 실제 접촉점이라
스케일된 사람 손끝보다 훨씬 멀다. 목표를 그대로 주면 두 목표를 동시에 만족할 수
없어 IK 가 어중간한 자세로 수렴한다.

그래서 TIP 목표는 **방향만 사람에게서 받고 길이는 LEAP 자신의 값**을 쓴다.
이러면 두 목표가 항상 도달 가능한 조합이 되어 IK 가 깨끗하게 풀린다.
(distal_mode="scaled" 로 끄고 비교해 볼 수 있다.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from leap_hand_mapping import hand_tracker as ht
from leap_hand_mapping import joint_map as jm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URDF = os.path.join(
    REPO, "third_party/Bidex_VisionPro_Teleop/leap_hand_mesh_right/robot_pybullet.urdf"
)

# PyBullet 링크 인덱스. 손가락마다 (말단 마디 시작, 실제 접촉점).
# 링크 이름은 fingertip / realtip 계열이다. 위 URDF 덤프로 확인했다.
EE_LINKS = {
    "index":  (3, 4),
    "middle": (8, 9),
    "ring":   (13, 14),
    "thumb":  (18, 19),
}

# 대응하는 MediaPipe 랜드마크. 엄지는 IP 관절이 다른 손가락의 DIP 자리에 해당한다.
EE_LANDMARKS = {
    "index":  (ht.INDEX_DIP, ht.INDEX_TIP),
    "middle": (ht.MIDDLE_DIP, ht.MIDDLE_TIP),
    "ring":   (ht.RING_DIP, ht.RING_TIP),
    "thumb":  (ht.THUMB_IP, ht.THUMB_TIP),
}

FINGERS = ["index", "middle", "ring", "thumb"]

# MuJoCo 관절 순서에서 손가락별 4개 자유도.
# 손가락끼리는 운동학적으로 독립이라 자코비안이 블록대각이고, 따라서 IK 실패도
# 손가락 단위로 일어난다. 실패한 손가락만 다른 시드로 다시 푸는 데 쓴다.
FINGER_JOINTS = {
    "index":  [0, 1, 2, 3],
    "middle": [4, 5, 6, 7],
    "ring":   [8, 9, 10, 11],
    "thumb":  [12, 13, 14, 15],
}

# 시드용 관절각을 재기 위한 랜드마크 (MCP, PIP, DIP, TIP).
SEED_LANDMARKS = {
    "index":  (ht.INDEX_MCP, ht.INDEX_PIP, ht.INDEX_DIP, ht.INDEX_TIP),
    "middle": (ht.MIDDLE_MCP, ht.MIDDLE_PIP, ht.MIDDLE_DIP, ht.MIDDLE_TIP),
    "ring":   (ht.RING_MCP, ht.RING_PIP, ht.RING_DIP, ht.RING_TIP),
}

# LEAP 손바닥 좌표계를 만들 기준 링크 (영점 자세에서의 원점 위치를 쓴다).
MCP_LINKS = {"index": 0, "middle": 5, "ring": 10}

# 손가락별 뿌리. 각 손가락 목표를 이 점에 앵커링한다.
# 엄지는 pip_4(모터 12)가 손바닥에 붙는 첫 링크라 그것이 뿌리다.
ROOT_LINKS = {"index": 0, "middle": 5, "ring": 10, "thumb": 15}

# 사람 쪽 대응 뿌리와, 뿌리에서 EE 까지의 마디 사슬.
# 마디 길이의 합은 손을 굽히든 펴든 변하지 않으므로 "뻗었을 때 길이"로 쓸 수 있다.
# 뿌리 -> ... -> 손끝. 마디 길이의 합이 "뻗었을 때 길이"다.
HUMAN_CHAIN = {
    "index":  (ht.INDEX_MCP, ht.INDEX_PIP, ht.INDEX_DIP, ht.INDEX_TIP),
    "middle": (ht.MIDDLE_MCP, ht.MIDDLE_PIP, ht.MIDDLE_DIP, ht.MIDDLE_TIP),
    "ring":   (ht.RING_MCP, ht.RING_PIP, ht.RING_DIP, ht.RING_TIP),
    "thumb":  (ht.THUMB_CMC, ht.THUMB_MCP, ht.THUMB_IP, ht.THUMB_TIP),
}


def orthonormal_frame(across: np.ndarray, along: np.ndarray) -> np.ndarray:
    """두 벡터로 정규직교 회전행렬을 만든다. 열이 (x, y, z) 축.

    along(손가락 방향)을 y축으로 신뢰하고, across 에서 y 성분을 뺀 나머지를 x축으로
    쓴다. 랜드마크 잡음으로 두 벡터가 정확히 수직이 아니어도 안전하다.
    """
    y = along / np.linalg.norm(along)
    x = across - np.dot(across, y) * y
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        raise ValueError("손바닥 좌표계를 만들 수 없다 (두 기준 벡터가 평행하다)")
    x = x / nx
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """단위벡터 a 를 b 로 보내는 최소 회전 (로드리게스)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        # 같은 방향이거나 정반대. 정반대면 아무 수직축으로 180도 돌린다.
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


@dataclass
class LeapReference:
    """LEAP 손의 영점 자세 기하. URDF 에서 한 번만 읽는다."""

    origin: np.ndarray              # 손바닥 좌표계 원점 (base 프레임)
    rotation: np.ndarray            # (3,3) base <- 손바닥 좌표계
    palm_width: float               # 검지 MCP <-> 약지 MCP
    distal_length: dict = field(default_factory=dict)   # 손가락별 fingertip -> realtip
    root: dict = field(default_factory=dict)            # 손가락별 뿌리 링크 원점
    reach: dict = field(default_factory=dict)           # 뿌리 -> 손끝 직선거리 (영점 자세)
    thumb_rest: np.ndarray | None = None                # 손바닥 좌표계에서 본 엄지 안착 방향


class LeapRetargeter:
    """MediaPipe 손 랜드마크를 LEAP 관절각(MuJoCo 순서)으로 바꾼다.

    IK 계산은 항상 DIRECT 클라이언트에서 한다(렌더링 없음).
    gui=True 로 하면 표시 전용 클라이언트가 하나 더 붙어서, IK **최종 결과**와
    목표점을 PyBullet 창으로 볼 수 있다. 계산과 표시를 분리해 둔 이유는
    생성자 주석에 적었다.
    """

    def __init__(
        self,
        urdf_path: str = DEFAULT_URDF,
        gui: bool = False,
        scale: float | None = None,
        distal_mode: str = "leap",
        ik_iterations: int = 30,
        ik_tolerance: float = 2e-4,
        ik_damping: float = 0.02,
        ik_max_step: float = 0.3,
        restart_threshold: float = 0.003,
        dip_weight: float = 0.3,
        smoothing: float = 0.4,
        max_speed: float = 8.0,
    ) -> None:
        import pybullet as p

        if distal_mode not in ("leap", "scaled"):
            raise ValueError("distal_mode 는 'leap' 또는 'scaled'")

        self.p = p
        self.gui = gui
        self.scale_override = scale
        self.distal_mode = distal_mode
        self.ik_iterations = ik_iterations
        self.ik_tolerance = ik_tolerance
        self.ik_damping = ik_damping
        self.ik_max_step = ik_max_step
        self.restart_threshold = restart_threshold
        # 목표 8개의 가중치. 손끝이 본 목표이고 앞마디는 자세를 잡아 주는 보조다.
        #
        # 앞마디 목표는 손끝에서 LEAP 말단 마디 길이만큼 되짚어 만든 점이라,
        # 이것까지 1.0 으로 맞추라고 하면 말단 링크의 **방향**까지 사람과 같으라는
        # 요구가 된다. 손가락에 남은 자유도로는 불가능해서 잔차가 5~26mm 에서
        # 안 내려가고, 그러면 재시도가 매 프레임 전부 돌아(33ms/프레임) 해가
        # 프레임마다 갈아타며 손이 떨린다.
        self.dip_weight = dip_weight
        self.smoothing = smoothing
        self.max_speed = max_speed

        # 계산용 클라이언트는 항상 DIRECT 다.
        #
        # IK 는 반복마다 resetJointState 로 자세를 바꿔 가며 푼다. 계산을 GUI
        # 클라이언트에서 하면 그 중간 자세가 전부 화면에 그려져서, 손이 미친듯이
        # 떨리고 손가락이 엉뚱한 데 갔다 오는 것처럼 보인다. 재시도 시드(굽힘
        # 0.4/0.9/1.4 자세)까지 찍히니 더 심하다. 명령이 튀는 게 아니라 탐색
        # 과정이 보이는 것이지만, 진단용 창이 진단을 방해하면 곤란하다.
        #
        # 그래서 표시용 클라이언트를 따로 두고 **최종 자세만** 넘긴다.
        self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, 0, physicsClientId=self.client)
        self.uid = p.loadURDF(
            urdf_path, [0, 0, 0], [0, 0, 0, 1],
            useFixedBase=True, physicsClientId=self.client,
        )

        self.gui_client = None
        self.gui_uid = None
        if gui:
            self.gui_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.gui_client)
            p.setGravity(0, 0, 0, physicsClientId=self.gui_client)
            self.gui_uid = p.loadURDF(
                urdf_path, [0, 0, 0], [0, 0, 0, 1],
                useFixedBase=True, physicsClientId=self.gui_client,
            )

        self.dof_indices = [
            i for i in range(p.getNumJoints(self.uid, physicsClientId=self.client))
            if p.getJointInfo(self.uid, i, physicsClientId=self.client)[2] != p.JOINT_FIXED
        ]
        if len(self.dof_indices) != jm.NUM_JOINTS:
            raise RuntimeError(f"revolute 관절이 16개가 아니다: {len(self.dof_indices)}")

        # PyBullet DOF 순서가 정말 MuJoCo 순서인지 확인한다. Phase 0 매핑의 전제다.
        names = [
            int(p.getJointInfo(self.uid, i, physicsClientId=self.client)[1])
            for i in self.dof_indices
        ]
        if names != jm.MUJOCO_TO_MOTOR.tolist():
            raise RuntimeError(
                f"URDF 관절 순서가 예상과 다르다: {names}\n"
                f"  기대: {jm.MUJOCO_TO_MOTOR.tolist()} (joint_map.MUJOCO_TO_MOTOR)"
            )

        self.ee_links = [idx for f in FINGERS for idx in EE_LINKS[f]]
        self.target_weights = np.repeat(
            np.array([self.dip_weight, 1.0] * len(FINGERS)), 3
        )
        self.reference = self._measure_reference()
        self._q = np.zeros(jm.NUM_JOINTS)
        self._last_targets: np.ndarray | None = None
        self.last_restarts = 0
        self.thumb_align = np.eye(3)
        self.frozen_scales: dict | None = None
        self._calib: list = []
        self._calib_scales: list = []
        self._debug_balls: list[int] = []
        if gui:
            self._create_debug_balls()

    # ------------------------------------------------------------------ 기하

    def _link_origin(self, link_index: int) -> np.ndarray:
        state = self.p.getLinkState(self.uid, link_index, physicsClientId=self.client)
        return np.array(state[4])

    def _measure_reference(self) -> LeapReference:
        """영점 자세에서 LEAP 의 손바닥 좌표계와 말단 마디 길이를 잰다."""
        for i in self.dof_indices:
            self.p.resetJointState(self.uid, i, 0.0, physicsClientId=self.client)

        mcp = {f: self._link_origin(idx) for f, idx in MCP_LINKS.items()}
        origin = 0.5 * (mcp["index"] + mcp["ring"])
        across = mcp["ring"] - mcp["index"]

        # 손가락 방향은 세 손가락의 MCP -> 말단마디 벡터를 평균해서 쓴다.
        along = np.mean(
            [self._link_origin(EE_LINKS[f][0]) - mcp[f] for f in MCP_LINKS], axis=0
        )

        distal = {
            f: float(np.linalg.norm(self._link_origin(b) - self._link_origin(a)))
            for f, (a, b) in EE_LINKS.items()
        }
        root = {f: self._link_origin(idx) for f, idx in ROOT_LINKS.items()}
        reach = {
            f: float(np.linalg.norm(self._link_origin(EE_LINKS[f][1]) - root[f]))
            for f in FINGERS
        }
        rotation = orthonormal_frame(across, along)
        # 목표를 손끝으로 앵커하므로 안착 방향도 손끝(realtip)까지 재야 한다.
        # 앞마디(thumb_fingertip)로 재면 마지막 마디만큼 어긋나서, 편 손인데도
        # th_cmc/th_axl 이 하한에 붙는다(실측).
        thumb_rest = rotation.T @ (self._link_origin(EE_LINKS["thumb"][1]) - root["thumb"])
        return LeapReference(
            origin=origin,
            rotation=rotation,
            palm_width=float(np.linalg.norm(across)),
            distal_length=distal,
            root=root,
            reach=reach,
            thumb_rest=thumb_rest / np.linalg.norm(thumb_rest),
        )

    def human_frame(self, world: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """사람 손 랜드마크 -> (원점, 회전행렬, 손바닥 폭). 로봇과 같은 방식."""
        origin = 0.5 * (world[ht.INDEX_MCP] + world[ht.RING_MCP])
        across = world[ht.RING_MCP] - world[ht.INDEX_MCP]
        along = world[ht.MIDDLE_MCP] - world[ht.WRIST]
        width = float(np.linalg.norm(across))
        return origin, orthonormal_frame(across, along), width

    def measure_scales(self, world: np.ndarray) -> dict:
        """손가락별 사람->로봇 길이 배율.

        왜 손바닥 폭 하나로 안 되는가
        ---------------------------
        LEAP 은 손바닥이 넓고 손가락이 짧다. 사람은 반대다. 웹캠으로 실측한 값:

            손가락  사람(MCP->DIP 마디합 / 손바닥 폭)   LEAP
            검지            1.23 ~ 1.27                 0.98
            중지            1.41 ~ 1.44                 0.98
            약지            1.23 ~ 1.28                 0.98

        손바닥 폭 비율로 균일 스케일하면 손끝 목표가 LEAP 도달거리보다 25~45%
        멀리 찍힌다. **구조적으로 도달 불가능**하므로 IK 가 관절 한계에 붙고,
        매 프레임 재시도 시드를 전부 소진한 뒤 덜 나쁜 해를 고른다. 그 선택이
        프레임마다 바뀌어서 손가락이 튄다(실측 잔차 29mm, 재시도 5회/프레임).

        그래서 손가락마다 **자기 길이 비율**로 스케일한다. 마디 길이의 합은 손을
        굽혀도 변하지 않으므로 매 프레임 안정적으로 잴 수 있다.
        """
        scales = {}
        for f, chain in HUMAN_CHAIN.items():
            length = sum(
                float(np.linalg.norm(world[chain[i + 1]] - world[chain[i]]))
                for i in range(len(chain) - 1)
            )
            scales[f] = self.reference.reach[f] / max(length, 1e-6)
        return scales

    def finger_scales(self, world: np.ndarray) -> dict:
        """실제로 쓰는 배율. 캘리브레이션을 했으면 그때 고정한 값이다."""
        return self.frozen_scales if self.frozen_scales else self.measure_scales(world)

    def compute_targets(self, world: np.ndarray) -> np.ndarray:
        """21 랜드마크 -> IK 목표점 8개 (LEAP base 프레임, 미터). 순서는 EE_LINKS 와 같다.

        손가락마다 **자기 뿌리를 원점으로** 삼고 자기 길이 배율로 스케일한다.
        손바닥 좌표계는 방향(회전)을 맞추는 데만 쓴다.

        손끝을 앵커로 삼는다
        -------------------
        LEAP 은 말단 마디가 도달거리의 83~89% 로 유별나게 길다(검지 74.3/161.1,
        엄지 71.6/148.5). 사람은 35% 근처다. 이 차이 때문에 손끝과 그 앞 마디를
        동시에 맞추는 것은 불가능하다. 둘 중 하나를 골라야 한다.

        조작에서 의미가 있는 것은 **손끝(접촉점)** 이다. 그래서 손끝 위치를 먼저
        맞추고, 그 앞 마디 목표는 LEAP 자신의 말단 마디 길이만큼 되짚어 만든다.
        방향은 사람에게서 받는다. 두 목표가 LEAP 기하와 항상 정합하므로 IK 가
        깨끗하게 풀린다.

        사람 비율로 합성한 손에서, 로봇 손끝이 사람 손끝(스케일)에서 벗어난 거리:

            자세      앞마디 앵커      손끝 앵커
            편 손        2.5mm          2.1mm
            살짝 굽힘   18.1mm          4.1mm
            반쯤 굽힘   34.5mm          7.0mm
            주먹        48.6mm         13.2mm

        목표 손끝은 뿌리에서 `배율 x 현재 뿌리->손끝 거리` 만큼 떨어지는데, 이 값은
        정의상 `배율 x 뻗었을 때 길이` = LEAP 도달거리 이하다. 즉 **거리 면에서는
        항상 도달 가능**하다. 남는 것은 방향이 관절 범위 안이냐뿐.
        """
        _, h_rot, _ = self.human_frame(world)
        ref = self.reference
        scales = self.finger_scales(world)
        gain = self.scale_override if self.scale_override else 1.0

        targets = []
        for f in FINGERS:
            chain = HUMAN_CHAIN[f]
            i_root, i_dip, i_tip = chain[0], chain[-2], chain[-1]
            s_f = scales[f] * gain

            # 엄지는 사람과 LEAP 의 안착 방향이 달라 정렬 회전을 한 번 더 건다.
            align = self.thumb_align if f == "thumb" else None

            def to_palm(v):
                local = h_rot.T @ v
                return align @ local if align is not None else local

            tip = ref.root[f] + ref.rotation @ (s_f * to_palm(world[i_tip] - world[i_root]))

            d = to_palm(world[i_tip] - world[i_dip])
            n = np.linalg.norm(d)
            if n < 1e-6:
                dip = tip
            elif self.distal_mode == "leap":
                # 손끝에서 LEAP 자신의 말단 마디 길이만큼 되짚는다. 방향만 사람 것.
                dip = tip - ref.distal_length[f] * (ref.rotation @ (d / n))
            else:
                dip = tip - ref.rotation @ (s_f * d)
            targets.append(dip)
            targets.append(tip)
        return np.array(targets)

    def observe_calibration(self, world: np.ndarray) -> None:
        """엄지 정렬용 표본을 모은다. 손을 편 상태로 몇 프레임 넣어 준다."""
        _, h_rot, _ = self.human_frame(world)
        # compute_targets 가 쓰는 것과 **같은 축**이어야 한다. CMC->IP 로 재고
        # CMC->TIP 로 목표를 만들면 마지막 마디만큼 어긋난다.
        d = h_rot.T @ (world[ht.THUMB_TIP] - world[ht.THUMB_CMC])
        n = np.linalg.norm(d)
        if n > 1e-6:
            self._calib.append(d / n)
            self._calib_scales.append(self.measure_scales(world))

    def finish_calibration(self) -> bool:
        """모은 표본으로 엄지 정렬 회전을 확정한다.

        왜 필요한가
        ----------
        사람 엄지가 손바닥 대비 놓인 방향과 LEAP 엄지 뿌리(pip_4)가 향한 방향은
        서로 다르다. 손바닥 좌표계 회전을 그대로 엄지에 적용하면 LEAP 이 낼 수
        없는 방향을 목표로 주게 되고, th_cmc 가 상한(2.094)에 붙은 채 손끝 잔차가
        40mm 대에서 안 내려간다. 목표 가중치를 조정해도(엄지 TIP 만 써도) 그대로다
        — 거리 문제가 아니라 **방향이 작업공간 밖**이기 때문이다.

        사람마다 엄지 안착 각도가 다르므로 상수로 박을 수 없다. 손을 편 자세를
        몇 프레임 보고, 그때의 엄지 방향을 LEAP 의 안착 방향으로 보내는 최소
        회전을 구해 둔다. 이후 엄지 움직임은 그 기준에서의 편차로 전달된다.
        """
        if len(self._calib) < 5:
            return False
        mean = np.mean(self._calib, axis=0)
        if np.linalg.norm(mean) < 1e-6:
            return False
        self.thumb_align = rotation_between(mean, self.reference.thumb_rest)

        # 배율을 여기서 고정한다.
        #
        # 마디 길이의 합은 원래 자세와 무관해야 하지만, MediaPipe 는 손가락을
        # 굽히면(특히 엄지) 가려짐 때문에 사슬을 짧게 추정한다. 실측에서 엄지
        # 배율이 편 손 1.67 -> 주먹 1.38 로 21% 흔들렸다. 매 프레임 다시 재면
        # 굽힐수록 로봇 손가락이 덜 뻗는다.
        #
        # 캘리브레이션 자세(편 손)는 가려짐이 가장 적어 길이를 가장 잘 잰다.
        # 그때 값을 붙박이로 쓴다.
        self.frozen_scales = {
            f: float(np.mean([s[f] for s in self._calib_scales]))
            for f in FINGERS
        }
        self._calib = []
        self._calib_scales = []
        return True

    def thumb_align_angle(self) -> float:
        """엄지 정렬 회전의 크기(rad). 캘리브레이션이 얼마나 보정했는지 보는 값."""
        c = (np.trace(self.thumb_align) - 1.0) / 2.0
        return float(np.arccos(np.clip(c, -1.0, 1.0)))

    def seed_from_human(self, world: np.ndarray) -> np.ndarray:
        """사람 손의 관절각을 그대로 재서 IK 시드로 쓴다.

        왜 필요한가
        ----------
        IK 를 영점에서 출발시키면 크게 굽힌 손에서 국소최소에 빠진다. 실제로
        중지/약지가 관절 한계에 붙은 채 손끝 잔차 158mm 로 수렴하는 자세가 나왔다
        (정답을 시드로 주면 같은 자세가 0.000mm 로 풀린다).

        사람 손 각도는 정확한 답은 아니지만 **올바른 골짜기**에 있다. 각도를 그대로
        쓰는 게 아니라 출발점으로만 쓰고, 손끝을 맞추는 일은 IK 에 맡긴다.

        부호 규약(URDF 실측)
        -------------------
        굽힘(mcp/pip/dip) 은 + 가 손바닥 쪽(-z), 벌림(rot) 은 + 가 약지 쪽(+x).
        사람 손도 같은 방식으로 손바닥 좌표계를 세우므로 그대로 대응된다.

        엄지는 LEAP 의 축 배치가 사람과 많이 달라(th_axl 은 축이 손끝을 지난다)
        각도를 옮기는 의미가 없다. 0 으로 두고 IK 에 맡긴다.
        """
        origin, rot, _ = self.human_frame(world)
        local = (world - origin) @ rot   # 손바닥 좌표계로. (21, 3)

        q = np.zeros(jm.NUM_JOINTS)
        for f, (i_mcp, i_pip, i_dip, i_tip) in SEED_LANDMARKS.items():
            s1 = local[i_pip] - local[i_mcp]
            s2 = local[i_dip] - local[i_pip]
            s3 = local[i_tip] - local[i_dip]
            # 각 마디가 손가락 방향(y) 대비 손바닥 쪽(-z)으로 얼마나 기울었는가.
            a1, a2, a3 = (np.arctan2(-s[2], s[1]) for s in (s1, s2, s3))
            j = FINGER_JOINTS[f]
            q[j[0]] = a1                          # mcp 굽힘
            q[j[1]] = np.arctan2(s1[0], s1[1])    # 벌림
            q[j[2]] = a2 - a1                     # pip
            q[j[3]] = a3 - a2                     # dip
        return jm.clip_mujoco(q)

    # -------------------------------------------------------------------- IK

    def _set_joints(self, q: np.ndarray) -> None:
        for i, joint in enumerate(self.dof_indices):
            self.p.resetJointState(self.uid, joint, float(q[i]), physicsClientId=self.client)

    _seed = _set_joints   # 다음 프레임 IK 의 시드. 해가 프레임마다 튀는 것을 막는다.

    def _ee_positions(self) -> np.ndarray:
        return np.array([self._link_origin(i) for i in self.ee_links])

    def _jacobian(self, q: np.ndarray) -> np.ndarray:
        """목표점 8개의 위치 자코비안을 세로로 쌓은 (24, 16) 행렬.

        이 URDF 는 말단 링크들의 관성 원점이 링크 원점과 같아서
        (실측: 8개 링크 전부 0.000mm) localPosition=[0,0,0] 이 곧 링크 프레임 원점이다.
        유한차분과 대조해 자코비안이 맞는 것을 확인했다(상대오차 0.00%).
        """
        zeros = [0.0] * jm.NUM_JOINTS
        ql = q.tolist()
        return np.vstack([
            np.array(self.p.calculateJacobian(
                self.uid, link, [0, 0, 0], ql, zeros, zeros, physicsClientId=self.client
            )[0])
            for link in self.ee_links
        ])

    def finger_residual(self, targets: np.ndarray, q: np.ndarray) -> dict:
        """자세 q 에서 손가락별 잔차(m). 앞마디는 가중치를 곱해 센다.

        앞마디를 1.0 으로 세면 도달 불가능한 잔차(5~26mm) 때문에 재시도가 매
        프레임 전부 돌고, 그 결과 해가 프레임마다 갈아타며 손이 떨린다.
        아예 빼 버리면 손끝만 맞고 앞마디가 엉뚱한 국소최소를 못 걸러 낸다.
        가중치를 곱해 세는 것이 그 사이다.
        """
        self._set_joints(q)
        err = np.linalg.norm(self._ee_positions() - targets, axis=1)
        return {
            f: float(max(err[2 * i] * self.dip_weight, err[2 * i + 1]))
            for i, f in enumerate(FINGERS)
        }

    def restart_seeds(self, world: np.ndarray | None = None) -> list:
        """국소최소에서 빠져나올 재시도 시드들. 앞쪽일수록 먼저 쓴다.

        사람 각도 시드는 검지/중지/약지에는 잘 듣지만 엄지에는 통하지 않는다.
        LEAP 엄지는 축 배치가 사람과 달라(th_axl 은 회전축이 손끝을 거의 지난다)
        각도를 옮겨 봐야 엉뚱한 골짜기다. 실제로 실패의 2/3 이 엄지였다.

        그래서 굽힘량을 달리한 자세 몇 개를 추가로 둔다. 엄지는 th_axl 값까지
        바꿔 가며 시도한다.
        """
        seeds = []
        if world is not None:
            seeds.append(self.seed_from_human(world))
        for curl, axial in ((0.9, 1.2), (0.4, 0.0), (1.4, 1.2), (0.9, 0.0)):
            q = np.zeros(jm.NUM_JOINTS)
            for f, j in FINGER_JOINTS.items():
                q[j[0]] = q[j[2]] = q[j[3]] = curl
                if f == "thumb":
                    q[j[1]] = axial
            seeds.append(jm.clip_mujoco(q))
        return seeds

    def solve_ik(
        self,
        targets: np.ndarray,
        seed: np.ndarray | None = None,
        alt_seeds=(),
    ) -> np.ndarray:
        """목표점 8개 -> 16 관절각. 실패한 손가락만 다른 시드로 다시 푼다.

        손가락끼리 자유도가 겹치지 않으므로 한 손가락의 재시도가 다른 손가락의
        해를 건드리지 않는다. 그래서 전체를 다시 풀고 실패한 손가락 블록만 가져와도
        된다(손가락 단위로 따로 푸는 것과 수학적으로 같다).

        재시도는 임계를 넘은 손가락이 남아 있는 동안만 돈다. 텔레오퍼레이션에서는
        직전 프레임 해가 시드라 대부분 첫 판에 끝나고, 재시도 비용은 거의 안 든다.
        """
        q = self._solve_dls(targets, self._q if seed is None else seed)
        residual = self.finger_residual(targets, q)
        self.last_restarts = 0

        for alt in alt_seeds:
            failed = [f for f, r in residual.items() if r > self.restart_threshold]
            if not failed:
                break
            self.last_restarts += 1
            q_alt = self._solve_dls(targets, alt)
            residual_alt = self.finger_residual(targets, q_alt)
            for f in failed:
                if residual_alt[f] < residual[f]:
                    q[FINGER_JOINTS[f]] = q_alt[FINGER_JOINTS[f]]
                    residual[f] = residual_alt[f]
        return q

    def _solve_dls(self, targets: np.ndarray, seed: np.ndarray | None = None) -> np.ndarray:
        """목표점 8개 -> 16 관절각(MuJoCo 순서). 감쇠최소자승(DLS) 반복.

        PyBullet 내장 calculateInverseKinematics2 를 쓰지 않는 이유
        ---------------------------------------------------------
        - 널스페이스 인자(lowerLimits/upperLimits/...)를 주면 해가 시드와 무관하게
          고정되어 손끝 잔차가 150mm 대로 망가진다.
        - 인자를 빼면 동작은 하지만 한 번 호출로 10mm, 30번 반복해도 1.25mm 에
          머문다(수렴이 선형).
        - IK_SDLS 는 이 모델에서 세그폴트한다.

        직접 푸는 편이 빠르고, 관절 범위를 매 반복 강제할 수 있어 결과도 안전하다.
        목표가 24차원, 자유도가 16이라 과결정계이므로 16x16 정규방정식으로 푼다.

            dq = (JᵀJ + λ²I)⁻¹ Jᵀ e
        """
        q = (self._q if seed is None else np.asarray(seed, dtype=float)).copy()
        lo = jm.LIMITS_INTERSECTION_MJ_LOWER
        hi = jm.LIMITS_INTERSECTION_MJ_UPPER
        eye = np.eye(jm.NUM_JOINTS)

        w = self.target_weights
        for _ in range(self.ik_iterations):
            self._set_joints(q)
            error = (targets - self._ee_positions()).ravel() * w
            if np.abs(error).max() < self.ik_tolerance:
                break
            J = self._jacobian(q) * w[:, None]
            dq = np.linalg.solve(J.T @ J + (self.ik_damping ** 2) * eye, J.T @ error)
            dq = np.clip(dq, -self.ik_max_step, self.ik_max_step)
            new_q = np.clip(q + dq, lo, hi)
            if np.abs(new_q - q).max() < 1e-9:
                break   # 관절 범위에 걸려 더 못 간다
            q = new_q

        return q

    def retarget(self, world: np.ndarray, dt: float = 0.02) -> np.ndarray:
        """21 랜드마크 -> 16 관절각(MuJoCo 순서). 평활화와 속도 제한까지 적용."""
        targets = self.compute_targets(world)
        raw = jm.clip_mujoco(
            self.solve_ik(targets, alt_seeds=self.restart_seeds(world))
        )

        # 지수 평활. MediaPipe 랜드마크는 프레임마다 몇 mm 씩 떨린다.
        a = np.clip(self.smoothing, 0.0, 1.0)
        q = a * raw + (1.0 - a) * self._q

        # 속도 제한. 검출이 한 프레임 튀어도 실기가 급발진하지 않도록.
        step = np.clip(q - self._q, -self.max_speed * dt, self.max_speed * dt)
        self._q = jm.clip_mujoco(self._q + step)

        self._seed(self._q)
        self._last_targets = targets
        self._update_gui(self._q, targets)
        return self._q.copy()

    def tip_error(self) -> np.ndarray:
        """직전 프레임의 목표점 대비 실제 손끝 오차(m), 목표 8개 각각."""
        if self._last_targets is None:
            return np.zeros(len(FINGERS) * 2)
        return np.linalg.norm(self._ee_positions() - self._last_targets, axis=1)

    def set_pose(self, q) -> None:
        """현재 자세를 바깥에서 지정한다. 다음 프레임 IK 의 시드가 된다."""
        self._q = jm.clip_mujoco(q)
        self._seed(self._q)

    def reset(self) -> None:
        """영점으로 되돌린다."""
        self.set_pose(np.zeros(jm.NUM_JOINTS))

    # ----------------------------------------------------------------- 디버그

    def _create_debug_balls(self) -> None:
        p = self.p
        colors = [(1, 0.3, 0.3, 1), (1, 0, 0, 1),
                  (0.3, 1, 0.3, 1), (0, 1, 0, 1),
                  (0.3, 0.3, 1, 1), (0, 0, 1, 1),
                  (1, 1, 0.3, 1), (1, 1, 0, 1)]
        for color in colors:
            vis = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.006, rgbaColor=color,
                physicsClientId=self.gui_client,
            )
            self._debug_balls.append(
                p.createMultiBody(
                    baseMass=0, baseVisualShapeIndex=vis, basePosition=[0, 0, 0],
                    physicsClientId=self.gui_client,
                )
            )

    def _update_gui(self, q: np.ndarray, targets: np.ndarray) -> None:
        """표시용 클라이언트에 최종 자세와 목표점만 반영한다.

        계산용 클라이언트와 분리되어 있으므로 IK 반복 중간 자세는 보이지 않는다.
        여기 보이는 것이 곧 MuJoCo/실기로 나가는 명령이다.
        """
        if self.gui_client is None:
            return
        for i, joint in enumerate(self.dof_indices):
            self.p.resetJointState(
                self.gui_uid, joint, float(q[i]), physicsClientId=self.gui_client
            )
        self._update_debug_balls(targets)

    def _update_debug_balls(self, targets: np.ndarray) -> None:
        for ball, target in zip(self._debug_balls, targets):
            self.p.resetBasePositionAndOrientation(
                ball, target.tolist(), [0, 0, 0, 1], physicsClientId=self.gui_client
            )

    def close(self) -> None:
        for client in (self.client, self.gui_client):
            if client is None:
                continue
            try:
                self.p.disconnect(physicsClientId=client)
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
