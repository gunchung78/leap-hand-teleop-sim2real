"""웹캠 -> MediaPipe Hands -> 21개 손 랜드마크.

Phase 1(텔레오퍼레이션)의 입력단. 이 모듈은 로봇을 전혀 모르고,
사람 손의 3D 랜드마크만 뱉는다. 로봇으로의 변환은 retarget.py 가 맡는다.

좌표계
------
MediaPipe 는 두 가지 랜드마크를 준다.

  image landmarks  화면 정규화 좌표 [0,1]. 그리기 전용.
  world landmarks  미터 단위 3D. 원점은 손의 기하 중심.
                   축은 이미지와 같은 방향(x 오른쪽, y 아래, z 카메라에서 멀어짐)
                   이므로 오른손 좌표계다.

리타겟팅에는 world landmarks 만 쓴다. image landmarks 는 사람 눈으로
추적이 되는지 확인하는 용도다.

거울 주의
--------
MediaPipe 의 handedness("Left"/"Right") 판정은 **입력 영상이 거울상이 아니라고**
가정한다. 셀피처럼 좌우를 뒤집어 넣으면 라벨이 반대로 나오고, 동시에 world
랜드마크의 손대칭(chirality)도 뒤집힌다. 뒤집힌 좌표를 그대로 리타겟팅하면
로봇 손이 손등 쪽으로 굽는 거울상이 된다.

그래서 이 모듈은 **handedness 라벨로 손을 고른다.** 오른손을 들었는데 "Right"
로 안 잡히면 영상이 뒤집힌 것이므로, 그때 mirror 를 켜면 라벨과 좌표가 동시에
제자리로 돌아온다. 라벨 필터가 곧 좌표계 정합 검사 역할을 한다.

모델 파일
--------
mediapipe 1.x 는 legacy `mp.solutions.hands` 를 없앴고 tasks API 만 남았다.
tasks API 는 모델을 패키지에 넣어 두지 않으므로 hand_landmarker.task 를
따로 받아야 한다(약 7.5MB).

    bash scripts/fetch_mediapipe_model.sh
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(REPO, "models", "hand_landmarker.task")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# MediaPipe Hands 21 랜드마크 인덱스.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

NUM_LANDMARKS = 21

# 손 골격 (그리기용).
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@dataclass
class HandObservation:
    """한 프레임에서 검출된 손 하나."""

    world: np.ndarray      # (21, 3) 미터
    image: np.ndarray      # (21, 3) 화면 정규화 좌표
    handedness: str        # "Left" / "Right"
    score: float

    def palm_width(self) -> float:
        """검지 MCP <-> 약지 MCP 거리(m). 스케일 기준으로 쓴다."""
        return float(np.linalg.norm(self.world[RING_MCP] - self.world[INDEX_MCP]))

    def hand_length(self) -> float:
        """손목 <-> 중지 MCP 거리(m). 검출 품질을 눈으로 볼 때 쓴다."""
        return float(np.linalg.norm(self.world[MIDDLE_MCP] - self.world[WRIST]))


class HandTracker:
    """웹캠 프레임을 넣으면 HandObservation 을 준다.

    MediaPipe tasks API 의 VIDEO 모드를 쓴다. VIDEO 모드는 타임스탬프가
    단조증가해야 하므로 프레임 번호를 직접 세서 넣는다.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        handedness: str = "Right",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        mirror: bool = False,
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe 모델이 없다: {model_path}\n"
                f"  bash scripts/fetch_mediapipe_model.sh   (또는 {MODEL_URL} 를 직접 받을 것)"
            )

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.handedness = handedness
        self.mirror = mirror
        self._frame_index = 0

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,   # 양손을 다 잡은 뒤 handedness 로 고른다
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """거울 설정을 적용한 프레임. 화면에 그릴 때도 이걸 써야 좌표가 맞는다."""
        return frame_bgr[:, ::-1] if self.mirror else frame_bgr

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None):
        """BGR 프레임 하나 -> HandObservation 또는 None.

        frame_bgr 은 preprocess() 를 이미 거친 프레임이어야 한다.
        """
        import mediapipe as mp

        if timestamp_ms is None:
            self._frame_index += 1
            timestamp_ms = self._frame_index * 33

        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))

        if not result.hand_world_landmarks:
            return None

        for i, category in enumerate(result.handedness):
            label = category[0].category_name
            if label != self.handedness:
                continue
            world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[i]])
            image = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[i]])
            return HandObservation(
                world=world, image=image, handedness=label, score=float(category[0].score)
            )
        return None

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw_landmarks(frame_bgr: np.ndarray, obs: HandObservation) -> np.ndarray:
    """검출 결과를 프레임 위에 그린다. 추적이 되는지 눈으로 보는 용도."""
    import cv2

    h, w = frame_bgr.shape[:2]
    pts = [(int(p[0] * w), int(p[1] * h)) for p in obs.image]
    for a, b in CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], (0, 200, 0), 2)
    for i, pt in enumerate(pts):
        # 리타겟팅에 실제로 쓰는 8개 점만 크게 그린다.
        key = i in (THUMB_IP, THUMB_TIP, INDEX_DIP, INDEX_TIP,
                    MIDDLE_DIP, MIDDLE_TIP, RING_DIP, RING_TIP)
        cv2.circle(frame_bgr, pt, 6 if key else 3, (0, 0, 255) if key else (255, 160, 0), -1)
    return frame_bgr
