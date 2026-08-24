# 서드파티 소프트웨어와 라이선스

이 저장소의 코드는 MIT (`LICENSE`). 아래는 이 저장소가 **참조·복사·내려받는** 외부 소프트웨어다.
`third_party/` 는 git 에 포함하지 않고 `docs/setup.md` 절차로 각자 클론한다 (커밋 고정).

| 이름 | 라이선스 | 이 저장소에서 쓰는 방식 | 주의 |
|---|---|---|---|
| [LEAP_Hand_API](https://github.com/leap-hand/LEAP_Hand_API) (b0d00c8) | **CC BY-NC 4.0** | 실기 드라이버(`python/`) import, ROS2 노드(`ros2_module`)를 `ros2_ws/setup_upstream.sh` 가 복사 + `patches/leap_hand_port_param.patch` 적용. **저장소에 포함하지 않음** | **비상업 조건.** 상업적 이용(유료 교육 포함 여부는 이용자가 판단)은 원저작자 조건을 따를 것 |
| [Bidex_VisionPro_Teleop](https://github.com/leap-hand/Bidex_VisionPro_Teleop) (4914349) | **CC BY-NC 4.0** | 공식 URDF(`leap_hand_mesh_right/robot_pybullet.urdf`) 를 PyBullet 로 읽어 IK·FK 검증. 저장소에 포함하지 않음 | 위와 같음 |
| [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (da76818) | Apache 2.0 (모델별 LICENSE 참조; `leap_hand` 는 MIT/CC 계열 CAD) | MuJoCo LEAP 모델 로드. 저장소에 포함하지 않음 | |
| [dex-urdf](https://github.com/dexsuite/dex-urdf) (f5e7132) | MIT | menagerie 모델의 출처 확인용(코드에서 쓰지 않음) | |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) Hand Landmarker | Apache 2.0 | pip 패키지 + 모델 파일(`hand_landmarker.task`)을 `scripts/phase1/p1_0_fetch_mediapipe_model.sh` 가 내려받음 | |
| [MuJoCo](https://github.com/google-deepmind/mujoco), [PyBullet](https://pybullet.org), [OpenCV](https://opencv.org), [NumPy](https://numpy.org), [Dynamixel SDK](https://github.com/ROBOTIS-GIT/DynamixelSDK) | Apache 2.0 / zlib / Apache 2.0 / BSD / Apache 2.0 | pip 의존성 | |
| ROS 2 Humble | Apache 2.0 | 시스템 설치 | |

`patches/leap_hand_port_param.patch` 는 CC BY-NC 4.0 저작물(LEAP_Hand_API `ros2_module`)의 **수정 diff** 이므로 원저작물과 같은 조건이 적용될 수 있다. 이 저장소의 나머지 코드(`leap_hand_mapping/`, `ros2_ws/src/leap_teleop/`, `scripts/`)는 위 저작물을 **import 하거나 별도 프로세스로 호출**할 뿐 복사하지 않는다.

LEAP Hand 하드웨어(CAD)는 [leap-hand.github.io](https://leap-hand.github.io) 의 CC BY-NC-SA 조건을 따른다.
