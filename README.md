# LEAP Hand v1 Lite — 웹캠 텔레오퍼레이션과 디지털 트윈

웹캠 앞에서 손을 움직이면 **MuJoCo 속 LEAP Hand 와 실물 LEAP Hand(Lite)가 같이 따라 움직이는** 오픈소스 스택이다.
MediaPipe 손 추적 → 손끝 위치 IK 리타겟팅 → ROS2 → MuJoCo 트윈 + 실기. 카메라 촬영에서 관절 명령까지 32 ms, 시뮬–실기 관절 오차 RMS 1.95°.

```
웹캠 → tracker_node → /hand/landmarks → retarget_node → /leap/joint_cmd ─┬─▶ sim_node (MuJoCo 트윈)
                                                                        └─▶ hand_bridge_node → 실기 (LEAP_Hand_API)
```

- **재현 가능**: 모든 수치는 저장소의 스크립트(`scripts/`)로 다시 잴 수 있다. 참조 저장소는 커밋 고정.
- **안전 우선**: 실기로 나가는 길은 노드 하나(`hand_bridge_node`)뿐 — 데드맨, 시작 자세 유지, 합류 램프, 전류 동결, 관절 범위 클립.
- **실기 없이도** 시뮬 트윈과 가짜 실기(`fake:=true`)로 전 경로를 돌려 볼 수 있다.

## 빠른 시작

```bash
# 환경: docs/setup.md (conda leap-hand, ROS2 Humble, 참조 저장소 클론) — 한 번
conda activate leap-hand && source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 launch leap_teleop sim.launch.py                    # 웹캠 → MuJoCo 손
ros2 launch leap_teleop real.launch.py fake:=true        # + 가짜 실기 (배선·안전 로직 시험)
ros2 launch leap_teleop real.launch.py                   # + 실기 (카메라 창 SPACE = 데드맨)
python scripts/phase1/p1_4_teleop_metrics.py --seconds 20   # Hz · 종단 지연 · 추종 오차 · 떨림
```

## 문서 지도

| 읽을 것 | 언제 |
|---|---|
| **[docs/setup.md](docs/setup.md)** | 처음 설치할 때. 두 환경, 버전 고정 이유, 확인 절차, 문제 해결 |
| **[docs/guide/](docs/guide/README.md)** | 스택을 이해하고 시험할 때. 로봇 손 구조 → 관절 매핑 → 실기 안전 → 손 추적 → 리타겟팅 → ROS2 → 측정 → 디버깅 사례집 |
| [docs/teleop_howto.md](docs/teleop_howto.md) | 텔레오퍼레이션을 돌리는 절차 (단일 스크립트 / ROS2) |
| [docs/real_hand_bringup.md](docs/real_hand_bringup.md) | 실기를 처음 켤 때 (전원 → 권한 → 지연시간 → 사전 점검 → 관절 하나씩) |
| [docs/design/](docs/design/) | 왜 이렇게 만들었나: [관절 매핑 검증](docs/design/joint_mapping.md) · [리타겟팅 파이프라인](docs/design/teleop_pipeline.md) · [ROS2 트윈과 측정](docs/design/ros2_twin.md) |
| [docs/history/](docs/history/) | 개발 기록(계획서·회고). 그때 기준의 문서 — 절차는 위 문서를 따를 것 |
| [NOTICE.md](NOTICE.md) | 서드파티 라이선스. **LEAP_Hand_API 는 CC BY-NC 4.0** |

## 저장소 구조

```
leap_hand_mapping/        코어 파이썬 패키지 (pip install -e .)
  joint_map.py            MuJoCo ↔ 실기 모터 매핑, 관절 범위, 클립     hand_tracker.py  웹캠 → MediaPipe 21점
  retarget.py             21점 → 16 관절각 (손끝 위치 IK)               real_hand.py     실기 드라이버 래퍼 (전류 350 mA 고정)
ros2_ws/src/leap_teleop/  ROS2 노드 5개 + 런치 2개 (sim.launch.py, real.launch.py)
ros2_ws/setup_upstream.sh 업스트림 ROS2 노드(LEAP_Hand_API) 복사 + patches/ 적용   ← 저장소에 포함하지 않음
scripts/phase0/           관절 매핑 FK 검증, 실기 사전 점검, 관절 순차 구동, 읽기 오류율
scripts/phase1/           손 추적 확인, 리타겟 왕복 시험, 단일 스크립트 텔레옵, 지표(p1_4), 계단 응답(p1_5)
data/                     매핑 JSON, 사람 손 녹화 npz        models/   MediaPipe 모델(내려받음)
third_party/              참조 저장소 클론 (gitignore)        docs/     문서
```

## 주요 수치 (실측, 재현 스크립트 표기)

| 항목 | 값 | 스크립트 |
|---|---|---|
| 관절 매핑 검증 (FK, 200 자세) | 채택안 std 0.06 mm vs 차점 25.6 mm | `p0_1_verify_mapping_fk.py` |
| 실기 계단 응답 (16관절, 20°) | 지연 83 ms · 상승 68 ms · 정상오차 0.81° | `p1_5_step_response.py --source real` |
| 라이브 종단 지연 (촬영→관절 명령) | 32 ms (추적 21.5 + 리타겟·전송 10.6) | `p1_4_teleop_metrics.py` |
| 시뮬–실기 추종 오차 | RMS 1.95° | 〃 |
| 정지 떨림 (명령 / 실기) | 0.75° / 0.96° (엄지 `th_cmc` 3° / 4.8°) | 〃 |
| 실기 읽기 오류율 (4 Mbps) | ~3% CRC, 무해 | `p0_4_read_reliability.py` |

## 하드웨어

LEAP Hand v1 **Lite**(Dynamixel XL330-M288-T ×16, 플라스틱 기어), U2D2, 5V 30A. Lite 는 전류 한계 **350 mA** 를 넘기면 기어가 상한다 —
코드에 고정돼 있고 올리지 않는다. 손목 회전 자세 추적은 범위 밖(단안 깊이 한계).

## 라이선스

이 저장소 코드는 [MIT](LICENSE). 참조하는 외부 소프트웨어와 조건은 [NOTICE.md](NOTICE.md) — 특히 LEAP_Hand_API / Bidex_VisionPro_Teleop 는 **CC BY-NC 4.0(비상업)** 이며 저장소에 포함하지 않고 이용자가 직접 받는다.
