# 가이드 — 스택을 이해하고 시험하기

이 스택이 **무엇으로 이루어져 있고, 왜 그렇게 되어 있고, 어떻게 확인하는지**를 파이프라인 순서대로 설명한다.
각 문서는 "이 문서가 답하는 것 → 배경 → 실행(실제 출력 포함) → 결과 읽는 법" 순이고, 실행은 저장소의 스크립트를 그대로 쓴다.
실기가 없어도 🤖 표시가 없는 문서는 시뮬만으로 끝까지 확인할 수 있다. 환경은 `../setup.md`.

| 순서 | 문서 | 답하는 것 | 확인에 쓰는 것 | 실기 |
|---|---|---|---|---|
| 1 | [hand_anatomy.md](hand_anatomy.md) | 모터 16개의 위치·이름·역할, 두 이름 체계, 순서/단위/0점 규약 | `python -m leap_hand_mapping.joint_map` | |
| 2 | [joint_mapping.md](joint_mapping.md) | 시뮬 모델과 실기의 관절 순서·0점 불일치를 순기구학으로 검증 | `p0_1_verify_mapping_fk.py` | |
| 3 | [real_hand_safety.md](real_hand_safety.md) | Lite 전류 한계, 켜는 순서, 통신 오류율, 안전 규칙이 코드 어디에 있나 | `p0_2`, `p0_3`, `p0_4` | 🤖 |
| 4 | [hand_tracking.md](hand_tracking.md) | MediaPipe 가 주는 것/안 주는 것, 손바닥 좌표계, 거울상, 손 위치 틀 | `tracker_node` | |
| 5 | [retargeting.md](retargeting.md) | 왜 손끝 IK 인가, 스케일·손끝점·널스페이스·재시도, 옵션의 효과 | `retarget_node` 옵션 | |
| 6 | [ros2_integration.md](ros2_integration.md) | 노드 그래프, QoS, `header.stamp` 전파, 업스트림 복사+패치 | `sim.launch.py`, `ros2 topic hz` | |
| 7 | [measurement.md](measurement.md) | 종단 지연·추종 오차·계단 응답·떨림 분해의 정의와 실측 | `p1_4`, `p1_5` | (🤖) |
| 8 | [debugging_casebook.md](debugging_casebook.md) | 개발 중 겪은 문제 13건: 증상 → 원인 → 수정 → 재측정 | 재현 명령 | (🤖) |

설계 배경(왜 이렇게 만들었나)은 `../design/`, 개발 기록은 `../history/`.
