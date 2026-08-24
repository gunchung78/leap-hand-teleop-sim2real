# data/

| 파일 | 내용 | 만든 것 |
|---|---|---|
| `joint_mapping.json` | 관절 매핑 테이블(MuJoCo ↔ 모터 ID, 범위) 내보내기. 언어 무관 재사용용 | `python -m leap_hand_mapping.joint_map --json data/joint_mapping.json` |
| `thumb_capture.npz` | 사람 손 4자세 녹화(MediaPipe world 랜드마크, 360여 프레임). 리타겟 실측에 썼던 데이터 | 브랜치 `bak/2026-08-21-thumb-work` 의 `p1_diag_record_poses.py` |
