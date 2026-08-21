# 2026-08-21 엄지 작업 보관본

main 은 geon 의 결정으로 **보정(엄지 정렬 캘리브레이션) 이전인 커밋 7cccfdd** 로
되돌렸다. "보정할수록 이상해진다" — 그래서 보정이 없던 마지막 상태에서 다시 시작한다.

이 폴더는 그 이후 작업의 **파일 복사본**이다. 같은 내용이 git 브랜치
`bak/2026-08-21-thumb-work` (커밋 d6e5509) 에도 있다. 되살리려면 브랜치 쪽을 쓴다:

    git diff main bak/2026-08-21-thumb-work --stat
    git checkout bak/2026-08-21-thumb-work -- <파일>

여기 있는 것
- leap_hand_mapping/retarget.py      엄지 결합(A2), 엄지 관절각 매핑(thumb_mode=map), 보정
- leap_hand_mapping/retarget_dex.py  dex-retargeting 어댑터
- leap_hand_mapping/joint_map.py     관절 한계를 공식값으로 통일 (LIMITS_MJ, apply_model_limits)
- scripts/phase1/p1_diag_*.py        진단 4종 + 녹화기(p1_diag_record_poses.py, 손 위치 틀)
- README.md / docs/                  실측 표와 결론 (자세 충실도, 엄지, 센서)

왜 되돌렸는지와 각 시도의 실측은 README.md 의 Phase 1 절에 있다. 요약:
- dex 는 PIP 를 안 쓴다(갈고리). ours 는 주먹 모양이 나온다
- 엄지 결합은 핀치 177 -> 54mm 까지만. 검지가 엄지 작업공간 밖이라 더는 안 닿는다
- 엄지 관절각 매핑은 편 손/붙임/주먹 모양을 잡지만 라이브에서 "보정할수록 이상"
- 손목 회전은 단안 깊이 한계라 범위 밖
