# 8단원 — 디버깅 사례집: 증상 → 가설 → 측정 → 수정 → 재측정

> 원문: `docs/phase1_retrospective.md` 3장(15건), `README.md` "떨림"·"벌림 관절 범위 제한", 커밋 로그

이 단원은 교과서에 없는 것이다. 실제로 겪은 문제를 **먼저 증상만** 주고, 원인을 추리한 뒤 해설과 대조한다.
사례마다 재현 방법이 있으니 시뮬/가짜 실기로 직접 겪어 보라. 🤖 는 실기에서만 재현된다.

## 방법 — 이 저장소가 쓴 디버깅 규칙

1. **증상을 숫자로 바꾼다** (7단원의 지표). "떤다"가 아니라 "명령 0.75°, 실기 4.8°(th_cmc)".
2. **원인 후보를 단계별로 나열**한다: 센서(랜드마크) → 리타겟 → 전송/큐 → 브리지 → 모터.
3. **한 단계씩 끊어서 잰다**: 손을 빼면(명령 고정) 모터만 남고, 카메라를 끄면(계단) 리타겟이 빠진다.
4. 고친 뒤 **같은 지표를 다시** 잰다. 고쳤다는 느낌은 증거가 아니다.
5. 원인이 겹치면 하나씩 바꿔서는 안 보인다 — 조합으로 멎게 한 뒤 하나씩 빼서 기여도를 가른다(사례 8).

---

## 사례 1 — 노드가 `mediapipe` 를 못 찾는다

**증상** `ros2 run leap_teleop tracker_node` → `ModuleNotFoundError: mediapipe`. 같은 터미널에서 `python -c "import mediapipe"` 는 된다.
**재현** `/usr/bin/colcon build` 로 빌드.
<details><summary>해설</summary>
설치된 실행 파일의 셔뱅이 `#!/usr/bin/python3`(시스템 파이썬). `ros2 run` 은 그 셔뱅을 따른다. conda 환경에 colcon 을 설치해
다시 빌드하면 셔뱅이 conda 파이썬이 된다. 교훈: "어느 파이썬이 실행되나"를 `head -1 install/.../tracker_node` 로 확인.
</details>

## 사례 2 — `rclpy._rclpy_pybind11 ... cpython-313`

**증상** 어떤 ros2 명령도 pybind 오류.
<details><summary>해설</summary>
conda base(py3.13)가 PATH 앞. Humble 은 3.10. `conda activate leap-hand`.
</details>

## 사례 3 🤖 — `Incorrect status packet` 이 초당 한두 번

**증상** 실기 런치 로그에 `ERROR:root:> read: [TxRxResult] Incorrect status packet!` 반복. 동작은 멀쩡해 보인다.
**가설을 셋 적어 보라.** 그리고 어떻게 가르겠는가?
<details><summary>해설</summary>
`p0_4_read_reliability.py` 로 읽기 방식(fast/sync) × 주기(30/15 Hz) 격자 측정 → 전부 ~3% → 폴링 부하·프로토콜 배제,
**배선/4 Mbps CRC**. `latency_timer 1`, `Return Delay 0` 도 무관. 업스트림 리더가 실패 시 직전 값을 돌려주므로 무해. 그대로 두고
진행했다. 교훈: 고치기 전에 "무해한가"를 먼저 판정하면 시간을 아낀다.
</details>

## 사례 4 🤖 — 런치하자마자 실기가 **확** 움직인다 (1)

**증상** 카메라를 켜기도 전에, 노드가 뜨는 순간 손가락이 튄다.
<details><summary>해설</summary>
업스트림 `leaphand_node` 가 토크를 켜며 목표를 편 손(π)으로 둔다 — 현재 자세가 무엇이든 거기로 스냅. 패치 `hold_on_start`:
토크 켜기 전에 현재 자세를 읽어 그 자세를 목표로 둔다.
</details>

## 사례 5 — 데드맨을 켜는 순간 **확** 움직인다 (2)

**증상** 사례 4 를 고친 뒤에도, 카메라가 이미 다른 자세를 보고 있으면 ON 순간 8 rad/s 로 점프.
**재현** `real.launch.py fake:=true`, 손을 주먹 쥔 채 `/teleop/enable true`.
<details><summary>해설</summary>
브리지가 ON 되면 먼저 실기 자세를 읽고 거기서 1 rad/s 로 **합류**한 뒤 8 rad/s 로. 뒤에 학습 정책(목표가 20 Hz 로 계속 움직임)을
붙였을 때 "합류 완료" 조건(모든 관절이 tol 안)이 영영 안 맞아 1 rad/s 에 갇혔다 → `engage_timeout` 3 s 추가. 교훈: 상태 전이의
**탈출 조건**을 항상 같이 설계.
</details>

## 사례 6 — 전류 초과 경고가 30 ms 마다 뜬다

**증상** `전류 초과 [('th_cmc', 322)]` 가 움직일 때마다 뜨고, 동결/해제가 반복돼 움직임이 더듬거린다.
**재현** `fake:=true`, `ros2 param set /fake_hand_node fake_current 320.0`.
<details><summary>해설</summary>
움직이는 순간의 300~380 은 스톨이 아니라 **가속 전류**다. 스톨은 한계(~469 = 350 × 1.34)에 **계속** 붙어 있다. 임계 400,
3표본(100 ms) 연속일 때만 얼린다. 교훈: 한 표본으로 판단하지 말 것; 위험한 건 지속 상태다.
</details>

## 사례 7 🤖 — 로봇이 떨린다 (핵심 사례)

**증상** 정지 상태에서 실기가 눈에 띄게 떤다. 시뮬도 조금 떤다.
**측정** `p1_4` 떨림 분해. **손을 카메라 밖으로 빼서** 명령을 고정하는 검사.
**후보** 랜드마크 잡음 / IK 재시도 / 널스페이스 / 모터 게인 — 넷 다 가능.
<details><summary>해설</summary>
인자를 하나씩 넣어 봤다: `kP 400`, `smoothing 0.2`, `deadband 1°`, `restart_mm 50`, `pip_target`, `tip_mode axis` — **하나씩은
"아직 떨려"**, 전부 넣으니 "잘되네". 원인이 넷 다 실재했고 하나를 막아도 나머지가 보였다. 조합을 기본값으로 올리고
(`27c8833`), 기여도 분해는 남은 일로 기록. 교훈: "한 번에 하나씩"은 원인이 하나일 때의 규칙이다.
</details>

## 사례 8 — `restart_mm:=15` 를 주니 시뮬이 안 움직인다

**증상** 런치는 떴는데 MuJoCo 손이 가만히 있다. 로그를 보면 `retarget_node` 가 죽어 있다.
**재현** 지금은 고쳐졌으니 `git show 97bb7eb` 로 전후를 보라.
<details><summary>해설</summary>
`15` 는 정수 → INTEGER 파라미터 → 노드가 DOUBLE 을 기대해 `InvalidParameterTypeException`. `ParameterValue(..., value_type=float)`.
그다음엔 같은 걸 `IncludeLaunchDescription` 의 `launch_arguments` 에도 넣어 `'ParameterValue' object is not iterable`. 교훈:
런치 인자는 **문자열**이고, 타입은 Node 파라미터에서만 고정한다.
</details>

## 사례 9 — 손가락 끝점과 로봇 끝점이 다르다

**증상** 손을 폈는데 로봇 DIP 가 과굽힘, 검지가 특히 이상.
<details><summary>해설</summary>
세 원인: (a) 로봇 손끝점 `realtip` 이 손가락 축에서 20° 벗어난 패드 접촉점(74 mm), (b) 2점 목표의 PIP 널스페이스,
(c) MediaPipe 검지 말단 편향. 옵션 `tip_mode axis`(축 위 점), `pip_target`(PIP 점 추가). 교훈: "IK 가 틀렸다"는 대개 "목표가 틀렸다".
</details>

## 사례 10 🤖 — 전류 동결이 한번 걸리면 안 풀린다

**증상** `rf_rot 468 → 동결`. 손을 빼서 명령이 영점으로 갔는데도 `FROZEN 445 → 425` 로 영영 유지.
**재현** `fake:=true`, `fake_current 450` 으로 얼린 뒤 `0` 으로 내려 보라 — 고친 버전은 풀리고 재합류한다.
<details><summary>해설</summary>
동결이 "명령을 멈추는 것"뿐이라 **막히던 목표가 모터에 남아** 계속 밀고 → 전류가 한계에 붙은 채 해제 임계 300 밑으로 못 내려옴.
얼리는 순간 실기 현재 자세를 한 번 명령해 힘을 뺀다. 교훈: 안전 상태로 들어가는 길만 설계하고 **나오는 길**을 안 보면 교착.
</details>

## 사례 11 🤖 — `rf_rot` 이 옆 손가락에 걸린다

**증상** 손가락을 모을 때 약지 벌림 모터가 전류 한계.
<details><summary>해설</summary>
모델 한계(±60°) 안이라도 실물은 서로 닿는다. `clip_mujoco` 의 텔레옵 표를 좁혔다: mf ±3°, if/rf 중지 쪽 3°(바깥 유지).
**부호는 MJCF 로 확인**(if_rot + 가 중지 쪽, rf_rot − 가 중지 쪽) — 추측하면 반대로 막는다. 나중에 학습 정책은 벌림을 ±20° 넘게
쓰므로 표를 **선택**할 수 있게 했다(`limits teleop|model`). 교훈: 안전 제한은 경로별로 다를 수 있다.
</details>

## 사례 12 — 카메라를 못 연다

**증상** `can't open camera by index`. 방금 전까지 됐다.
<details><summary>해설</summary>
이전 시험 런치의 프로세스가 죽지 않고 `/dev/video0` 을 쥐고 있었다. `fuser /dev/video0`, `pkill -9 -f "[l]ib/leap_teleop/"`.
(`pkill -f "leap_teleop/"` 는 자기 셸까지 죽인다 — 대괄호 트릭.) 교훈: 시험을 띄운 사람이 치운다.
</details>

## 사례 13 — 종료할 때 트레이스백이 쏟아진다

**증상** Ctrl-C 에 `RCLError`, `GLXBadContext`.
<details><summary>해설</summary>
launch 의 SIGINT 와 컨텍스트 종료의 경합. 타이머 콜백에 `if not rclpy.ok(): return`. GLX 는 MuJoCo 뷰어 종료 잡음, 무시.
</details>

## 종합 문제

다음 로그만 보고 원인 단계와 다음 한 수를 적어라.

```
[retarget_node]: 명령 604  리타겟 10.0 ms  촬영->명령 지연 32.1 ms  손끝잔차 39.6 mm  데드밴드 유지 252/604  재시도 0
[hand_bridge_node]: 데드맨 FROZEN  명령 1627  동결 1회  전류초과 [('rf_rot', 445)]
[retarget_node]: 손 유실 1.5s — 1.0s 에 걸쳐 영점으로
[hand_bridge_node]: 데드맨 FROZEN  명령 1627  동결 1회  전류초과 [('rf_rot', 425)]
```

<details><summary>해설</summary>
명령 카운트가 1627 에서 멈춘 채 FROZEN 지속 + 손 유실로 명령은 영점인데 전류가 안 내려옴 → 사례 10(동결 교착). 그 원인 자세는
`rf_rot` 이 옆 손가락에 닿은 것 → 사례 11. 손끝잔차 39.6 mm 는 엄지 쪽 목표가 작업공간 밖인 정상 범위(5단원)라 여기선 무관.
다음 한 수: 동결 시 힘 빼기(코드), 벌림 범위 제한(표), 그리고 **같은 로그를 다시** 받아 `동결 0회` 를 확인.
</details>

## 이 과정을 마치며

Phase 0/1 에서 배운 것을 한 줄로 줄이면: **로봇이 이상하게 움직일 때, 느낌으로 고치지 말고 단계를 끊어 재라.** 그러면 원인은
언제나 하나의 숫자 뒤에 있다 — 그리고 가끔은 넷이 겹쳐 있다.
