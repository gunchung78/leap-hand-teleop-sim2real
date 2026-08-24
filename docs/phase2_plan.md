# Phase 2 계획 — MJX 강화학습 (`LeapCubeRotateZAxis`) → 시뮬 → 실기

작성 2026-08-23. Phase 1 완료(`phase1_retrospective.md`) 직후. 원칙은 Phase 1 과 같다:
**업스트림 그대로 먼저, 바꾸는 건 인자로, 숫자로 확인한 뒤 기본값.**

---

## 1. 목표와 산출물

목표: 손바닥 위 큐브를 z축으로 계속 돌리는 정책을 GPU 시뮬(MJX)에서 학습해, Phase 1 의 같은 ROS2
경로(`/leap/joint_cmd` → 시뮬 트윈 / 실기)로 내려보낸다. 사람 손 대신 신경망이 명령을 낸다.

| 산출물 | 완료 기준 |
|---|---|
| 학습 환경 `leap-mjx` + 설치 스크립트 | `scripts/phase2/setup_mjx_env.sh` 한 번으로 재현, GPU 에서 env 1024개 스텝 |
| 학습 스크립트·설정·체크포인트·곡선 | 업스트림 `train_jax_ppo.py` 호출 래퍼, `logs/` 에 체크포인트 + TensorBoard |
| 시뮬 재생 | 학습 정책이 MuJoCo(CPU) 에서 큐브를 돌린다 — 영상 + 회전 속도·낙하 시간 수치 |
| `policy_node` (ROS2) | `/sim/joint_states` 또는 `/real/joint_states` → 정책 → `/leap/joint_cmd` 20 Hz |
| 실기 시도 | 데드맨·전류 동결 하에서 실물 큐브. 되면 영상, 안 되면 무엇이 막는지 수치 |

최소 성공선은 **시뮬 재생까지**. 실기는 되면 보너스로 본다(아래 7장 위험).

---

## 2. 업스트림 환경이 무엇인가 (`mujoco_playground` e74217b, `_src/manipulation/leap_hand/rotate_z.py`)

| 항목 | 값 | 비고 |
|---|---|---|
| 관절/액추에이터 | 16, 이름 = `MUJOCO_JOINT_NAMES` 와 동일 | Phase 0 매핑 그대로 |
| 제어 주기 | `ctrl_dt` 0.05 s (20 Hz), `sim_dt` 0.01 | 정책은 20 Hz 로 목표각을 낸다 |
| 행동 | 16차원, `action_scale` 0.6, 목표각 = 기본자세 + 0.6·a (위치 PD) | 브리지가 받는 형태와 같다 |
| **정책 입력** | **관절각 16개 + ±0.05 rad 잡음, 이력 1** | 큐브 상태 없음 → 실기에서 카메라 불필요. `/real/joint_states` 가 그대로 입력 |
| 비평가 입력 | 관절각 + 큐브 위치/자세/각속도/선속도 | 학습 때만 |
| 보상 | `angvel`(큐브 z 각속도) ×1.0, `termination` −100, 나머지(토크·에너지·자세·행동변화) 0 | **토크/에너지 벌점이 기본 0** — 실기 전류에 직접 닿는 지점 |
| 종료 | 큐브 z < −0.05 m (떨어짐) | |
| 에피소드 | 500 스텝 = 25 s | |
| 도메인 무작위화 | 손끝 마찰 U(0.5,1.0), 큐브 질량 ×U(0.8,1.2)·무게중심, qpos0 ±0.05, 정지 마찰 ×U(0.9,1.1) | `domain_randomize`, 학습 시 `--domain_randomization` |
| 큐브 | 7 cm 정육면체, 108 g | 실물도 이 근처로 |
| PPO (업스트림 기본) | 1e8 스텝, **8192 env**, unroll 40, minibatch 32, lr 3e-4, entropy 1e-2, 망 (512,256,128) | 6 GB GPU 에선 env 수를 줄인다 |

### 액추에이터 모델 ↔ 실기 (sim-to-real 의 핵심 대응)
`leap_rh_mjx.xml`:
```
<position kp="3.0"/>                        ← N·m/rad 위치 PD
<joint damping="0.2" armature="0.00149376"  ← 로터 관성 × 288²
       actuatorfrcrange="-0.2196 0.2196"    ← "max torque = 600/1000 * 0.366"  (600 mA × 0.366 N·m/A)
       frictionloss="0.02"/>
```
업스트림은 **전류 600 mA 의 풀 LEAP** 을 모델링했다. 우리는 **Lite, `curr_lim` 350 mA** → 같은 환산으로
**0.128 N·m.** 학습 정책이 0.2196 까지 쓰면 실기에선 전류 한계에 붙어 브리지가 얼린다.
그래서 학습에 두 단계를 둔다: **v0 업스트림 그대로**(비교 기준) → **v1 토크 상한 0.128 + 토크/에너지 벌점**.
`kp` 3.0 N·m/rad 과 Dynamixel `kP` 400 의 대응은 계단 응답으로 맞춘다(Phase 1 의 `p1_5`: 실기 상승 68 ms).

---

## 3. 왜 학습 환경을 따로 두나

`mujoco_playground` 최신판은 **Python ≥ 3.11**, ROS2 Humble 은 **3.10** 고정. pip 의 옛 `playground 0.1.0` 은
3.10 에 깔리지만 mujoco 3.11 과 안 맞는다(`mjx.make_data(nconmax)` 오류, 08-23 확인).

→ **`leap-mjx`(py3.11): 학습·시뮬 재생. `leap-hand`(py3.10): ROS2 추론.** 정책은 **파일**(파라미터 npz +
망 정의, 또는 onnx)로 넘긴다. 추론은 MLP 3층이라 numpy 로도 된다. 두 환경이 같은 `mujoco` 버전을 쓰도록
맞춘다(트윈과 학습 모델의 물리를 같게).

---

## 4. 우리 로봇에 맞출 것 (학습 전에)

| 항목 | 업스트림 | 우리 | 어떻게 |
|---|---|---|---|
| 관절 이름·순서 | 동일 | 동일 | 확인만 (Phase 0) |
| 관절 범위 | MJCF 기본 (rot ±60°) | 벌림 `LIMITS_TELEOP_MJ_*` (mf ±3°, if/rf 중지 쪽 3°) | **보류.** rotate_z 는 손가락을 벌려 큐브를 잡는다. 실기에서 걸리는 건 *빈손 + 텔레옵* 이었다. 먼저 v0 정책이 rot 을 얼마나 쓰는지 보고, 걸리면 그때 범위/벌점 |
| 토크 상한 | 0.2196 (600 mA) | 0.128 (350 mA) | v1: `jnt_actfrcrange` 를 모델 로드 후 덮어쓰기 (업스트림 코드 수정 없이 래퍼) |
| 토크/에너지 벌점 | 0 | v1 에서 켬 | `reward_config.scales.torques/energy` 인자 |
| 제어 주기 | 20 Hz | 브리지 60 Hz 램프, 실기 반응 ~80 ms | `policy_node` 20 Hz, 브리지 `max_speed` 는 그대로(램프가 저역통과 역할) |
| 큐브 | 7 cm / 108 g | 실물 준비 | 3D 프린트 또는 나무 블록, 무게 맞추기 |
| 손 자세 | 손바닥 위 (palm up) | 실기 거치대 방향 확인 | 현재 거치대가 palm-up 인지 확인 필요 (**geon**) |

---

## 5. 작업 순서

| 단계 | 내용 | 완료 기준 | 예상 |
|---|---|---|---|
| **S0** | `leap-mjx` 환경 + 설치 스크립트, GPU 스모크(1024 env 스텝 속도, 메모리) | 숫자 (env-steps/s, MiB) | 0.5 h |
| **S1** | 업스트림 그대로 **v0 짧은 학습**(예: 2e7 스텝, env 1024~2048) — 파이프라인·시간 가늠 | 보상 곡선 상승, 체크포인트, 시뮬 재생 영상 | 1~3 h (대부분 GPU 시간) |
| **S2** | 정책 내보내기 + `scripts/phase2/p2_1_play_policy.py` (CPU MuJoCo 재생, 회전 속도·낙하 시간 수치) | 표 | 2 h |
| **S3** | v0 **본 학습** (1e8 스텝 또는 곡선 포화까지) + DR | 수치·영상, README | 밤새 |
| **S4** | **v1** 토크 상한 0.128 + 벌점 → 같은 지표로 v0 와 비교 (전류가 얼마나 줄고 회전이 얼마나 느려지나) | 비교표 | 반나절 + 학습 |
| **S5** | `policy_node` + `policy.launch.py`: 트윈에서 먼저(`/sim/joint_states` 입력, 시뮬 큐브) | 트윈에서 큐브가 돈다 | 반나절 |
| **S6** | 실기: 데드맨 ON, 빈손 먼저(전류·떨림), 그다음 실물 큐브 | 영상 or 막힌 이유 수치 | 반나절 |
| **S7** | README "Phase 2", `docs/phase2_retrospective.md` | 커밋 | 2 h |

S1 이 중요하다 — 6 GB 에서 몇 env 가 들어가고 1e8 스텝이 몇 시간인지 **여기서 알게 된다.** 그 숫자로 S3 을 정한다.

---

## 6. 지표 (재현 스크립트와 같이 커밋)

- 학습: 보상 곡선(TensorBoard), 에피소드 길이, 스텝/초, GPU 메모리.
- 시뮬 재생(`p2_1`): 큐브 z 각속도 평균(rad/s), 떨어지기까지 시간, 관절 토크 RMS·최대(→ 예상 전류), rot 관절 사용 범위.
- 트윈/실기(`policy_node` + 기존 `p1_4` 확장): 명령 주기, 실기 전류 분포(>400 비율, 동결 횟수), 큐브 회전(카메라로 손 세기 — 처음엔 수동).

---

## 7. 위험과 미리 정한 대응

| 위험 | 대응 |
|---|---|
| GPU 6 GB 에 8192 env 안 들어감 | S0 에서 실측, 1024~2048 로. 스텝 수는 같게 두고 시간을 더 쓴다 |
| 학습 시간 (노트북 3060) | S1 짧게 돌려 외삽. 밤에 돌린다. 체크포인트로 재개 |
| Lite 기어: 정책이 공격적 | v1 토크 상한·벌점. 브리지 전류 동결(힘 빼기)이 마지막 방어. `curr_lim` 350 불변 |
| sim-to-real 격차 (마찰, 기어 백래시, 응답 80 ms) | DR 켜기, v1, 실기 빈손 시험 먼저. 안 되면 "시뮬 재생까지"로 선을 긋고 이유를 수치로 |
| 실물 큐브 | 7 cm/108 g 근처로. 마찰은 DR 범위(0.5~1.0) 안에 들게 테이프 등 |
| 업스트림 버전 드리프트 | playground 는 저장소 클론(커밋 고정) editable, 설치 스크립트 한 개 |

---

## 8. 진행 기록

**S0 (08-23)** `scripts/phase2/setup_mjx_env.sh` → `leap-mjx`(py3.11). 버전 고정 이유:
- playground 최신(e74217b) 은 py≥3.11, pip 0.1.0 은 mujoco 3.11 과 불일치 → 저장소 클론을 editable 로.
- brax 0.14.2 가 `jax.device_put_replicated` 를 써서 jax 0.10 에서 죽음 → **jax[cuda12]==0.7.2** 고정(클론 설치 뒤에).
- 영상 저장에 ffmpeg(conda-forge). tensorboard 로 곡선.
- `p2_0_mjx_smoke.py`: 6 GB 에 **8192 env 들어간다.** 순수 시뮬 20k env-steps/s(8192), 학습 포함 25k steps/s → 1e8 ≈ 70 min.

**S1 (08-23)** `p2_1_train.sh` (업스트림 `train_jax_ppo.py` 호출, 인자만). 2e7 스텝 v0-short: 보상 −0.41 → 5.5,
angvel 항 −0.5 → 113 (에피소드 합, 확률적 평가 → 평균 ≈0.23 rad/s), 에피소드 길이 ≈490/500. 파이프라인 OK.
업스트림 스크립트는 마지막 rollout mp4 저장에서 ffmpeg 없으면 죽는다(체크포인트는 이미 저장됨).

**S2 (08-23)** `p2_2_play_policy.py`: CPU MuJoCo 재생 + 지표. brax 0.14.2 `checkpoint.load_policy` 가 설정 JSON 의
`mean_kernel_init_fn: null` 에서 KeyError → 자체 로더(None 키 제거). 같은 정책을 MJX env 안에서 돌려 로더 검증:
확률적 행동 angvel 합 [159, 75, 32] (학습 평가 113 ± 큰 분산) — 일치. 결정적은 [29, 3, 63] → 20M 정책은 아직 약하다
(CPU 재생 0.02~0.06 rad/s, 25 s 동안 안 떨어뜨림). **알게 된 것 둘:**
- 정책이 벌림(rot) 관절을 `mf_rot` −20~+26°, `rf_rot` −13~+22° 로 쓴다 → 텔레옵 제한(±3°)을 정책 경로에 그대로 쓰면 정책이 깨진다. S5 에서 클립 표를 선택할 수 있게 한다.
- 토크(qfrc_actuator, ±0.2196 로 잘린 값)가 37% 의 시간 동안 350 mA 환산을 넘는다(최대 600 mA = 업스트림 한계). Lite 실기에서는 얼린다 → v1 필요(4장).
`p2_3_export_policy.py`: 체크포인트 → numpy npz(MLP 32→512→256→128→32, silu, tanh(loc), 관측 정규화, default_pose). jax 와 최대 차 3e-7. ROS2 노드는 numpy 만 쓴다.

**S3 진행 중** `v0-dr` (1e8, DR 켬) 15:24 시작. 평가 angvel 합 11M 14 → 22M 109 → 33M 197 (sps 23k).

**S4 준비 (08-23)** `lite_env.py`: 업스트림 클래스 상속, 모델 로드 뒤 `jnt_actfrcrange` 를 ±0.128 로, 벌점
torques −0.1 / energy −1e-3 / action_rate −0.001 (reorient 환경 값 참고). `p2_4_train_lite.py` 가 등록 후 업스트림
train main 호출. CPU 스모크 통과. 본 학습은 v0-dr 끝난 뒤 GPU 에서.

**S5 (08-23, 골격 완료)** `policy_node`(numpy 추론 0.22 ms, 20 Hz) + `policy.launch.py` + `sim_node scene:=cube`
(`leap_hand_mapping/cube_scene.py` 가 playground 장면을 assets 사전으로 연다) + `limits` 표 선택(model|teleop).
헤드리스 트윈 확인: 20M 정책이 큐브를 25 s 쥐고 있음(각속도 ≈0, 정책이 약해서). `real:=true fake:=true` 도 확인 —
이때 브리지가 ENGAGING 에 갇혔다(목표가 20 Hz 로 움직여 tol 안에 못 듦) → `engage_timeout` 3 s 추가.
남은 것: 본 학습 정책으로 트윈에서 회전 확인 → S6 실기.

**S3 (08-24)** 컴퓨터 재부팅으로 v0-dr 이 66.8M 에서 끊김 → `--load_checkpoint_path <run>/checkpoints`(디렉터리)로
나머지 34M 이어 돌림(v0-dr-resume). **66.8M 정책**: CPU MuJoCo 재생 **0.79 rad/s**(잡음 0.05) / 0.43(잡음 0), 25 s
×3 에피소드 낙하 0. 토크 >350 mA 비율 71% (예상대로 Lite 실기엔 과함). `models/rotate_z_v0_67M.npz` 로 내보냄(jax 대비 4e-7).

**S5 (08-24)** ROS2 트윈(`policy.launch.py`, 큐브 장면)에서 67M 정책: 관측 잡음 0 → 회전 ≈0(정체), **잡음 0.05 → 0.35~0.86 rad/s**.
런치 기본 `noise` 를 0.05 로. 최소 성공선(시뮬 재생) 달성.

**S3 완료 (08-24)** v0-dr 이어 돌리기(34M) 끝 → 총 1e8. 평가 angvel 합 319 → 426(≈0.85 rad/s, 확률적), 에피소드 길이 489/500.
CPU 재생(결정적, 잡음 0.05) 3 에피소드 평균 **0.52 rad/s** [0.08, 0.66, 0.83], 낙하 0, 토크 >350 mA 비율 72%, rot 사용 범위
if [−14,10] / mf [−7,32] / rf [−21,21]°. `models/rotate_z_v0.npz` (jax 대비 6e-7). GIF `docs/img/phase2_rotate_z_v0.gif`.

**S4 v1 (08-24)** v1-lite(토크 상한 0.128 + torques −0.1 / energy −1e-3 / action_rate −0.001, DR). 컴퓨터 종료로 두 번 끊겨
이어 돌림(누적 68M). 결과: 토크는 정확히 350 mA 에서 잘리고(>350 비율 0%) 큐브를 25 s 쥐지만 **회전 ≈0.03 rad/s** —
평가 angvel 합이 68M 에서 25(v0 는 33M 에 197)인데 torques 벌점 합이 −51~−62 로 **벌점이 보상을 압도**해 "가만히 쥐기"를
배웠다. 벌점 −0.1 은 과했다. → **v2-lite-caponly**: 토크 상한만 두고 보상은 업스트림 그대로(벌점 0). 상한 자체가 물리적으로
힘을 제한하므로 최소 변경이다. 12:10 시작, 1e8.

## 9. 결정 대기

- 결정 대기 (geon): 실기 거치대가 손바닥 위(palm-up)인가 / 실물 큐브 준비 가능한가. **S5 전까지만 있으면 된다** — S0~S4 는 시뮬만.
