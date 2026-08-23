#!/usr/bin/env bash
# p2_1_train.sh — 업스트림 train_jax_ppo.py 를 우리 기본값으로 호출한다 (S1/S3). 업스트림 코드는 건드리지 않는다.
#
#   conda activate leap-mjx
#   bash scripts/phase2/p2_1_train.sh                          # v0: 업스트림 그대로, 1e8 스텝 (본 학습)
#   STEPS=20000000 SUFFIX=v0-short bash scripts/phase2/p2_1_train.sh     # 짧게 (파이프라인·시간 가늠)
#   DR=1 bash scripts/phase2/p2_1_train.sh                     # 도메인 무작위화 켬
#   EXTRA='--num_envs 4096' bash scripts/phase2/p2_1_train.sh  # 나머지 플래그는 그대로 전달
#
# 결과: logs/phase2/<env>-<시각>-<suffix>/  (checkpoints/, TensorBoard, rollout*.mp4)
#   tensorboard --logdir logs/phase2
#
# PPO 하이퍼파라미터는 업스트림 manipulation_params.brax_ppo_config("LeapCubeRotateZAxis") 가 기본이다
# (8192 env, unroll 40, minibatch 32, lr 3e-4, entropy 1e-2, 망 512-256-128, 정책 입력 state / 비평가 privileged_state).
# 여기서는 스텝 수·로그 위치·영상 수만 정한다. impl 은 jax (warp 는 설치하지 않았다).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ENV_NAME=${ENV_NAME:-LeapCubeRotateZAxis}
STEPS=${STEPS:-100000000}
SUFFIX=${SUFFIX:-v0}
EVALS=${EVALS:-10}
VIDEOS=${VIDEOS:-2}
SEED=${SEED:-1}
LOGDIR=${LOGDIR:-$REPO/logs/phase2}
DR=${DR:-0}
EXTRA=${EXTRA:-}

ARGS=(--env_name "$ENV_NAME" --impl jax --num_timesteps "$STEPS" --num_evals "$EVALS"
      --num_videos "$VIDEOS" --seed "$SEED" --logdir "$LOGDIR" --suffix "$SUFFIX" --use_tb)
if [ "$DR" = "1" ]; then ARGS+=(--domain_randomization); fi

mkdir -p "$LOGDIR"
echo "train_jax_ppo.py ${ARGS[*]} $EXTRA"
cd "$REPO/third_party/mujoco_playground/learning"
exec python train_jax_ppo.py "${ARGS[@]}" $EXTRA
