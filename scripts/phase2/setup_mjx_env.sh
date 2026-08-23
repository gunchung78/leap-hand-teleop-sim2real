#!/usr/bin/env bash
# setup_mjx_env.sh — Phase 2 학습 전용 conda 환경 `leap-mjx` 를 만든다.
#
# 왜 환경을 따로 두나
#   mujoco_playground 최신판(third_party/mujoco_playground, e74217b)은 Python >= 3.11 을 요구한다.
#   ROS2 Humble(rclpy)은 3.10 고정이라 `leap-hand` 환경에는 못 넣는다. pip 의 playground 0.1.0 은
#   3.10 에 깔리지만 mujoco 3.11 과 안 맞는다(mjx.make_data(nconmax) 오류).
#   → 학습은 leap-mjx(3.11), 추론/ROS2 는 leap-hand(3.10). 정책은 파일(onnx/npz)로 넘긴다.
#
# 쓰는 법
#   bash scripts/phase2/setup_mjx_env.sh
#   conda activate leap-mjx
#   python -c "import jax; print(jax.devices())"      # CudaDevice 가 보여야 한다
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ENV_NAME=${ENV_NAME:-leap-mjx}
PY=${PY:-3.11}

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" "python=$PY" -y
fi
conda activate "$ENV_NAME"

# JAX CUDA 휠은 pypi 에서. playground 는 저장소 안의 클론(커밋 고정)을 editable 로.
pip install -U pip
pip install -e "$REPO/third_party/mujoco_playground"
# brax 0.14.2 는 jax.device_put_replicated 를 쓰는데 jax 0.10 에서 제거됐다 (08-23 확인) -> 0.7.2 고정.
# playground/mjx 가 jax 를 최신으로 끌어올리므로 **그 뒤에** 고정한다.
pip install -U "jax[cuda12]==0.7.2"
# 선택: 학습 로그·영상
pip install tensorboardX tensorboard mediapy onnx tf2onnx 2>/dev/null || pip install tensorboardX tensorboard mediapy
conda install -y -c conda-forge ffmpeg      # mediapy 영상 저장 (업스트림 train 의 rollout mp4 도 이걸 쓴다)

python - <<'EOF'
import jax, mujoco, mujoco_playground, brax
print("jax", jax.__version__, jax.devices())
print("mujoco", mujoco.__version__, "brax", brax.__version__)
print("playground", mujoco_playground.__file__)
EOF
echo "done: conda activate $ENV_NAME"
