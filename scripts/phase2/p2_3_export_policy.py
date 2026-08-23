#!/usr/bin/env python3
"""p2_3_export_policy.py — brax PPO 체크포인트를 **numpy 만으로 돌릴 수 있는 npz** 로 내보낸다 (S2→S5 다리).

학습 환경(leap-mjx, py3.11, jax 0.7)과 ROS2 환경(leap-hand, py3.10, jax 0.6/brax 0.14.1)이 달라서
체크포인트를 ROS2 쪽에서 brax 로 읽는 건 버전 지뢰다. 정책은 MLP 4층(32→512→256→128→32)이라
가중치와 관측 정규화 통계만 있으면 numpy 20줄로 같은 출력이 나온다. 여기서 내보내고 **같은 입력에
대해 jax 추론과 numpy 추론이 일치하는지** 검사한 뒤 저장한다.

npz 내용
    W0,b0 … W3,b3     MLP (silu, 마지막 층은 선형) — 출력 32 = loc 16 + scale 16. 결정적 행동 = tanh(loc)
    obs_mean, obs_std "state" 관측 정규화 (brax running_statistics: (obs - mean) / std)
    default_pose      16, 학습 env 기본 자세 (keyframe home). 목표각 = default_pose + action_scale * tanh(loc)
    action_scale, ctrl_dt, obs_layout("q16+last_act16"), joint_names, source(체크포인트 경로)

    conda activate leap-mjx
    python scripts/phase2/p2_3_export_policy.py --ckpt logs/phase2/<run>/checkpoints --out models/rotate_z_v0.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

JOINT_NAMES = ["if_mcp", "if_rot", "if_pip", "if_dip", "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
               "rf_mcp", "rf_rot", "rf_pip", "rf_dip", "th_cmc", "th_axl", "th_mcp", "th_ipl"]


def numpy_policy(z: dict):
    """npz 내용으로 결정적 정책 함수(obs32 -> action16)를 만든다. ROS2 노드도 이 함수를 그대로 쓴다."""
    Ws = [z[f"W{i}"] for i in range(int(z["n_layers"]))]
    bs = [z[f"b{i}"] for i in range(int(z["n_layers"]))]
    mean, std = z["obs_mean"], z["obs_std"]

    def silu(x):
        return x / (1.0 + np.exp(-x))

    def act(obs):
        x = (np.asarray(obs, dtype=np.float64) - mean) / std
        for W, b in zip(Ws[:-1], bs[:-1]):
            x = silu(x @ W + b)
        x = x @ Ws[-1] + bs[-1]
        loc = x[: x.shape[-1] // 2]
        return np.tanh(loc)

    return act


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env_name", default="LeapCubeRotateZAxis")
    ap.add_argument("--n_check", type=int, default=200)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from p2_2_play_policy import latest_ckpt, load_policy  # noqa: E402

    import jax
    from brax.training import checkpoint as brax_ckpt
    from mujoco_playground import registry

    ckpt = latest_ckpt(Path(args.ckpt).resolve())
    cfg = json.loads((ckpt / "ppo_network_config.json").read_text())
    kw = cfg["network_factory_kwargs"]
    assert kw["activation"] == "silu" and kw["distribution_type"] == "tanh_normal", kw
    assert not kw["state_dependent_std"], "state_dependent_std 는 지원하지 않는다"
    params = brax_ckpt.load(ckpt)
    norm, pol = params[0], params[1]["params"]
    layers = sorted(pol.keys(), key=lambda k: int(k.split("_")[1]))
    Ws = [np.asarray(pol[k]["kernel"]) for k in layers]
    bs = [np.asarray(pol[k]["bias"]) for k in layers]
    obs_mean = np.asarray(norm.mean["state"]); obs_std = np.asarray(norm.std["state"])
    # brax running_statistics.normalize 와 같은 std 처리(아주 작은 std 보호)를 수치 검사로 확인한다
    from lite_env import maybe_register  # noqa: E402
    maybe_register(args.env_name)
    env_cfg = registry.get_default_config(args.env_name)
    env_cfg.impl = "jax"
    env = registry.load(args.env_name, config=env_cfg)
    default_pose = np.asarray(env.mj_model.key("home").qpos[:16])

    z = {f"W{i}": W for i, W in enumerate(Ws)} | {f"b{i}": b for i, b in enumerate(bs)}
    z.update(n_layers=np.int64(len(Ws)), obs_mean=obs_mean, obs_std=obs_std, default_pose=default_pose,
             action_scale=np.float64(env_cfg.action_scale), ctrl_dt=np.float64(env_cfg.ctrl_dt),
             obs_layout=np.str_("q16+last_act16"), joint_names=np.array(JOINT_NAMES),
             source=np.str_(str(ckpt)), env_name=np.str_(args.env_name))
    np_act = numpy_policy(z)

    # 검사: jax 추론(브락스 경로) vs numpy — 같은 관측 200개
    jax_policy = jax.jit(load_policy(ckpt, deterministic=True))
    rng = np.random.default_rng(0)
    # 관측 분포는 정규화 통계 근처로 만든다 (관절각 ± , last_act ∈ [-1,1])
    obs = obs_mean + obs_std * rng.standard_normal((args.n_check, obs_mean.size))
    obs[:, 16:] = np.clip(obs[:, 16:], -1, 1)
    worst = 0.0
    for o in obs:
        a_j = np.asarray(jax_policy({"state": o.astype(np.float32), "privileged_state": np.zeros(105, np.float32)},
                                    jax.random.PRNGKey(0))[0])
        a_n = np_act(o)
        worst = max(worst, float(np.abs(a_j - a_n).max()))
    print(f"jax vs numpy 최대 |차이| {worst:.2e} (관측 {args.n_check}개)")
    if worst > 1e-3:
        raise SystemExit("numpy 추론이 jax 와 다르다. brax running_statistics.normalize 의 std 하한(clip) 과"
                         " 활성 함수 정의를 대조할 것")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **z)
    print(f"저장 {out}  층 {[W.shape for W in Ws]}  action_scale {float(z['action_scale'])}  ctrl_dt {float(z['ctrl_dt'])}")
    print(f"기본 자세(deg) {np.degrees(default_pose).round(0).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
