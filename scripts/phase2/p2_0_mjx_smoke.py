#!/usr/bin/env python3
"""p2_0_mjx_smoke.py — MJX LeapCubeRotateZAxis 가 이 GPU 에서 몇 env 로 얼마나 빨리 도는지 잰다 (S0).

학습 전에 알아야 할 숫자 둘: env 수에 따른 GPU 메모리와 env-steps/s. 업스트림 기본 8192 env 가
6 GB 노트북 GPU 에 들어가는지, 1e8 스텝이 몇 시간인지 여기서 외삽한다.

    conda activate leap-mjx
    python scripts/phase2/p2_0_mjx_smoke.py                   # 1024, 2048, 4096 순서로
    python scripts/phase2/p2_0_mjx_smoke.py --envs 8192       # 하나만
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def gpu_mib() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        used, total = out.split(",")
        return f"{int(used)}/{int(total)} MiB"
    except Exception:  # noqa: BLE001
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", type=int, nargs="*", default=[1024, 2048, 4096])
    ap.add_argument("--steps", type=int, default=100, help="계측 스텝 수 (JIT 뒤)")
    ap.add_argument("--env_name", default="LeapCubeRotateZAxis")
    args = ap.parse_args()

    import jax
    import jax.numpy as jp
    from mujoco_playground import registry

    print(f"jax {jax.__version__}  devices {jax.devices()}  GPU {gpu_mib()}")
    cfg = registry.get_default_config(args.env_name)
    cfg.impl = "jax"
    env = registry.load(args.env_name, config=cfg)
    print(f"{args.env_name}: ctrl_dt {cfg.ctrl_dt}  sim_dt {cfg.sim_dt}  action {env.action_size}  "
          f"obs {dict(env.observation_size) if isinstance(env.observation_size, dict) else env.observation_size}")

    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    print(f"\n{'envs':>6} {'JIT s':>7} {'step ms':>8} {'env-steps/s':>12} {'1e8 스텝 시간':>12}  GPU")
    print("-" * 70)
    for n in args.envs:
        try:
            rng = jax.random.split(jax.random.PRNGKey(0), n)
            t0 = time.perf_counter()
            st = reset(rng)
            act = jp.zeros((n, env.action_size))
            st = step(st, act)
            jax.block_until_ready(st.reward)
            jit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(args.steps):
                st = step(st, act)
            jax.block_until_ready(st.reward)
            dt = time.perf_counter() - t0
            sps = args.steps * n / dt
            hours = 1e8 / sps / 3600
            print(f"{n:>6} {jit_s:>7.1f} {dt / args.steps * 1000:>8.1f} {sps / 1e3:>10.0f}k {hours:>10.1f} h  {gpu_mib()}")
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:80]
            print(f"{n:>6}  실패: {type(e).__name__}: {msg}")
            break
    print("\n읽는 법: env-steps/s 는 순수 시뮬 속도라 PPO(네트워크·학습 갱신)까지 넣으면 대략 절반으로 본다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
