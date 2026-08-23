#!/usr/bin/env python3
"""p2_4_train_lite.py — Lite(350 mA) 변형 환경으로 업스트림 PPO 학습을 돌린다 (S4, v1).

업스트림 `learning/train_jax_ppo.py` 를 그대로 쓰되, 먼저 `lite_env.register_lite()` 로
`LeapCubeRotateZAxisLite` 를 등록한 뒤 그 스크립트의 main 을 호출한다. 업스트림 코드 수정 없음.
나머지 플래그(--num_timesteps, --domain_randomization, --suffix ...)는 그대로 통과한다.

    conda activate leap-mjx
    python scripts/phase2/p2_4_train_lite.py --num_timesteps 100000000 --domain_randomization --suffix v1-lite
    python scripts/phase2/p2_4_train_lite.py --torque_limit 0.128 --torques -0.1 --energy -1e-3 --action_rate -0.001 ...

결과는 p2_1 과 같은 logs/phase2/<env>-<시각>-<suffix>/. 재생·내보내기는 p2_2 / p2_3 에 --env_name LeapCubeRotateZAxisLite.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LEARNING = REPO / "third_party" / "mujoco_playground" / "learning"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--torque_limit", type=float, default=None, help="N·m. 기본 0.350*0.366=0.128 (Lite 350 mA)")
    ap.add_argument("--torques", type=float, default=-0.1, help="Σ τ² 벌점 계수 (음수)")
    ap.add_argument("--energy", type=float, default=-1e-3)
    ap.add_argument("--action_rate", type=float, default=-0.001)
    ap.add_argument("--logdir", default=str(REPO / "logs" / "phase2"))
    args, rest = ap.parse_known_args()

    sys.path.insert(0, str(HERE))
    from lite_env import LITE_TORQUE_LIMIT, register_lite  # noqa: E402

    env_name = register_lite(torque_limit=args.torque_limit or LITE_TORQUE_LIMIT, torques=args.torques,
                             energy=args.energy, action_rate=args.action_rate)

    sys.path.insert(0, str(LEARNING))
    os.chdir(LEARNING)                      # 업스트림 스크립트가 상대 경로(logs/)를 쓸 수 있어 같은 cwd 로
    import train_jax_ppo  # noqa: E402  (absl 플래그 정의)
    from absl import app  # noqa: E402

    argv = [sys.argv[0], f"--env_name={env_name}", "--impl=jax", f"--logdir={args.logdir}", "--use_tb"] + rest
    print("train_jax_ppo.py", " ".join(argv[1:]))
    print(f"  torque_limit {args.torque_limit or LITE_TORQUE_LIMIT:.4f} N·m  torques {args.torques}  energy {args.energy}"
          f"  action_rate {args.action_rate}")
    app.run(train_jax_ppo.main, argv=argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
