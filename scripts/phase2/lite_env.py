"""lite_env.py — LEAP Hand **Lite** 용 rotate_z 변형 환경 `LeapCubeRotateZAxisLite` 를 playground 레지스트리에 등록한다.

업스트림 `LeapCubeRotateZAxis` 는 관절 토크 상한 ±0.2196 N·m (= 600 mA × 0.366) 의 풀 LEAP 을 모델링한다.
우리 실기는 Lite 라 `curr_lim` 350 mA → 같은 환산으로 **0.128 N·m.** 학습 정책이 그 위의 힘을 쓰면 실기에선
전류 한계에 붙어 브리지가 얼린다. 여기서는 업스트림 클래스를 **상속**해 모델 로드 뒤 `jnt_actfrcrange` 만 바꾼다.
업스트림 코드는 건드리지 않는다. 보상 벌점(torques/energy/action_rate)은 업스트림 설정에 이미 있는 키라
`--playground_config_overrides` 나 default_config 로 넣는다.

    from lite_env import register_lite
    register_lite()                       # 이후 registry.load("LeapCubeRotateZAxisLite") 가 된다
"""

from __future__ import annotations

import numpy as np

ENV_NAME = "LeapCubeRotateZAxisLite"
BASE_NAME = "LeapCubeRotateZAxis"
TORQUE_PER_AMP = 0.366           # leap_rh_mjx.xml 주석
LITE_TORQUE_LIMIT = 0.350 * TORQUE_PER_AMP   # 0.128 N·m


def register_lite(torque_limit: float = LITE_TORQUE_LIMIT,
                  torques: float = -0.1, energy: float = -1e-3, action_rate: float = -0.001) -> str:
    """등록하고 env 이름을 돌려준다. 이미 등록돼 있으면 그대로."""
    from ml_collections import config_dict
    from mujoco import mjx
    from mujoco_playground._src import manipulation
    from mujoco_playground._src.manipulation.leap_hand import leap_hand_constants as consts
    from mujoco_playground._src.manipulation.leap_hand import rotate_z
    from mujoco_playground.config import manipulation_params

    if ENV_NAME in manipulation._envs:
        return ENV_NAME

    def lite_default_config() -> config_dict.ConfigDict:
        cfg = rotate_z.default_config()
        cfg.torque_limit = torque_limit                 # N·m, 손 관절 16개 공통
        cfg.reward_config.scales.torques = torques      # Σ τ² 에 곱한다 (음수 = 벌점)
        cfg.reward_config.scales.energy = energy        # Σ |q̇ τ|
        cfg.reward_config.scales.action_rate = action_rate
        return cfg

    class CubeRotateZAxisLite(rotate_z.CubeRotateZAxis):
        """업스트림 + 관절 토크 상한(Lite 350 mA)."""

        def __init__(self, config=None, config_overrides=None):
            super().__init__(config=config or lite_default_config(), config_overrides=config_overrides)
            cap = float(self._config.torque_limit)
            jids = [self._mj_model.joint(n).id for n in consts.JOINT_NAMES]
            self._mj_model.jnt_actfrclimited[jids] = 1
            self._mj_model.jnt_actfrcrange[jids] = np.array([-cap, cap])
            self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

    manipulation._envs[ENV_NAME] = CubeRotateZAxisLite
    manipulation._cfgs[ENV_NAME] = lite_default_config
    manipulation._randomizer[ENV_NAME] = rotate_z.domain_randomize    # 같은 관절/지오메트리 id

    # PPO 하이퍼파라미터는 업스트림 rotate_z 것을 그대로
    _orig = manipulation_params.brax_ppo_config

    def _patched(env_name, impl=None):
        return _orig(BASE_NAME if env_name == ENV_NAME else env_name, impl)

    manipulation_params.brax_ppo_config = _patched
    return ENV_NAME


def maybe_register(env_name: str) -> None:
    if env_name == ENV_NAME:
        register_lite()
