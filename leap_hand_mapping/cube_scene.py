"""cube_scene.py — mujoco_playground 의 LEAP 큐브 장면(rotate_z 학습 환경과 같은 모델)을 **순수 MuJoCo** 로 연다.

playground 의 `scene_mjx_cube.xml` 은 메시를 `../../../../../mujoco_menagerie/...` 같은 상대 경로로 가리켜
`MjModel.from_xml_path` 로는 열리지 않는다. playground 는 파일들을 **basename 으로 assets 사전**에 넣고
`from_xml_string` 으로 연다(`leap_hand/base.py:get_assets`). 여기서 같은 일을 playground 를 import 하지 않고
한다 — ROS2 환경(leap-hand, py3.10)에는 playground 가 없기 때문이다.

필요한 것: 저장소의 `third_party/mujoco_playground`(클론) 와 `third_party/mujoco_menagerie`(클론). 둘 다
README 환경 구성대로.

    from leap_hand_mapping.cube_scene import load_cube_model
    model = load_cube_model()            # mujoco.MjModel. 키프레임 "home" = 학습 기본 자세 + 큐브 위치
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLAYGROUND_XMLS = REPO / "third_party/mujoco_playground/mujoco_playground/_src/manipulation/leap_hand/xmls"
MENAGERIE_LEAP = REPO / "third_party/mujoco_menagerie/leap_hand"
SCENE_XML = PLAYGROUND_XMLS / "scene_mjx_cube.xml"


def _add(assets: dict, folder: Path, pattern: str = "*") -> None:
    for f in sorted(folder.glob(pattern)):
        if f.is_file():
            assets[f.name] = f.read_bytes()


def cube_assets() -> dict:
    for d in (PLAYGROUND_XMLS, MENAGERIE_LEAP / "assets"):
        if not d.is_dir():
            raise FileNotFoundError(f"{d} 가 없다. README 환경 구성의 third_party 클론을 확인할 것")
    assets: dict = {}
    _add(assets, MENAGERIE_LEAP / "assets")
    _add(assets, PLAYGROUND_XMLS, "*.xml")
    _add(assets, PLAYGROUND_XMLS / "reorientation_cube_textures")
    _add(assets, PLAYGROUND_XMLS / "meshes")
    return assets


def load_cube_model(scene: os.PathLike | str = SCENE_XML):
    """학습 환경과 같은 LEAP + 큐브 MuJoCo 모델. 키프레임 "home" 포함, mocap(goal) 은 호출자가 숨긴다."""
    import mujoco

    xml = Path(scene).read_text()
    return mujoco.MjModel.from_xml_string(xml, cube_assets())
