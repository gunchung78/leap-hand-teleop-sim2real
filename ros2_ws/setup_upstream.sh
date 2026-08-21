#!/usr/bin/env bash
# 업스트림 LEAP_Hand_API/ros2_module 을 ros2_ws/src/leap_hand 로 가져오고 우리 패치를 얹는다.
#
# 왜 복사본을 커밋하지 않는가
#   - 업스트림 라이선스가 CC BY-NC 4.0 이라 우리 저장소에 통째로 넣지 않는다.
#   - 우리가 바꾼 것은 patches/leap_hand_port_param.patch 하나뿐이고, 그게 전부다.
#   - third_party/ 는 README 안내대로 각자 clone 한다 (다른 자산과 같은 규칙).
#
# 사용
#   bash ros2_ws/setup_upstream.sh          # 복사 + 패치
#   cd ros2_ws && colcon build --symlink-install
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/third_party/LEAP_Hand_API/ros2_module"
DST="$REPO/ros2_ws/src/leap_hand"
PATCH="$REPO/patches/leap_hand_port_param.patch"

[ -d "$SRC" ] || { echo "업스트림이 없다: $SRC  (README 환경 구성의 LEAP_Hand_API clone 참고)"; exit 1; }
rm -rf "$DST"
cp -r "$SRC" "$DST"
patch -p1 -d "$DST" < "$PATCH"
echo "ok: $DST  (업스트림 $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo '?') + $(basename "$PATCH"))"
