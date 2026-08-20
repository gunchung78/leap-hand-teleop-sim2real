#!/usr/bin/env bash
# MediaPipe HandLandmarker 모델을 받는다 (약 7.5MB).
#
# mediapipe 1.x 는 legacy mp.solutions.hands 를 없앴고 tasks API 만 남았는데,
# tasks API 는 모델을 패키지에 넣어 두지 않는다. 저장소에 넣기엔 커서 받아 쓴다.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p models
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
curl -sSfL -o models/hand_landmarker.task "$URL"
ls -lh models/hand_landmarker.task
