"""tracker_node — 웹캠 -> MediaPipe 21 랜드마크 -> /hand/landmarks (PointCloud2).

로봇을 모른다. 손을 못 잡으면 **아무것도 publish 하지 않는다.** 판단은 하류가 한다.

메시지
  sensor_msgs/PointCloud2, 21점 xyz32, MediaPipe world 좌표(미터, 손 중심 원점),
  frame_id = "hand", header.stamp = 프레임을 **읽은 시각**(= 촬영 시각에 가장 가깝다).
  이 stamp 가 끝까지 전파돼서 종단 지연의 기준이 된다.

PointCloud2 를 고른 이유
  Header 가 있어 지연을 잴 수 있고, rviz2 에서 공짜로 보인다. 커스텀 메시지를 만들지
  않으므로 leap_teleop 은 순수 ament_python 으로 남는다.

거울상 함정
  좌우 반전 입력은 handedness 라벨과 world 랜드마크의 손대칭을 **동시에** 뒤집는다.
  오른손이 Right 로 안 잡히면 영상이 뒤집힌 것 -> mirror 파라미터. 라벨 필터가 곧
  좌표계 정합 검사다 (leap_hand_mapping.hand_tracker 참고).

파라미터
  camera(0) width(640) height(480) hand("Right") mirror(false) model_path(기본 모델)
  show(false)       카메라 창 + 손 위치 틀. 틀 밖이어도 publish 는 한다(경고만)
  hand_min/hand_max 틀의 거리 범위 (hand_tracker.HandGuide 기본값)
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header

from leap_hand_mapping import hand_tracker as ht
from leap_teleop import SENSOR_QOS, TOPIC_LANDMARKS


class TrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("tracker_node")
        self.declare_parameter("camera", 0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("hand", "Right")
        self.declare_parameter("mirror", False)
        self.declare_parameter("model_path", ht.DEFAULT_MODEL)
        self.declare_parameter("show", False)
        self.declare_parameter("hand_min", ht.HandGuide.hand_min)
        self.declare_parameter("hand_max", ht.HandGuide.hand_max)
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(int(p("camera")))
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 {p('camera')} 를 열 수 없다. ls /dev/video* 로 확인할 것.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(p("width")))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(p("height")))

        self.tracker = ht.HandTracker(
            model_path=str(p("model_path")), handedness=str(p("hand")), mirror=bool(p("mirror"))
        )
        self.guide = ht.HandGuide(hand_min=float(p("hand_min")), hand_max=float(p("hand_max")))
        self.show = bool(p("show"))

        self.pub = self.create_publisher(pc2.PointCloud2, TOPIC_LANDMARKS, SENSOR_QOS)

        # 카메라 read() 는 다음 프레임까지 블록한다. 타이머는 그보다 촘촘하게 돌려 두고
        # read() 가 페이스를 정하게 한다. 별도 스레드를 쓰지 않는 이유: 이 노드는 이것
        # 말고 할 일이 없다.
        self.timer = self.create_timer(1.0 / 120.0, self._tick)

        self._frames = self._published = self._out_of_box = 0
        self._t0 = time.time()
        self._last_log = self._t0
        self.get_logger().info(
            f"camera {p('camera')} {p('width')}x{p('height')}  hand={p('hand')} mirror={p('mirror')}"
            f"  -> {TOPIC_LANDMARKS}"
        )

    def _tick(self) -> None:
        ok, frame = self.cap.read()
        stamp = self.get_clock().now().to_msg()   # 읽은 직후. 촬영 시각에 가장 가깝다
        if not ok:
            return
        self._frames += 1
        frame = self.tracker.preprocess(frame)
        obs = self.tracker.process(frame)
        in_box, reason, frac = self.guide.check(obs, frame.shape)

        if obs is not None:
            if not in_box:
                self._out_of_box += 1
            header = Header(stamp=stamp, frame_id="hand")
            self.pub.publish(pc2.create_cloud_xyz32(header, obs.world.astype(np.float32)))
            self._published += 1

        if self.show:
            cv2 = self.cv2
            if obs is not None:
                ht.draw_landmarks(frame, obs)
            ht.draw_guide(frame, self.guide, in_box, reason, frac)
            status = f"{obs.handedness} {obs.score:.2f}" if obs else "no hand"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 0) if obs else (0, 0, 255), 2)
            cv2.imshow("tracker_node", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        now = time.time()
        if now - self._last_log >= 5.0:
            el = now - self._t0
            self.get_logger().info(
                f"{self._frames / el:5.1f} fps  검출 {self._published}/{self._frames}"
                f"  틀 밖 {self._out_of_box}"
            )
            self._last_log = now

    def destroy_node(self) -> bool:
        try:
            self.cap.release()
            self.tracker.close()
            if self.show:
                self.cv2.destroyAllWindows()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
