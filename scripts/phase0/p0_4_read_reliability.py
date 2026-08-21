#!/usr/bin/env python3
"""p0_4_read_reliability.py — 실기 상태 읽기의 오류율을 잰다. 토크는 건드리지 않는다.

증상: ROS2 브리지가 /leap_pos_vel_eff 를 30 Hz 로 읽으면 업스트림 드라이버가
`> read: [TxRxResult] Incorrect status packet!` 를 초당 한두 번 찍는다(약 3%). latency_timer 1,
Return Delay Time 0 으로 바꿔도 그대로였다. 깨진 읽기는 업스트림 코드가 **직전 값을 돌려주므로**
위험하지는 않지만, 어디서 오는지와 어떤 조건에서 줄어드는지를 숫자로 알아야 한다.

무엇을 재는가
  읽기 방식 x 조회 주기 격자에서 N 회 읽고 실패 비율을 센다.
    fast   GroupFastSyncRead (업스트림 기본, Protocol 2.0 0x8A 한 패킷에 16모터)
    sync   GroupSyncRead     (모터마다 응답 패킷, 구형)
    pos    위치만 / pvc 위치+속도+전류 (브리지가 쓰는 것)
  주기 30 / 15 Hz.

쓰는 법 (포트를 쓰는 런치/스크립트를 먼저 끈다 — 포트는 하나만 연다)
    python scripts/phase0/p0_4_read_reliability.py
    python scripts/phase0/p0_4_read_reliability.py --port /dev/serial/by-id/... --n 300
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party/LEAP_Hand_API/python"))

from leap_hand_mapping.real_hand import find_port  # noqa: E402

MOTORS = list(range(16))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=4_000_000)
    ap.add_argument("--n", type=int, default=300, help="조합당 읽기 횟수")
    ap.add_argument("--rates", type=float, nargs="*", default=[30.0, 15.0])
    args = ap.parse_args()

    from leap_hand_utils.dynamixel_client import DynamixelClient

    port = args.port or find_port()
    if port is None:
        print("U2D2 포트를 못 찾았다. ls /dev/serial/by-id 로 확인하고 --port 로 지정할 것.")
        return 2
    lat = None
    dev = os.path.realpath(port)
    lt = f"/sys/bus/usb-serial/devices/{os.path.basename(dev)}/latency_timer"
    if os.path.exists(lt):
        lat = open(lt).read().strip()
    print(f"포트 {port} -> {dev}  baud {args.baud}  latency_timer {lat}")

    client = DynamixelClient(MOTORS, port, args.baud)
    client.connect()                     # 포트만 연다. 토크/모드는 건드리지 않는다

    # 실패 횟수는 handle_packet_result 를 감싸서 센다 (업스트림 코드 수정 없음)
    fails = {"n": 0}
    orig = client.handle_packet_result

    def counting(comm_result, dxl_error=None, dxl_id=None, context=None):
        ok = orig(comm_result, dxl_error, dxl_id, context)
        if not ok:
            fails["n"] += 1
        return ok

    client.handle_packet_result = counting
    readers = {"pos": client._pos_reader, "pvc": client._pos_vel_cur_reader}

    print(f"\n{'방식':<6}{'읽기':<6}{'Hz':>5}{'시도':>6}{'실패':>6}{'실패율':>8}{'평균ms':>8}{'최대ms':>8}")
    print("-" * 54)
    for mode in ("fast", "sync"):
        for rname, reader in readers.items():
            op = reader.operation
            saved = getattr(op, "fastSyncRead", None)
            if mode == "sync" and saved is not None:
                # 업스트림 read() 는 fastSyncRead 가 예외를 내면 txRxPacket(일반 SyncRead) 로 간다
                def _raise(*a, **k):
                    raise AttributeError("forced plain SyncRead")
                op.fastSyncRead = _raise
            for hz in args.rates:
                fails["n"] = 0
                dts = []
                period = 1.0 / hz
                with contextlib.redirect_stdout(io.StringIO()):   # "Update your Dynamixel_SDK" 출력 억제
                    for _ in range(args.n):
                        t0 = time.perf_counter()
                        reader.read()
                        dts.append((time.perf_counter() - t0) * 1000)
                        rest = period - (time.perf_counter() - t0)
                        if rest > 0:
                            time.sleep(rest)
                print(f"{mode:<6}{rname:<6}{hz:5.0f}{args.n:6d}{fails['n']:6d}{fails['n'] / args.n:8.1%}"
                      f"{np.mean(dts):8.1f}{np.max(dts):8.1f}")
            if mode == "sync" and saved is not None:
                op.fastSyncRead = saved
    client.disconnect()
    print("\n읽는 법: fast 와 sync 의 실패율이 같으면 배선/전기 문제(4 Mbps 에서 CRC 깨짐)고,")
    print("         fast 만 높으면 FastSyncRead 펌웨어/SDK 문제다. 주기를 낮춰도 비율이 같으면 주기 탓이 아니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
