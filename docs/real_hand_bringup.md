# 실기 LEAP Hand Lite 시작하기

작업대에서 그대로 따라갈 수 있게 순서대로 정리했다.
**0단계부터 순서대로.** 건너뛰면 모터가 상한다.

> **Lite 라는 것을 잊지 말 것.** 스톨 토크 0.52 N·m, 연속 정격 0.10 N·m, 기어는
> 엔지니어링 플라스틱이다. 전류 제한 350 mA 는 코드에 고정해 두었고 인자로 노출하지
> 않았다. Full 값인 550 으로 올리면 기어 이빨이 깨진다.

---

## 이 머신에서 확인된 사실

| 항목 | 값 |
|---|---|
| U2D2 | `FT232H Single` (`0403:6014`) |
| 포트 | `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBIN91W-if00-port0` → `ttyUSB0` |
| 다른 FTDI | `FT2232 Dual` → `ttyUSB1`, `ttyUSB2` — **U2D2 가 아니다. 쓰지 말 것** |

`ttyUSB0` 이라는 번호는 USB 를 다시 꽂으면 바뀔 수 있다. 항상 `by-id` 경로를 쓴다.

---

## 0단계 — 전원과 배선 (손대기 전)

- [ ] 5V 30A 전원이 Power Hub 에 연결되어 있고 켜져 있다
- [ ] U2D2 가 Power Hub 와 PC 에 연결되어 있다
- [ ] 데이지체인 커넥터가 16개 모터에 모두 물려 있다
- [ ] **Dynamixel Wizard 가 떠 있으면 종료한다** — 포트를 점유해서 API 가 못 붙는다
- [ ] 손 주변에 손가락이 부딪힐 물건이 없다

빨간 LED 가 점멸 중인 모터가 있으면 과부하 상태다. **전원을 껐다 켜야만** 복구된다.

---

## 1단계 — 포트 권한 (1회만)

현재 `geon` 은 `dialout` 그룹에 없어서 포트를 열 수 없다.

```bash
sudo usermod -aG dialout $USER
```

**로그아웃 후 다시 로그인해야 적용된다.** 확인:

```bash
id | grep dialout
```

재로그인 없이 지금 당장 해보려면 임시로:

```bash
sudo chmod 666 /dev/ttyUSB0
```

이 방법은 USB 를 다시 꽂으면 초기화되므로 매번 해줘야 한다. `usermod` 쪽이 낫다.

---

## 2단계 — USB 지연시간 (문서 4.4)

현재 FTDI latency timer 가 **16 ms** (기본값)다. 이대로면 제어 주파수가 크게 떨어진다.

```bash
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

재부팅/재연결하면 초기화된다. 영구 적용하려면 udev 규칙을 만든다.

```bash
sudo tee /etc/udev/rules.d/99-u2d2-latency.rules >/dev/null <<'EOF'
ACTION=="add", SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Return Delay Time

모터 쪽 지연(레지스터 9)은 기본 250 µs 다. 0 으로 낮추면 쿼리레이트가 올라간다
(이론 최대 ~500 Hz). Dynamixel Wizard 에서 16개 모터의 **Return Delay Time 을 0** 으로
바꾸고 **반드시 Wizard 를 종료**한다. 한 번만 하면 EEPROM 에 남는다.

지금 당장 안 해도 손은 움직인다. 제어 주파수를 재기 시작할 때 하면 된다.

---

## 3단계 — 사전 점검 (모터를 움직이지 않는다)

```bash
conda activate leap-hand
cd ~/Project/leap-hand-teleop-sim2real
python scripts/preflight_real_hand.py
```

토크를 켜지 않고 연결만 해서 확인한다.

1. 포트 존재 / 권한
2. latency timer
3. 4 Mbps 통신
4. **모터 ID 0~15 응답 여부**
5. 현재 관절각·전류 (모터 ID 순서와 MuJoCo 순서를 나란히 출력)

이때 손은 **힘없이 늘어져 있어야** 정상이다. 뻣뻣하면 이전 세션의 토크가 남은 것이니
전원을 껐다 켠다.

무응답 ID 가 나오면 배선/커넥터, 해당 모터의 ID 와 baudrate(4 Mbps)를 확인한다.
"범위 밖" 경고가 뜨면 손을 편 자세로 두고 다시 돌려 본다. 그래도 남으면 영점 교정 문제다.

---

## 4단계 — 관절 하나만 움직여 보기

**여기서 처음으로 손이 움직인다.** 반드시 한 관절부터.

```bash
# 먼저 시뮬만 돌려서 무엇이 움직일지 눈으로 익힌다
python scripts/sweep_joints.py --joints 0

# 실기. MuJoCo 0번 = if_mcp = 실기 모터 ID 1 (검지 굽힘)
python scripts/sweep_joints.py --real --joints 0
```

검지가 굽혔다 폈다 하면 성공이다. **다른 손가락이 움직이면 매핑이 틀린 것**이니
즉시 중단하고(Ctrl+C) 알린다.

다음으로 짝을 이루는 관절을 확인한다. 이 둘이 바로 매핑의 핵심이다.

```bash
python scripts/sweep_joints.py --real --joints 1   # if_rot = 모터 0 (검지 벌림)
```

0번은 **굽힘**, 1번은 **벌림**이어야 한다. 반대로 나오면 매핑이 뒤집힌 것이다.

엄지는 범위가 좁고 간섭이 잦으니 따로 본다.

```bash
python scripts/sweep_joints.py --real --joints 12 13 14 15
```

`th_axl`(13번)은 축방향 회전이라 손끝이 거의 안 움직인다. 정상이다.

---

## 5단계 — 전체 순차 구동

```bash
python scripts/sweep_joints.py --real
```

16개 관절을 하나씩 왕복시키고, 끝나면 관절별 **추종 오차(deg)** 를 집계한다.
이 숫자가 sim-real 갭의 첫 정량 지표이고, 이력서에 넣을 수치의 출발점이다.

진동하거나 과부하가 잦으면 게인을 낮춘다 (문서 4.2).

```bash
python scripts/sweep_joints.py --real --kp 400
```

---

## 안전 수칙

| 상황 | 대처 |
|---|---|
| 빨간 LED 점멸 | 과부하. **전원 사이클** 외에는 복구 안 됨 |
| 전류 임계 초과 | 스크립트가 자동으로 해당 관절 구동을 중단한다 |
| 손가락이 뭔가에 걸림 | 즉시 Ctrl+C. 스크립트가 토크를 끄고 나간다 |
| 진동 / 떨림 | `--kp 400` 으로 낮춘다 |
| 그립 자세 장시간 유지 | **하지 말 것.** 연속 정격 0.10 N·m 는 스톨의 1/5. 발열·마모 급증 |

플라스틱 기어라 스톨을 반복하면 **백래시가 누적**된다. sim2real 갭의 주범이므로
주기적으로 영점을 재교정하고, 스페어 모터 2~3개를 확보해 두는 게 좋다.

---

## 코드에서 직접 쓰기

바깥 코드는 **MuJoCo 관절 순서만** 알면 된다. 모터 ID 치환과 π 오프셋은 드라이버가 처리한다.

```python
import numpy as np
from leap_hand_mapping.real_hand import LeapHandDriver, find_port

with LeapHandDriver(port=find_port(), kp=600) as hand:   # 빠져나갈 때 토크 자동 해제
    hand.command_mujoco(np.zeros(16))     # MuJoCo qpos -> 실기. 범위 클립 후 전송
    q = hand.read_mujoco()                # 실기 -> MuJoCo 규약
    over = hand.check_current()           # [(모터ID, 전류)] — 비어 있으면 정상
```

`command_mujoco` 는 두 모델의 관절 범위 **교집합**으로 클립한 뒤 보낸다.
MJCF 와 공식 URDF 의 엄지 범위가 서로 달라서 필요한 조치다.

---

## 문제 해결

**`Failed to open port`**
Dynamixel Wizard 종료 확인. 5V 전원 확인. 권한(1단계) 확인.
다른 파이썬 프로세스가 잡고 있는지: `ls -l /proc/*/fd 2>/dev/null | grep ttyUSB`

**일부 모터만 무응답**
데이지체인은 순서대로 물리므로, 무응답 ID 가 연속이면 그 앞 커넥터가 빠졌을 가능성이 높다.
Wizard 로 baudrate 가 4 Mbps 인지 확인한다.

**손이 뻣뻣한데 명령이 안 먹음**
이전 세션 토크가 남은 것. 전원 사이클.

**추종 오차가 특정 관절만 큼**
그 관절 기어의 백래시이거나 영점이 틀어진 것. Wizard 로 해당 모터 위치를 직접 확인.
