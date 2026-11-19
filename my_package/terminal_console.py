#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[관제 PC] 웹 서버 대신 터미널로 테스트하는 콘솔 (my_package_BU 버전 web_relay 용)

웹 서버가 보내던 JSON 을 web_relay 에 그대로 보낸다.
  보냄:  /<robot>/web_command          (std_msgs/String JSON) drive·stop·cancel·pause·navigate·dock·undock
         /<robot>/user_register/reply  (std_msgs/String 'y'/'n')  웹 '맞음'/'아님' → worker_follow 추종 허용
         /<robot>/user_register/retry  (std_msgs/Bool)            웹 '재인식'
         /<robot>/user_register/cancel (std_msgs/Bool)            웹 '인식 취소'
  받음:  /<robot>/web_feedback, /<robot>/web_status, /<robot>/dock_status
         /<robot>/user_register/image  (등록 사진 → ~/worker_register_<robot>.jpg)
         /<robot>/user_track/following (작업자 인식 중 / 미인식)

실행 (web_relay, worker_follow, worker_detect 를 먼저 실행)
  ros2 run my_package console 1
"""

import json
import os
import select
import sys
import termios
import threading
import time
import tty

import rclpy
from irobot_create_msgs.msg import DockStatus
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String

ROBOTS = {'1': 'robot6', '2': 'robot11'}

# 구역 (x, y, yaw rad) — final_map 에서 도크로부터 도달 가능 확인 (2026-11-19)
ZONES = {'A': (-0.83, -0.90, 0.0), 'B': (-2.93, 0.0, 0.0), 'C': (-4.6, 0.6, 0.0)}   # 실제 좌표는 web_relay 표가 기준

# 도킹 대기 지점 (x, y, yaw rad): 웹 robot1 = robot6, robot2 = robot11
DOCK_WAYPOINTS = {'robot6': (-0.33, 1.23, 0.1), 'robot11': (-0.45, -0.49, 0.13)}   # 실제 좌표는 web_relay 표가 기준

TELEOP_LINEAR = 0.2
TELEOP_ANGULAR = 0.8

HELP = """
 ─────────────────────────── 명령 ───────────────────────────
  undock           언독 (= 웹에서 이 로봇 선택)
  y / n            작업자 등록 사진 '맞음'(추종 시작) / '아님'  (사진: ~/worker_register_<robot>.jpg)
  retry / rcancel  재인식 / 인식 취소(추종 끔)
  a / b / c        구역 A/B/C 로 이동
  go X Y [YAW]     좌표로 이동 (map 기준)
  dock             도킹 대기 지점으로 이동 후 도킹
  call             B로봇(나머지 로봇) 호출: 도킹이면 언독 → 이 로봇 앞까지 이동
  pause / resume   일시정지(추종 멈춤) / 재개
  stop / cancel    정지 / 진행 중 이동 취소
  teleop           키보드 수동 주행 (w/s/a/d, 스페이스 정지, x 나가기)
  status           상태 보기          help / quit
 ──────────────────────────────────────────────────────────────
 순서: undock → (웹캠 앞에서 손 들기 → 사진 확인 → y) → 추종 → a/b/c → dock
"""

PRINT_LOCK = threading.Lock()
PROMPT = ['']


def say(text):
    with PRINT_LOCK:
        sys.stdout.write(f'\r\033[K{text}\n{PROMPT[0]}')
        sys.stdout.flush()


class Console(Node):
    def __init__(self, robot):
        super().__init__('terminal_console')
        self.robot = robot
        ns = f'/{robot}'
        self.status, self.docked, self.following = {}, None, None
        self.cmd_id = int(time.time()) % 100000 * 10
        self.drive_fail_shown, self.teleop_abort = False, None
        self.pub_cmd = self.create_publisher(String, f'{ns}/web_command', 10)
        self.pub_reply = self.create_publisher(String, f'{ns}/user_register/reply', 10)
        self.pub_retry = self.create_publisher(Bool, f'{ns}/user_register/retry', 10)
        self.pub_rcancel = self.create_publisher(Bool, f'{ns}/user_register/cancel', 10)
        self.create_subscription(String, f'{ns}/web_feedback', self.on_feedback, 10)
        self.create_subscription(String, f'{ns}/web_status', self.on_status, 10)
        self.create_subscription(CompressedImage, f'{ns}/user_register/image', self.on_image, 10)
        self.create_subscription(Bool, f'{ns}/user_track/following', self.on_following, 10)
        self.create_subscription(DockStatus, f'{ns}/dock_status',
                                 lambda m: setattr(self, 'docked', bool(m.is_docked)),
                                 qos_profile_sensor_data)
        # B로봇 호출: 나머지 로봇의 web_command 로 보내고 결과를 받는다
        self.other = next((r for r in ROBOTS.values() if r != robot), None)
        if self.other:
            self.pub_call = self.create_publisher(String, f'/{self.other}/web_command', 10)
            self.create_subscription(String, f'/{self.other}/web_feedback', self.on_call_feedback, 10)
        self.call_ids = set()

    # ---------- incoming ----------
    def on_feedback(self, msg):
        try:
            f = json.loads(msg.data)
        except ValueError:
            return
        if f.get('command') == 'drive':
            if f.get('ok'):
                return
            self.teleop_abort = f.get('message')
            if self.drive_fail_shown:
                return
            self.drive_fail_shown = True
        mark = 'OK ' if f.get('ok') else 'FAIL'
        say(f'[{self.robot}] ◀ {mark} {f.get("command")} (id {f.get("id")}): {f.get("message")}')
        if f.get('ok') and f.get('command') == 'undock':
            say(f'[{self.robot}] 다음: 웹캠 앞 1.5~2.5 m 에서 손을 머리 위로 1초 → 사진 오면 y')

    def on_call_feedback(self, msg):
        try:
            f = json.loads(msg.data)
        except ValueError:
            return
        if f.get('id') not in self.call_ids:
            return
        self.call_ids.discard(f.get('id'))
        mark = 'OK ' if f.get('ok') else 'FAIL'
        say(f'[{self.other}] ◀ {mark} 호출 (id {f.get("id")}): {f.get("message")}')

    def on_status(self, msg):
        try:
            s = json.loads(msg.data)
        except ValueError:
            return
        if self.status and s.get('connected') != self.status.get('connected'):
            say(f'[{self.robot}] 연결: {self.status.get("connected")} → {s.get("connected")}')
        self.status = s

    def on_image(self, msg):
        path = os.path.expanduser(f'~/worker_register_{self.robot}.jpg')
        try:
            with open(path, 'wb') as fp:
                fp.write(bytes(msg.data))
        except OSError as exc:
            say(f'[{self.robot}] 등록 사진 저장 실패: {exc}')
            return
        say(f'[{self.robot}] 등록 사진 도착 → {path}  (맞으면 y, 아니면 n)')

    def on_following(self, msg):
        if self.following != msg.data:
            say(f'[{self.robot}] 작업자 {"인식 중" if msg.data else "미인식"}')
        self.following = msg.data

    # ---------- outgoing ----------
    def send(self, command, **fields):
        self.cmd_id += 1
        c = {'id': self.cmd_id, 'robot': self.robot, 'command': command, **fields}
        self.pub_cmd.publish(String(data=json.dumps(c)))
        if command != 'drive':
            shown = {k: v for k, v in c.items() if k not in ('id', 'robot')}
            say(f'[{self.robot}] ▶ {shown} (id {self.cmd_id})')

    def relay_ok(self):
        # Discovery via the remote server can take a few seconds after start-up.
        deadline = time.monotonic() + 5.0
        while not self.status and time.monotonic() < deadline:
            time.sleep(0.2)
        if not self.status:
            say(f'[{self.robot}] web_relay 응답 없음 → ros2 run my_package web_relay 실행 확인')
            return False
        return True

    def must_be_undocked(self, what):
        if self.docked:
            say(f'[{self.robot}] {what} 불가: 로봇이 도크 위에 있습니다 → 먼저 undock')
            return False
        return self.relay_ok()

    def zone(self, z):
        if not self.must_be_undocked('구역 이동'):
            return
        x, y, yaw = ZONES[z]
        say(f'[{self.robot}] 구역 {z} ({x}, {y}) 로 이동')
        self.send('navigate', purpose='zone', zone=z, x=x, y=y, yaw=yaw)

    def dock(self):
        if not self.relay_ok():
            return
        if self.robot not in DOCK_WAYPOINTS:
            say(f'[{self.robot}] 도킹 대기 지점이 없습니다 (DOCK_WAYPOINTS)')
            return
        x, y, yaw = DOCK_WAYPOINTS[self.robot]
        say(f'[{self.robot}] 도킹 대기 지점 ({x}, {y}, {yaw}) 이동 후 도킹')
        self.send('dock', x=x, y=y, yaw=yaw)

    def call_other(self):
        """나머지 로봇을 이 로봇 위치로 호출 (웹 'B로봇 호출' 과 같은 명령)."""
        if not self.other:
            say('호출할 다른 로봇이 없습니다.')
            return
        if not self.relay_ok():
            return
        self.cmd_id += 1
        c = {'id': self.cmd_id, 'robot': self.other, 'command': 'navigate',
             'purpose': 'call', 'caller': self.robot, 'yaw': 0.0}
        x, y = self.status.get('x'), self.status.get('y')
        if x is not None and y is not None:     # web_relay 는 최신 amcl_pose 를 우선 사용
            c['x'], c['y'] = x, y
        self.call_ids.add(self.cmd_id)
        self.pub_call.publish(String(data=json.dumps(c)))
        say(f'[{self.other}] ▶ 호출: {self.robot} 위치 ({x}, {y}) 로 (id {self.cmd_id})')

    def show_status(self):
        s = self.status
        if not s:
            say(f'[{self.robot}] web_status 를 아직 받지 못함 (web_relay 실행 확인)')
        else:
            say(f'[{self.robot}] ' + ', '.join(f'{k}={s.get(k)}' for k in
                                              ('connected', 'battery', 'x', 'y', 'yaw', 'ping_ms')))
        say(f'[{self.robot}] 도크 위: {self.docked} | 작업자 인식: '
            f'{self.following if self.following is not None else "정보 없음 (등록 전)"}')

    def teleop(self):
        if not sys.stdin.isatty():
            say('teleop 은 터미널에서만 됩니다.')
            return
        if not self.must_be_undocked('teleop'):
            return
        self.drive_fail_shown, self.teleop_abort = False, None
        say('teleop: w 전진 / s 후진 / a 좌회전 / d 우회전 / 스페이스 정지 / x 나가기 (누르고 있는 동안 이동)')
        keys = {'w': (TELEOP_LINEAR, 0.0), 's': (-TELEOP_LINEAR, 0.0),
                'a': (0.0, TELEOP_ANGULAR), 'd': (0.0, -TELEOP_ANGULAR)}
        old = termios.tcgetattr(sys.stdin)
        cur, last_key = (0.0, 0.0), 0.0
        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                if self.teleop_abort:
                    say(f'[{self.robot}] teleop 중단: {self.teleop_abort}')
                    break
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                now = time.monotonic()
                if r:
                    k = sys.stdin.read(1).lower()
                    if k == 'x':
                        break
                    if k == ' ':
                        cur, last_key = (0.0, 0.0), 0.0
                        self.send('stop')
                        continue
                    if k in keys:
                        cur, last_key = keys[k], now
                if last_key and now - last_key < 0.6:
                    self.send('drive', linear=cur[0], angular=cur[1])
                elif last_key:
                    last_key = 0.0
                    self.send('stop')
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            self.send('stop')
            say('teleop 종료')


def choose_robot(arg=None):
    if arg:
        r = ROBOTS.get(arg, arg if (arg in ROBOTS.values() or arg.startswith('robot')) else None)
        if r is None:
            print(f'알 수 없는 로봇 "{arg}" (1, 2, robot6, robot11)')
        return r
    while True:
        try:
            c = input('로봇 선택 -> [1] robot6  [2] robot11  (q: 종료): ').strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if c.lower() == 'q':
            return None
        if c in ROBOTS:
            return ROBOTS[c]
        print('1 또는 2를 입력하세요.')


def main():
    args = remove_ros_args(sys.argv)[1:]
    robot = choose_robot(args[0] if args else None)
    if robot is None:
        return
    rclpy.init()
    node = Console(robot)
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    say(HELP)
    say('구역: ' + ', '.join(f'{z}{v[:2]}' for z, v in ZONES.items())
        + f' | 도킹 대기 지점: {DOCK_WAYPOINTS.get(robot)}')
    try:
        while rclpy.ok():
            try:
                PROMPT[0] = f'[{robot}] 명령> '
                line = input(PROMPT[0]).strip()
                PROMPT[0] = ''
            except EOFError:
                break
            if not line:
                continue
            parts = line.split()
            c = parts[0].lower()
            try:
                if c in ('quit', 'exit', 'q'):
                    break
                elif c == 'help':
                    say(HELP)
                elif c == 'undock':
                    if node.relay_ok():
                        node.send('undock')
                elif c in ('y', 'n'):
                    node.pub_reply.publish(String(data=c))
                    say(f'[{robot}] ▶ 등록 확인 응답 {c}')
                elif c == 'retry':
                    node.pub_retry.publish(Bool(data=True))
                    say(f'[{robot}] ▶ 재인식 요청')
                elif c == 'rcancel':
                    node.pub_rcancel.publish(Bool(data=True))
                    say(f'[{robot}] ▶ 인식 취소')
                elif c in ('a', 'b', 'c'):
                    node.zone(c.upper())
                elif c == 'go':
                    x, y = float(parts[1]), float(parts[2])
                    yaw = float(parts[3]) if len(parts) > 3 else 0.0
                    if node.must_be_undocked('이동'):
                        node.send('navigate', x=x, y=y, yaw=yaw)
                elif c == 'dock':
                    node.dock()
                elif c == 'call':
                    node.call_other()
                elif c == 'pause':
                    node.send('pause', value=True)
                elif c == 'resume':
                    node.send('pause', value=False)
                elif c in ('stop', 'cancel'):
                    node.send(c)
                elif c == 'teleop':
                    node.teleop()
                elif c == 'status':
                    node.show_status()
                else:
                    say(f'알 수 없는 명령 "{c}" (help)')
            except (IndexError, ValueError):
                say('형식이 잘못됐습니다. 예: go -1.0 0.5 0.0')
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
