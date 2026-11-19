#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[관제 PC에서 실행] 웹 서버 <-> 로봇 중계 노드 (관제PC_연동_요청.txt 기준)

  웹 서버 ──{ns}/web_command──▶ 이 노드 ──▶ 로봇  (drive·stop·cancel·pause·navigate·dock·undock)
  웹 서버 ◀──{ns}/web_feedback── 이 노드            (navigate·dock·undock 결과)
  웹 서버 ◀──{ns}/web_status──── 이 노드 ◀── 로봇  (배터리·위치·연결, 0.5초 주기)

로봇별 ({ns} = /robot6, /robot11)
  구독: {ns}/web_command (std_msgs/String JSON)
        {ns}/battery_state, {ns}/amcl_pose, {ns}/dock_status (상태 / 연결 판단)
  발행: {ns}/web_feedback, {ns}/web_status (std_msgs/String JSON)
        {ns}/cmd_vel (TwistStamped)
        {ns}/pause (std_msgs/Bool) → worker_follow 노드 정지/재개
  액션: {ns}/navigate_to_pose, {ns}/dock, {ns}/undock

웹 명령과 자동 추종(worker_follow)의 우선순위: 웹 명령 우선
  - pause(true)            → {ns}/pause=True (추종 정지 유지). pause(false) → 재개
  - drive / navigate / dock → 실행하는 동안 {ns}/pause=True 로 추종을 멈춤
      drive    : stop 이 오거나 0.5초 동안 drive 가 없으면 정지하고 해제
      navigate / dock : 끝나면(성공·실패·취소) 해제 (도킹된 뒤에는 worker_follow 가 스스로 멈춤)
  - undock(웹에서 로봇 선택): dock_status 확인 → 도킹이면 언독, 이미 언독이면 건너뜀
  - B로봇 호출(navigate, purpose "call", caller "robot6"): 호출된 로봇의 web_command 로 옴
      1) 호출된 로봇이 도킹이면 언독 먼저 (이미 언독이면 건너뜀)
      2) 부른 로봇(추종 중인 로봇)의 좌표 = 관제 PC 가 받은 최신 amcl_pose (없으면 명령의 x, y)
      3) 부른 로봇 앞 call_stop_distance 지점으로 이동 (부른 로봇을 바라봄)
         가는 동안 부른 로봇이 call_update_distance 이상 움직이면 목표를 다시 보냄
         가는 중 카메라에 터틀봇이 잡히면({ns}/robot_follow/found) 이동을 멈추고 성공 처리
         → worker_follow robot 모드가 그때부터 터틀봇 추종
      4) 도착/인식/실패 시 web_feedback (command "navigate")
  - user_register/* 토픽은 웹 서버 ↔ 객체탐지 PC 가 직접 주고받으므로 중계하지 않음

실행
  ros2 run my_package web_relay
  ros2 run my_package web_relay --ros-args -p robots:="['robot6']"
  ros2 run my_package web_relay --ros-args -p robot_ips:="['192.168.0.106', '192.168.0.111']"
"""

import json
import math
import re
import subprocess
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, String

FEEDBACK_COMMANDS = ('navigate', 'dock', 'undock')

# 구역 / 도킹 대기 지점 좌표 (map 기준 x, y, yaw rad) — 관제 PC(web_relay)가 기준
#   웹(스마트폰·관제 모니터) 서버는 구역 이름('A')·'dock' 만 보내므로 여기서 좌표를 채운다.
#   (final_map 에서 도크로부터 도달 가능·벽과 0.4 m 이상 확인, 2026-11-19)
ZONE_POSES = {
    'A': (-0.83, -0.90, 0.0),
    'B': (-2.93, 0.0, 0.0),
    'C': (-4.6, 0.6, 0.0),
}
DOCK_WAYPOINTS = {            # 웹 robot1 = robot6, robot2 = robot11
    'robot6': (-0.33, 1.23, 0.1),
    'robot11': (-0.45, -0.49, 0.13),
}


def fill_coordinates(robot, c):
    """구역 이동·도킹 명령의 좌표를 관제 PC 표로 채운다 (보낸 쪽 좌표보다 우선)."""
    command = c.get('command')
    if command == 'navigate' and c.get('purpose') == 'zone':
        zone = str(c.get('zone', '')).upper()
        if zone not in ZONE_POSES:
            raise ValueError(f'unknown zone "{c.get("zone")}" (known: {sorted(ZONE_POSES)})')
        c['x'], c['y'], c['yaw'] = ZONE_POSES[zone]
    elif command == 'dock':
        if robot not in DOCK_WAYPOINTS:
            raise ValueError(f'no dock waypoint for {robot}')
        c['x'], c['y'], c['yaw'] = DOCK_WAYPOINTS[robot]


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def finite(*vals) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


class RobotLink:
    """로봇 한 대의 상태 · 명령 처리."""

    def __init__(self, node: 'WebRelay', name: str, ip: str):
        self.node = node
        self.name = name
        self.ns = '/' + name
        self.ip = ip
        log = node.get_logger()
        self.log = log

        # 상태
        self.battery = None
        self.pose = None              # (x, y, yaw)
        self.pose_time = 0.0          # amcl_pose 마지막 수신 (monotonic)
        self.is_docked = None
        self.last_robot_msg = 0.0     # 로봇 토픽 마지막 수신 (연결 판단)
        self.ping_ms = None

        # 명령 상태
        self.paused = False           # 웹 pause(true/false)
        self.driving = False
        self.last_drive = 0.0
        self.nav_active = False       # navigate 실행 중 (추종 정지)
        self.goal_handle = None       # 현재 Nav2 / Dock 액션 goal
        self.task = None              # {'id', 'command', 'stage'} 진행 중 navigate/dock
        self.task_seq = 0
        self.last_pause_pub = None
        self.last_pause_time = 0.0

        n = node
        f = lambda t: f'{self.ns}/{t}'   # noqa: E731
        pose_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        n.create_subscription(String, f('web_command'), self.on_command, 10)
        n.create_subscription(BatteryState, f('battery_state'), self.on_battery,
                              qos_profile_sensor_data)
        n.create_subscription(PoseWithCovarianceStamped, f('amcl_pose'), self.on_pose, pose_qos)
        n.create_subscription(DockStatus, f('dock_status'), self.on_dock_status,
                              qos_profile_sensor_data)
        self.pub_feedback = n.create_publisher(String, f('web_feedback'), 10)
        self.pub_status = n.create_publisher(String, f('web_status'), 10)
        self.pub_cmd = n.create_publisher(TwistStamped, f('cmd_vel'), 10)
        self.pub_pause = n.create_publisher(Bool, f('pause'), 10)
        # 구역 이동·도킹 때 추종을 끊기 위한 등록 취소 (worker_follow·worker_detect 가 구독)
        self.pub_register_cancel = n.create_publisher(Bool, f('user_register/cancel'), 10)
        # 호출로 가는 중 터틀봇(부른 로봇)이 카메라에 잡힘 (worker_follow robot 모드가 발행)
        n.create_subscription(Bool, f('robot_follow/found'), self.on_robot_found, 10)
        self.nav_client = ActionClient(n, NavigateToPose, f('navigate_to_pose'))
        self.dock_client = ActionClient(n, Dock, f('dock'))
        self.undock_client = ActionClient(n, Undock, f('undock'))

        if ip:
            threading.Thread(target=self.ping_loop, daemon=True).start()
        log.info(f'[{name}] relay ready (ping {ip or "off"})')

    # ================= 로봇 → 상태 =================
    def on_battery(self, m: BatteryState):
        p = m.percentage
        if math.isfinite(p):
            self.battery = round(p * 100.0 if p <= 1.0 else p, 1)
        self.last_robot_msg = time.monotonic()

    def on_pose(self, m: PoseWithCovarianceStamped):
        p = m.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quat(p.orientation))
        self.pose_time = self.last_robot_msg = time.monotonic()

    def on_dock_status(self, m: DockStatus):
        self.is_docked = bool(m.is_docked)
        self.last_robot_msg = time.monotonic()

    def ping_loop(self):
        while rclpy.ok():
            start = time.time()
            ms = None
            try:
                out = subprocess.run(['ping', '-c', '1', '-W', '1', self.ip],
                                     capture_output=True, text=True, timeout=3).stdout
                m = re.search(r'time[=<]([\d.]+)\s*ms', out)
                ms = float(m.group(1)) if m else None
            except (OSError, subprocess.TimeoutExpired):
                pass
            self.ping_ms = ms
            time.sleep(max(0.0, 1.0 - (time.time() - start)))

    def status_json(self) -> dict:
        now = time.monotonic()
        x = y = yaw = None
        if self.pose is not None:
            x, y, yaw = (round(v, 3) for v in self.pose)
        return {
            'robot': self.name,
            'battery': self.battery,
            'x': x, 'y': y, 'yaw': yaw,
            'connected': self.last_robot_msg > 0
                         and now - self.last_robot_msg <= self.node.connected_timeout,
            'ping_ms': self.ping_ms,
        }

    # ================= 주기 처리 =================
    def tick(self):
        now = time.monotonic()
        # drive 안전: 마지막 drive 후 drive_timeout 동안 새 drive 가 없으면 정지
        if self.driving and now - self.last_drive > self.node.drive_timeout:
            self.log.warning(f'[{self.name}] drive timeout -> stop')
            self.stop_motion()
        self.publish_pause_state(now)

    def publish_status(self):
        self.pub_status.publish(String(data=json.dumps(self.status_json(), ensure_ascii=False)))

    def follow_blocked(self) -> bool:
        return self.paused or self.driving or self.nav_active

    def publish_pause_state(self, now=None):
        """worker_follow 에 정지/재개 전달: 바뀔 때 + 1초마다 (나중에 켠 노드도 알게)."""
        now = time.monotonic() if now is None else now
        blocked = self.follow_blocked()
        if blocked != self.last_pause_pub or now - self.last_pause_time >= 1.0:
            if blocked != self.last_pause_pub:
                self.log.info(f'[{self.name}] follow {"PAUSED" if blocked else "RESUMED"} '
                              f'(web_pause={self.paused} web_drive={self.driving} '
                              f'web_goal_running={self.nav_active})')
            self.pub_pause.publish(Bool(data=blocked))
            self.last_pause_pub, self.last_pause_time = blocked, now

    # ================= 웹 명령 =================
    def feedback(self, cmd_id, command, ok, message=''):
        msg = {'id': cmd_id, 'command': command, 'ok': bool(ok), 'message': message}
        self.pub_feedback.publish(String(data=json.dumps(msg, ensure_ascii=False)))
        self.log.info(f'[{self.name}] feedback {msg}')

    def on_command(self, msg: String):
        try:
            c = json.loads(msg.data)
            if not isinstance(c, dict):
                raise ValueError('not a JSON object')
        except ValueError as exc:
            self.log.error(f'[{self.name}] bad web_command ({exc}): {msg.data[:200]}')
            return
        cmd_id, command = c.get('id'), c.get('command')
        if c.get('robot') not in (None, self.name):
            self.log.error(f'[{self.name}] web_command for "{c.get("robot")}" on {self.ns}; ignored')
            if command in FEEDBACK_COMMANDS:
                self.feedback(cmd_id, command, False, 'robot mismatch')
            return
        if command != 'drive':
            self.log.info(f'[{self.name}] web_command {c}')
        try:
            handler = {
                'drive': self.cmd_drive, 'stop': self.cmd_stop, 'cancel': self.cmd_cancel,
                'pause': self.cmd_pause, 'navigate': self.cmd_navigate,
                'dock': self.cmd_dock, 'undock': self.cmd_undock,
            }.get(command)
            if command == 'navigate' and c.get('purpose') == 'call':
                handler = self.cmd_call          # B로봇 호출: 언독 → 부른 로봇 위치로
            if handler is None:
                self.log.error(f'[{self.name}] unknown command "{command}"')
                return
            fill_coordinates(self.name, c)
            handler(c)
        except Exception as exc:  # 명령 하나가 실패해도 노드는 계속
            self.log.error(f'[{self.name}] command {command} failed: {exc}')
            if command in FEEDBACK_COMMANDS:
                self.feedback(cmd_id, command, False, f'error: {exc}')

    # ---------- drive / stop ----------
    def publish_cmd(self, linear, angular):
        m = TwistStamped()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = 'base_link'
        m.twist.linear.x, m.twist.angular.z = float(linear), float(angular)
        self.pub_cmd.publish(m)

    def cmd_drive(self, c):
        lin, ang = float(c.get('linear', 0.0)), float(c.get('angular', 0.0))
        if not finite(lin, ang):
            raise ValueError('linear/angular not finite')
        ml, ma = self.node.max_linear, self.node.max_angular
        lin, ang = max(-ml, min(ml, lin)), max(-ma, min(ma, ang))
        if not self.driving:
            self.log.info(f'[{self.name}] drive start ({lin:.2f}, {ang:.2f})')
        self.driving = True
        self.last_drive = time.monotonic()
        self.publish_pause_state()        # 추종 먼저 정지
        self.publish_cmd(lin, ang)

    def stop_motion(self):
        self.driving = False
        self.publish_cmd(0.0, 0.0)
        self.publish_pause_state()

    def cmd_stop(self, c):
        self.stop_motion()

    # ---------- cancel ----------
    def cmd_cancel(self, c):
        self.cancel_task('canceled by web')

    def cancel_task(self, reason):
        """진행 중 navigate / dock 취소 (결과 feedback 은 ok false)."""
        task, gh = self.task, self.goal_handle
        self.task, self.goal_handle = None, None
        self.nav_active = False
        if gh is not None:
            try:
                gh.cancel_goal_async()
            except Exception as exc:
                self.log.warning(f'[{self.name}] cancel failed: {exc}')
        if task is not None:
            self.feedback(task['id'], task['command'], False, reason)
        self.publish_pause_state()

    # ---------- pause ----------
    def cmd_pause(self, c):
        self.paused = bool(c.get('value', True))
        if self.paused:
            self.stop_motion()
        self.publish_pause_state()

    # ---------- navigate ----------
    def make_nav_goal(self, c):
        x, y, yaw = float(c['x']), float(c['y']), float(c.get('yaw', 0.0))
        if not finite(x, y, yaw):
            raise ValueError('x/y/yaw not finite')
        g = NavigateToPose.Goal()
        g.pose.header.frame_id = 'map'
        g.pose.header.stamp = self.node.get_clock().now().to_msg()
        g.pose.pose.position.x, g.pose.pose.position.y = x, y
        g.pose.pose.orientation.z = math.sin(yaw / 2.0)
        g.pose.pose.orientation.w = math.cos(yaw / 2.0)
        return g

    def start_task(self, c):
        """새 navigate/dock: 이전 작업은 취소 처리 후 새 작업 등록."""
        if self.task is not None:
            self.cancel_task(f'replaced by command id {c.get("id")}')
        self.task_seq += 1
        self.task = {'id': c.get('id'), 'command': c['command'], 'seq': self.task_seq}
        return self.task

    def task_alive(self, task) -> bool:
        return self.task is task

    def finish_task(self, task, ok, message):
        if not self.task_alive(task):
            return
        self.task, self.goal_handle = None, None
        self.nav_active = False
        self.feedback(task['id'], task['command'], ok, message)
        self.publish_pause_state()

    def send_action(self, task, client, goal, label, on_done):
        """액션 goal 전송 → 결과를 on_done(ok, message) 로."""
        if not client.server_is_ready():
            on_done(False, f'{label} server not ready')
            return

        def on_goal(fut):
            if not self.task_alive(task):
                try:
                    fut.result().cancel_goal_async()
                except Exception:
                    pass
                return
            try:
                gh = fut.result()
            except Exception as exc:
                on_done(False, f'{label} request failed: {exc}')
                return
            if not gh.accepted:
                on_done(False, f'{label} goal rejected')
                return
            self.goal_handle = gh

            def on_result(rf):
                if not self.task_alive(task):
                    return
                try:
                    status = rf.result().status
                except Exception as exc:
                    on_done(False, f'{label} result error: {exc}')
                    return
                ok = status == GoalStatus.STATUS_SUCCEEDED
                on_done(ok, f'{label} {"succeeded" if ok else f"failed (status {status})"}')
            gh.get_result_async().add_done_callback(on_result)

        client.send_goal_async(goal).add_done_callback(on_goal)

    FOLLOW_HANDOFF_SEC = 0.6   # 추종 중이었으면 이 시간 뒤에 이동 시작 (추종 노드의 마지막 goal 이 덮어쓰지 않게)

    def cut_following(self, c):
        """구역 이동·도킹: 추종을 끊는다 (도착 후에도 다시 사람에게 돌아가지 않게 등록 취소)."""
        if c.get('command') == 'dock' or c.get('purpose') == 'zone':
            self.pub_register_cancel.publish(Bool(data=True))
            self.log.info(f'[{self.name}] following cut (register cancel) for {c.get("command")} '
                          f'{c.get("zone", "")}')

    def start_after_handoff(self, task, was_following, start):
        """추종 중이었으면 pause 가 먼저 적용되도록 잠깐 기다린 뒤 이동 시작."""
        if not was_following:
            start()
            return
        holder = {}

        def fire():
            holder['t'].cancel()
            if self.task_alive(task):
                start()
        holder['t'] = self.node.create_timer(self.FOLLOW_HANDOFF_SEC, fire)

    def cmd_navigate(self, c):
        goal = self.make_nav_goal(c)
        was_following = not self.follow_blocked()
        task = self.start_task(c)
        self.nav_active = True
        self.stop_motion()                # drive 중이었으면 정지, 추종 정지 (pause 발행)
        self.cut_following(c)
        what = (f'zone {c.get("zone")}' if c.get('purpose') == 'zone'
                else f'call from {c.get("caller")}' if c.get('purpose') == 'call' else 'goal')
        self.log.info(f'[{self.name}] navigate ({what}) -> '
                      f'({c["x"]:.2f}, {c["y"]:.2f}, yaw {float(c.get("yaw", 0.0)):.2f})'
                      + (' (was following: stop first)' if was_following else ''))
        self.start_after_handoff(task, was_following, lambda: self.send_action(
            task, self.nav_client, goal, 'navigate', lambda ok, m: self.finish_task(task, ok, m)))

    # ---------- dock ----------
    def cmd_dock(self, c):
        goal = self.make_nav_goal(c)
        was_following = not self.follow_blocked()
        task = self.start_task(c)
        self.nav_active = True
        self.stop_motion()
        self.cut_following(c)

        def after_nav(ok, message):
            if not ok:
                self.finish_task(task, False, f'dock waypoint: {message}')
                return
            if not self.task_alive(task):
                return
            self.log.info(f'[{self.name}] dock waypoint reached -> docking')
            self.send_action(task, self.dock_client, Dock.Goal(), 'dock',
                             lambda ok2, m2: self.finish_task(task, ok2, m2))

        self.log.info(f'[{self.name}] dock: waypoint ({c["x"]:.2f}, {c["y"]:.2f}) then dock'
                      + (' (was following: stop first)' if was_following else ''))
        self.start_after_handoff(task, was_following, lambda: self.send_action(
            task, self.nav_client, goal, 'navigate to dock waypoint', after_nav))

    # ---------- undock ----------
    def cmd_undock(self, c):
        cmd_id = c.get('id')
        if self.task is not None:
            self.cancel_task(f'replaced by undock id {cmd_id}')
        self.undock_when_known(cmd_id, time.monotonic() + self.node.dock_status_wait)

    def undock_when_known(self, cmd_id, deadline):
        """dock_status 로 도킹 여부 확인 → 도킹이면 언독 / 언독이면 건너뜀 / 모르면 기다림."""
        if self.is_docked is None:
            if time.monotonic() < deadline:
                timer = []

                def retry():
                    timer[0].cancel()
                    self.node.destroy_timer(timer[0])
                    self.undock_when_known(cmd_id, deadline)
                timer.append(self.node.create_timer(0.2, retry))
                return
            self.feedback(cmd_id, 'undock', False,
                          f'no dock status on {self.ns}/dock_status; undock not sent')
            return
        if self.is_docked is False:
            self.log.info(f'[{self.name}] already undocked; undock skipped')
            self.feedback(cmd_id, 'undock', True, 'already undocked')
            return
        self.log.info(f'[{self.name}] docked -> undocking')
        if not self.undock_client.server_is_ready():
            self.feedback(cmd_id, 'undock', False, 'undock server not ready')
            return

        def on_goal(fut):
            try:
                gh = fut.result()
            except Exception as exc:
                self.feedback(cmd_id, 'undock', False, f'undock request failed: {exc}')
                return
            if not gh.accepted:
                self.feedback(cmd_id, 'undock', False, 'undock goal rejected')
                return

            def on_result(rf):
                try:
                    status = rf.result().status
                except Exception as exc:
                    self.feedback(cmd_id, 'undock', False, f'undock result error: {exc}')
                    return
                ok = status == GoalStatus.STATUS_SUCCEEDED
                self.feedback(cmd_id, 'undock', ok,
                              'undock succeeded' if ok else f'undock failed (status {status})')
            gh.get_result_async().add_done_callback(on_result)

        self.undock_client.send_goal_async(Undock.Goal()).add_done_callback(on_goal)

    # ---------- call (B로봇 호출) ----------
    def call_later(self, sec, fn):
        """sec 초 뒤 fn() 한 번 실행."""
        holder = []

        def fire():
            holder[0].cancel()
            self.node.destroy_timer(holder[0])
            fn()
        holder.append(self.node.create_timer(sec, fire))

    def caller_position(self, c):
        """부른 로봇 위치 (x, y, 출처): 관제 PC 가 받은 최신 amcl_pose 우선, 없으면 명령의 x, y."""
        link = self.node.link_by_name(c.get('caller'))
        if link is not None and link.pose is not None:
            return link.pose[0], link.pose[1], 'live amcl_pose'
        if finite(c.get('x'), c.get('y')):
            return float(c['x']), float(c['y']), 'command x/y'
        return None

    def make_call_goal(self, cx, cy):
        """부른 로봇 앞 call_stop_distance 지점 (부른 로봇을 바라봄). 이미 그 안이면 None."""
        d = self.node.call_stop_distance
        if self.pose is not None:
            rx, ry = self.pose[0], self.pose[1]
        else:                                    # 내 위치를 아직 모름: 보통 도크에 있으므로 도킹 지점 기준
            rx, ry = DOCK_WAYPOINTS.get(self.name, (cx - d - 1.0, cy, 0.0))[:2]
        dx, dy = cx - rx, cy - ry
        dist = math.hypot(dx, dy)
        if dist <= d + 0.05:
            return None
        gx, gy = cx - dx / dist * d, cy - dy / dist * d
        return self.make_nav_goal({'x': gx, 'y': gy, 'yaw': math.atan2(dy, dx)})

    def cmd_call(self, c):
        caller = c.get('caller')
        if caller == self.name:
            raise ValueError('caller is this robot')
        if self.caller_position(c) is None:
            raise ValueError(f'no position for caller "{caller}"')
        was_following = not self.follow_blocked()
        task = self.start_task(c)
        task['caller'] = caller
        task['call'] = True                 # 가는 중 터틀봇을 인식하면 추종으로 넘김 (on_robot_found)
        self.nav_active = True               # 이동하는 동안 이 로봇의 추종은 멈춤
        self.stop_motion()
        self.log.info(f'[{self.name}] call from {caller}: '
                      f'{"undock first, then " if self.is_docked else ""}go to {caller}')
        deadline = time.monotonic() + self.node.dock_status_wait
        self.start_after_handoff(task, was_following, lambda: self.ensure_undocked(
            task, deadline, lambda: self.go_to_caller(task, c)))

    def ensure_undocked(self, task, deadline, then):
        """도킹이면 언독 후 then(), 이미 언독이면 바로 then(), 도킹 여부를 모르면 기다림."""
        if not self.task_alive(task):
            return
        if self.is_docked is None:
            if time.monotonic() < deadline:
                self.call_later(0.2, lambda: self.ensure_undocked(task, deadline, then))
                return
            self.finish_task(task, False, f'no dock status on {self.ns}/dock_status; call aborted')
            return
        if self.is_docked is False:
            then()
            return
        self.log.info(f'[{self.name}] call: docked -> undocking first')

        def after_undock(ok, message):
            if not ok:
                self.finish_task(task, False, f'call: {message}')
                return
            self.log.info(f'[{self.name}] call: undocked -> going to caller')
            then()
        self.send_action(task, self.undock_client, Undock.Goal(), 'undock', after_undock)

    def go_to_caller(self, task, c):
        if not self.task_alive(task):
            return
        pos = self.caller_position(c)
        if pos is None:
            self.finish_task(task, False, f'no position for caller "{c.get("caller")}"')
            return
        goal = self.make_call_goal(pos[0], pos[1])
        if goal is None:
            self.finish_task(task, True, f'already near {c.get("caller")}')
            return
        self.send_call_goal(task, c, goal, pos)
        task['timer'] = self.node.create_timer(self.node.call_update_period,
                                               lambda: self.update_call_goal(task, c))

    def send_call_goal(self, task, c, goal, pos):
        task['nav_seq'] = seq = task.get('nav_seq', 0) + 1
        task['target'] = (pos[0], pos[1])
        p = goal.pose.pose.position
        self.log.info(f'[{self.name}] call: goal ({p.x:.2f}, {p.y:.2f}) near {c.get("caller")} '
                      f'at ({pos[0]:.2f}, {pos[1]:.2f}) [{pos[2]}]')

        def done(ok, message):
            if task.get('nav_seq') != seq:
                return                       # 새 목표로 대체된 이전 goal 의 결과
            self.stop_call_timer(task)
            self.finish_task(task, ok, f'call from {c.get("caller")}: {message}')
        self.send_action(task, self.nav_client, goal, 'navigate to caller', done)

    def update_call_goal(self, task, c):
        """가는 동안 부른 로봇이 움직였으면 목표 갱신."""
        if not self.task_alive(task):
            self.stop_call_timer(task)
            return
        pos = self.caller_position(c)
        if pos is None or 'target' not in task:
            return
        tx, ty = task['target']
        moved = math.hypot(pos[0] - tx, pos[1] - ty)
        if moved < self.node.call_update_distance:
            return
        goal = self.make_call_goal(pos[0], pos[1])
        if goal is None:
            return                           # 이미 충분히 가까움: 지금 goal 이 끝나게 둠
        self.log.info(f'[{self.name}] call: {c.get("caller")} moved {moved:.2f} m -> update goal')
        self.send_call_goal(task, c, goal, pos)

    def on_robot_found(self, msg: Bool):
        """호출로 가는 중 카메라에 터틀봇이 잡힘 → 좌표 이동을 멈추고 호출 성공 처리 (→ 터틀봇 추종 시작)."""
        task = self.task
        if not msg.data or task is None or not task.get('call'):
            return
        self.log.info(f'[{self.name}] call: TurtleBot detected on the way -> stop going, hand over to robot follow')
        self.stop_call_timer(task)
        gh = self.goal_handle
        if gh is not None:
            try:
                gh.cancel_goal_async()
            except Exception as exc:
                self.log.warning(f'[{self.name}] cancel failed: {exc}')
        self.finish_task(task, True, f'call from {task.get("caller", "?")}: '
                                     'TurtleBot detected on the way -> robot follow')

    def stop_call_timer(self, task):
        t = task.pop('timer', None)
        if t is not None:
            t.cancel()
            self.node.destroy_timer(t)

    def shutdown(self):
        if self.driving:
            self.publish_cmd(0.0, 0.0)


class WebRelay(Node):
    def __init__(self):
        super().__init__('web_relay')
        dp = self.declare_parameter
        dp('robots', ['robot6', 'robot11'])
        dp('robot_ips', [''])                  # robots 와 같은 순서, '' = Ping 안 함
        dp('status_period', 0.5)               # web_status 주기 (초)
        dp('connected_timeout', 3.0)           # 로봇 토픽이 이 시간 안 오면 connected=false
        dp('drive_timeout', 0.5)               # 마지막 drive 후 정지까지 (초)
        dp('max_linear', 0.2)
        dp('max_angular', 0.8)
        dp('dock_status_wait', 3.0)            # undock 명령 때 도킹 여부를 모르면 기다리는 시간
        dp('call_stop_distance', 0.8)          # 호출: 부른 로봇에서 이 거리 앞에 멈춤 (m)
        dp('call_update_distance', 0.5)        # 호출: 부른 로봇이 이만큼 움직이면 목표 갱신 (m)
        dp('call_update_period', 2.0)          # 호출: 목표 갱신 확인 주기 (초)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.connected_timeout = float(g('connected_timeout'))
        self.drive_timeout = float(g('drive_timeout'))
        self.max_linear = abs(float(g('max_linear')))
        self.max_angular = abs(float(g('max_angular')))
        self.dock_status_wait = float(g('dock_status_wait'))
        self.call_stop_distance = abs(float(g('call_stop_distance')))
        self.call_update_distance = abs(float(g('call_update_distance')))
        self.call_update_period = max(0.2, float(g('call_update_period')))

        names = [str(r).strip('/') for r in g('robots') if str(r).strip('/')]
        ips = list(g('robot_ips'))
        self.links = [RobotLink(self, n, ips[i] if i < len(ips) else '')
                      for i, n in enumerate(names)]
        self.create_timer(float(g('status_period')), self.publish_status)
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f'web relay started for {names}')

    def link_by_name(self, name):
        for link in self.links:
            if link.name == name:
                return link
        return None

    def publish_status(self):
        for link in self.links:
            link.publish_status()

    def tick(self):
        for link in self.links:
            link.tick()


def main(args=None):
    rclpy.init(args=args)
    node = WebRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for link in node.links:
            try:
                link.shutdown()
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
