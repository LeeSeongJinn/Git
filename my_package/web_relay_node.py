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
        self.last_robot_msg = time.monotonic()

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
            if handler is None:
                self.log.error(f'[{self.name}] unknown command "{command}"')
                return
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

    def cmd_navigate(self, c):
        goal = self.make_nav_goal(c)
        task = self.start_task(c)
        self.nav_active = True
        self.stop_motion()                # drive 중이었으면 정지, 추종 정지
        what = (f'zone {c.get("zone")}' if c.get('purpose') == 'zone'
                else f'call from {c.get("caller")}' if c.get('purpose') == 'call' else 'goal')
        self.log.info(f'[{self.name}] navigate ({what}) -> '
                      f'({c["x"]:.2f}, {c["y"]:.2f}, yaw {float(c.get("yaw", 0.0)):.2f})')
        self.send_action(task, self.nav_client, goal, 'navigate',
                         lambda ok, m: self.finish_task(task, ok, m))

    # ---------- dock ----------
    def cmd_dock(self, c):
        goal = self.make_nav_goal(c)
        task = self.start_task(c)
        self.nav_active = True
        self.stop_motion()

        def after_nav(ok, message):
            if not ok:
                self.finish_task(task, False, f'dock waypoint: {message}')
                return
            if not self.task_alive(task):
                return
            self.log.info(f'[{self.name}] dock waypoint reached -> docking')
            self.send_action(task, self.dock_client, Dock.Goal(), 'dock',
                             lambda ok2, m2: self.finish_task(task, ok2, m2))

        self.log.info(f'[{self.name}] dock: waypoint ({c["x"]:.2f}, {c["y"]:.2f}) then dock')
        self.send_action(task, self.nav_client, goal, 'navigate to dock waypoint', after_nav)

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
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.connected_timeout = float(g('connected_timeout'))
        self.drive_timeout = float(g('drive_timeout'))
        self.max_linear = abs(float(g('max_linear')))
        self.max_angular = abs(float(g('max_angular')))
        self.dock_status_wait = float(g('dock_status_wait'))

        names = [str(r).strip('/') for r in g('robots') if str(r).strip('/')]
        ips = list(g('robot_ips'))
        self.links = [RobotLink(self, n, ips[i] if i < len(ips) else '')
                      for i, n in enumerate(names)]
        self.create_timer(float(g('status_period')), self.publish_status)
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f'web relay started for {names}')

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
