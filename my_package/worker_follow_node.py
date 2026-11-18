#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[이 PC에서 실행] 웹캠 PC가 보낸 작업자 인식 결과로 작업자를 추종하는 노드 (TurtleBot4 + Nav2)

입력
  /<robot>/worker_detect/target  (std_msgs/Float32MultiArray, worker_detect_node.py 가 발행)
      [찾음, 중심x, 중심y, 폭, 높이(0~1), 점수, 거리, 거리출처, 영상폭, 영상높이, 등록여부]
      (앞 8칸은 tb_detect_node.py 형식과 동일)
  /<robot>/scan, /<robot>/tf, /<robot>/tf_static, /<robot>/dock_status, /<robot>/map(TF용 아님)
  웹 서버(turtlebot4_ws/web_server) 연동:
  /<robot>/user_register/reply   (std_msgs/String) 'y' → 추종 허용 / 'n' → 추종 끔
  /<robot>/user_register/retry   (std_msgs/Bool)   → 추종 끔 (재인식)
  /<robot>/user_register/cancel  (std_msgs/Bool)   → 즉시 정지·추종 끔 (취소·구역 이동·작업 종료)
  /<robot>/pause                 (std_msgs/Bool)   True → 즉시 정지 / False → 재개
                                 (일시정지·수동 조작·관제 정지/도킹/텔레옵·비상정지)

출력
  /<robot>/navigate_to_pose   (Nav2 액션: 작업자 뒤 target_distance 위치로 이동)
  /<robot>/cmd_vel_nav        (제자리 회전: Velocity Smoother → Collision Monitor 를 거침)
  /<robot>/undock             (기본: 언독 안 함. 웹에서 로봇 선택 시 web_relay 가 언독
                               auto_undock:=true 면 시작 시 도킹 상태일 때 직접 언독)

동작
  1. 터미널에서 1(robot6) / 2(robot11) 선택
  2. map 위치 인식(map->base_link TF 안정) → Nav2 준비 → 대기 (켜기만 해서는 언독 안 함)
     웹에서 로봇 선택 → web_relay 가 dock_status 확인 후 도킹이면 언독
     웹에서 작업자 '맞음'(reply 'y') + 일시정지 아님 + 도킹 아님 일 때만 추종
     (require_web_confirm:=false 면 웹 없이 바로 추종: 단독 테스트용)
  3. 바운딩박스 중심 방향(카메라 화각으로 계산)에 있는 LiDAR 클러스터만 작업자로 연결
     (카메라가 작업자라고 하지 않은 물체는 절대 추종하지 않음)
  4. LiDAR 거리로 거리 유지: 멀면 Nav2 접근, 걷는 중이면 예측 위치로 계속 추종
  5. 정지/대기 중에는 박스 중심이 화면 중앙에서 center_tolerance_px 이상 벗어나면
     작업자가 중앙에 오도록 제자리 회전 (Nav2 Goal 과 동시에 내지 않음)
  6. 부드러운 정지: 목표 거리 도달 시 Goal 을 취소하지 않고 Nav2 가 끝내게 두며,
     회전·정지 명령은 Velocity Smoother 를 거쳐 감속 (config/nav2_follow.yaml 의 감속 설정)

실행
  ros2 run my_package worker_follow all        # 두 로봇 모두: 웹에서 선택·'맞음' 된 로봇만 추종
                                               #   (켜기만 해서는 언독 안 함: 웹 선택 시 web_relay 가 언독)
  ros2 run my_package worker_follow            # 1 / 2 / a 입력
  ros2 run my_package worker_follow 2 --ros-args -p target_distance:=1.0
  # 웹 없이 단독 테스트 (worker_detect --no-web 과 함께):
  ros2 run my_package worker_follow 1 --ros-args -p require_web_confirm:=false -p auto_undock:=true
"""

import math
import os
import sys
import time
from collections import deque
from typing import Dict, Optional, Tuple

import rclpy
import rclpy.executors
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from rclpy.utilities import remove_ros_args

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped, Twist, TwistStamped
from irobot_create_msgs.action import Undock
from irobot_create_msgs.msg import DockStatus
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage
import tf2_ros
from tf2_ros import TransformException


# 터미널 번호 -> 로봇 (나중에 웹 클릭으로 대체)
ROBOTS = {
    '1': {'name': 'robot6', 'domain_id': None},   # None: 현재 셸의 ROS_DOMAIN_ID 사용
    '2': {'name': 'robot11', 'domain_id': None},
}

MODE_IDLE = 'IDLE'
MODE_APPROACH = 'APPROACH'
MODE_HOLD = 'HOLD'
MODE_TOO_CLOSE = 'TOO_CLOSE'

PHASE_WAIT_MAP = 'WAIT_MAP_WHILE_DOCKED'
PHASE_UNDOCKING = 'UNDOCKING'
PHASE_FOLLOW = 'FOLLOW'
PHASE_FAILED = 'STARTUP_FAILED'


def apply_transform_xyz(transform_stamped, point: Tuple[float, float, float]):
    """Apply a geometry_msgs TransformStamped to a 3D point."""
    px, py, pz = point
    tr = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    q_norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if not math.isfinite(q_norm) or q_norm < 1e-9:
        raise ValueError('TF quaternion is invalid or has zero length')
    qx, qy, qz, qw = q.x / q_norm, q.y / q_norm, q.z / q_norm, q.w / q_norm
    tx = 2.0 * (qy * pz - qz * py)
    ty = 2.0 * (qz * px - qx * pz)
    tz = 2.0 * (qx * py - qy * px)
    rx = px + qw * tx + (qy * tz - qz * ty)
    ry = py + qw * ty + (qz * tx - qx * tz)
    rz = pz + qw * tz + (qx * ty - qy * tx)
    return rx + tr.x, ry + tr.y, rz + tr.z


def yaw_from_transform(transform_stamped) -> float:
    q = transform_stamped.transform.rotation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class WorkerFollower(Node):
    def __init__(self, robot_ns: str = 'robot6', default_auto_undock: bool = False):
        ns = robot_ns.strip('/')
        super().__init__(f'worker_follow_{ns}')
        self.robot_ns = ns

        dp = self.declare_parameter
        # Topics / frames
        dp('scan_topic', f'/{ns}/scan')
        dp('tf_topic', f'/{ns}/tf')
        dp('tf_static_topic', f'/{ns}/tf_static')
        dp('nav2_action', f'/{ns}/navigate_to_pose')
        dp('worker_topic', f'/{ns}/worker_detect/target')
        dp('cmd_vel_topic', f'/{ns}/cmd_vel_nav')      # Velocity Smoother 입력
        dp('cmd_vel_stamped', True)                   # nav2.yaml enable_stamped_cmd_vel: true
        dp('undock_action', f'/{ns}/undock')
        dp('dock_status_topic', f'/{ns}/dock_status')
        dp('debug_target_topic', f'/{ns}/worker_follow/target')
        dp('map_frame', 'map')
        dp('base_frame', 'base_link')

        # Web server (turtlebot4_ws/web_server/config.py 토픽과 같아야 함)
        dp('require_web_confirm', True)       # 웹 '맞음' 전에는 추종하지 않음
        dp('register_reply_topic', f'/{ns}/user_register/reply')
        dp('register_retry_topic', f'/{ns}/user_register/retry')
        dp('register_cancel_topic', f'/{ns}/user_register/cancel')
        dp('pause_topic', f'/{ns}/pause')

        # Webcam geometry (웹캠이 로봇 위에 정면으로 달려 있다고 가정)
        dp('camera_hfov_deg', 60.0)          # 웹캠 가로 화각 (실측 권장)
        dp('camera_yaw_offset_deg', 0.0)     # 웹캠이 로봇 정면에서 돌아간 각도 (왼쪽 +)
        dp('camera_x_offset', 0.0)           # base_link 기준 웹캠 위치 (m)
        dp('camera_y_offset', 0.0)
        dp('camera_mirrored', False)         # 좌우 반전 영상이면 true
        dp('image_width', 640)               # 메시지에 영상폭이 없을 때만 사용
        dp('min_match_score', 0.6)
        dp('camera_lost_timeout', 1.0)
        dp('camera_assoc_max_age', 0.5)

        # LiDAR clustering / association
        dp('min_range', 0.25)
        dp('max_range', 4.0)
        dp('front_angle_deg', 75.0)          # 획득 전 각도 제한 (base_link 기준)
        dp('track_angle_deg', 180.0)         # 고정 후 각도 제한
        dp('cluster_gap', 0.18)
        dp('min_cluster_points', 3)
        dp('min_cluster_width', 0.06)
        dp('max_cluster_width', 0.55)
        dp('leg_merge_distance', 0.45)
        dp('target_match_distance', 0.80)
        dp('assoc_angle_tol_deg', 8.0)
        dp('assoc_bbox_width_factor', 0.35)
        dp('assoc_angle_tol_max_deg', 15.0)
        dp('reassoc_mismatch_scans', 5)      # 카메라 방향과 계속 어긋나면 다시 연결
        dp('target_filter_alpha', 0.5)

        # Following
        dp('target_distance', 1.0)
        dp('approach_margin', 0.35)          # > Nav2 xy_goal_tolerance(0.25)
        dp('hold_margin', 0.05)
        dp('too_close_margin', 0.30)
        dp('target_moving_speed', 0.12)
        dp('velocity_window_sec', 0.8)
        dp('lead_time', 0.6)
        dp('max_lead_distance', 0.5)
        dp('moving_follow_margin', 0.10)
        dp('goal_update_distance', 0.30)
        dp('goal_update_period', 1.0)
        dp('goal_update_distance_moving', 0.15)
        dp('goal_update_period_moving', 0.4)

        # Centering rotation (bbox centre -> image centre)
        dp('center_tolerance_px', 20.0)      # 이 픽셀 이상 벗어나면 회전 시작
        dp('center_release_px', 8.0)         # 이 픽셀 안에 들어오면 회전 종료
        dp('rotation_kp', 0.004)             # rad/s per px
        dp('max_rotation_speed', 0.8)
        dp('min_rotation_speed', 0.12)
        dp('heading_max_age', 0.4)

        # Safety
        dp('scan_lost_timeout', 0.7)
        dp('target_lost_timeout', 1.5)
        dp('tf_lost_timeout', 1.5)
        dp('nav2_retry_delay', 1.0)
        dp('nav2_max_retry_delay', 5.0)

        # Startup (localize while docked -> undock)
        dp('auto_undock', default_auto_undock)   # 시작 시 도킹 상태면 언독 (이미 언독이면 건너뜀)
        dp('map_ready_stable_sec', 2.0)
        dp('wait_nav2_before_undock', True)
        dp('dock_status_wait_sec', 5.0)
        dp('undock_timeout', 60.0)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.scan_topic = str(g('scan_topic'))
        self.tf_topic = str(g('tf_topic'))
        self.tf_static_topic = str(g('tf_static_topic'))
        self.action_name = str(g('nav2_action'))
        self.worker_topic = str(g('worker_topic'))
        self.cmd_vel_topic = str(g('cmd_vel_topic'))
        self.cmd_vel_stamped = bool(g('cmd_vel_stamped'))
        self.undock_action = str(g('undock_action'))
        self.dock_status_topic = str(g('dock_status_topic'))
        self.debug_target_topic = str(g('debug_target_topic'))
        self.map_frame = str(g('map_frame'))
        self.base_frame = str(g('base_frame'))
        self.require_web_confirm = bool(g('require_web_confirm'))

        self.tan_half_hfov = math.tan(math.radians(float(g('camera_hfov_deg'))) / 2.0)
        self.cam_yaw_offset = math.radians(float(g('camera_yaw_offset_deg')))
        self.cam_xy = (float(g('camera_x_offset')), float(g('camera_y_offset')))
        self.cam_mirrored = bool(g('camera_mirrored'))
        self.image_width_default = int(g('image_width'))
        self.min_match_score = float(g('min_match_score'))
        self.camera_lost_timeout = float(g('camera_lost_timeout'))
        self.camera_assoc_max_age = float(g('camera_assoc_max_age'))

        self.min_range = float(g('min_range'))
        self.max_range = float(g('max_range'))
        self.front_angle = math.radians(float(g('front_angle_deg')))
        self.track_angle = math.radians(float(g('track_angle_deg')))
        self.cluster_gap = float(g('cluster_gap'))
        self.min_points = int(g('min_cluster_points'))
        self.min_width = float(g('min_cluster_width'))
        self.max_width = float(g('max_cluster_width'))
        self.leg_merge_distance = float(g('leg_merge_distance'))
        self.match_distance = float(g('target_match_distance'))
        self.assoc_angle_tol = math.radians(float(g('assoc_angle_tol_deg')))
        self.assoc_bbox_factor = float(g('assoc_bbox_width_factor'))
        self.assoc_angle_tol_max = math.radians(float(g('assoc_angle_tol_max_deg')))
        self.reassoc_mismatch_scans = int(g('reassoc_mismatch_scans'))
        self.filter_alpha = float(g('target_filter_alpha'))

        self.target_distance = float(g('target_distance'))
        self.approach_margin = float(g('approach_margin'))
        self.hold_margin = float(g('hold_margin'))
        self.too_close_margin = float(g('too_close_margin'))
        self.target_moving_speed = float(g('target_moving_speed'))
        self.velocity_window = float(g('velocity_window_sec'))
        self.lead_time = float(g('lead_time'))
        self.max_lead_distance = float(g('max_lead_distance'))
        self.moving_follow_margin = float(g('moving_follow_margin'))
        self.goal_update_distance = float(g('goal_update_distance'))
        self.goal_update_period = float(g('goal_update_period'))
        self.goal_update_distance_moving = float(g('goal_update_distance_moving'))
        self.goal_update_period_moving = float(g('goal_update_period_moving'))

        self.center_tol_px = float(g('center_tolerance_px'))
        self.center_release_px = float(g('center_release_px'))
        self.rotation_kp = float(g('rotation_kp'))
        self.max_rot = float(g('max_rotation_speed'))
        self.min_rot = float(g('min_rotation_speed'))
        self.heading_max_age = float(g('heading_max_age'))

        self.scan_lost_timeout = float(g('scan_lost_timeout'))
        self.target_lost_timeout = float(g('target_lost_timeout'))
        self.tf_lost_timeout = float(g('tf_lost_timeout'))
        self.nav2_retry_delay = float(g('nav2_retry_delay'))
        self.nav2_max_retry_delay = float(g('nav2_max_retry_delay'))

        self.auto_undock = bool(g('auto_undock'))
        self.map_ready_stable_sec = float(g('map_ready_stable_sec'))
        self.wait_nav2_before_undock = bool(g('wait_nav2_before_undock'))
        self.dock_status_wait_sec = float(g('dock_status_wait_sec'))
        self.undock_timeout = float(g('undock_timeout'))

        if self.target_distance <= 0.0 or self.max_range <= self.min_range:
            raise ValueError('target_distance / range parameters are invalid')
        if self.center_release_px > self.center_tol_px:
            raise ValueError('center_release_px must be <= center_tolerance_px')
        if not 0.0 < self.tan_half_hfov < 20.0:
            raise ValueError('camera_hfov_deg is invalid')

        # ---- TF: same manual /<ns>/tf subscription + latest-TF policy ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0), node=self)
        self.create_subscription(
            TFMessage, self.tf_topic, self.dynamic_tf_callback,
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100,
                       reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.VOLATILE))
        self.create_subscription(
            TFMessage, self.tf_static_topic, self.static_tf_callback,
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100,
                       reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, self.worker_topic, self.worker_callback, 10)
        self.create_subscription(DockStatus, self.dock_status_topic, self.dock_status_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(String, str(g('register_reply_topic')), self.reply_callback, 10)
        self.create_subscription(Bool, str(g('register_retry_topic')), self.retry_callback, 10)
        self.create_subscription(Bool, str(g('register_cancel_topic')), self.cancel_callback, 10)
        self.create_subscription(Bool, str(g('pause_topic')), self.pause_callback, 10)
        self.nav_client = ActionClient(self, NavigateToPose, self.action_name)
        self.undock_client = ActionClient(self, Undock, self.undock_action)
        self.cmd_pub = self.create_publisher(
            TwistStamped if self.cmd_vel_stamped else Twist, self.cmd_vel_topic, 10)
        self.debug_pub = self.create_publisher(PointStamped, self.debug_target_topic, 10)

        # ---- Web state ----
        self.web_enabled = not self.require_web_confirm   # 웹 '맞음' 받으면 True
        self.paused = False

        # ---- Camera (worker) state ----
        self.cam: Optional[Dict] = None          # latest FOUND worker detection
        self.last_cam_msg = 0.0                  # any message (camera alive)
        self.last_worker_seen: Optional[float] = None

        # ---- LiDAR target state ----
        self.prev_target_map: Optional[Tuple[float, float]] = None
        self.filtered_target: Optional[Tuple[float, float]] = None
        self.track: Optional[Dict] = None
        self.target_hist = deque()
        self.mismatch_count = 0

        # ---- Nav2 state ----
        self.goal_handle = None
        self.goal_active = False
        self.goal_request_pending = False
        self.cancel_pending = False
        self.goal_seq = 0
        self.want_goal = False
        self.goal_target_ref: Optional[Tuple[float, float]] = None
        self.last_goal_sent = 0.0
        self.last_goal_succeeded = False
        self.nav_fail_count = 0
        self.nav_retry_after = 0.0

        # ---- Mode / rotation ----
        self.mode = MODE_IDLE
        self.rotating = False
        self.centering = False

        # ---- Timing ----
        self.last_scan_received = 0.0
        self.last_target_seen: Optional[float] = None
        self.last_tf_success: Optional[float] = None
        self.last_tf_warning = 0.0
        self.last_lost_log = 0.0

        # ---- Startup ----
        self.phase = PHASE_WAIT_MAP
        self.node_start = time.monotonic()
        self.map_ready_since: Optional[float] = None
        self.is_docked: Optional[bool] = None
        self.undock_goal_handle = None
        self.undock_sent_at = 0.0

        self.create_timer(0.10, self.control_loop)

        log = self.get_logger().info
        log(f'Worker follower for {ns} started.')
        log(f'Worker input: {self.worker_topic} (Float32MultiArray from worker_detect_node)')
        log(f'Scan: {self.scan_topic} | TF: {self.tf_topic} | Nav2: {self.action_name}')
        log(f'Rotation cmd: {self.cmd_vel_topic} (through velocity_smoother + collision_monitor)')
        log(f'Startup: localize while docked, then undock via {self.undock_action} '
            f'(auto_undock={self.auto_undock})')
        log(f'Web confirm required: {self.require_web_confirm} '
            f'(reply/retry/cancel: /{ns}/user_register/*, pause: {g("pause_topic")})')

    # ================= TF =================
    def dynamic_tf_callback(self, msg: TFMessage):
        for t in msg.transforms:
            try:
                self.tf_buffer.set_transform(t, f'{self.robot_ns}_tf')
            except Exception as exc:
                self.get_logger().error(f'Dynamic TF insert failed: {exc}',
                                        throttle_duration_sec=2.0)

    def static_tf_callback(self, msg: TFMessage):
        for t in msg.transforms:
            try:
                self.tf_buffer.set_transform_static(t, f'{self.robot_ns}_tf_static')
            except Exception as exc:
                self.get_logger().error(f'Static TF insert failed: {exc}',
                                        throttle_duration_sec=2.0)

    def warn_tf(self, message: str):
        now = time.monotonic()
        if now - self.last_tf_warning >= 2.0:
            self.get_logger().warning(message)
            self.last_tf_warning = now

    # ================= Web server commands =================
    def set_web_enabled(self, enabled: bool, why: str):
        if enabled != self.web_enabled:
            self.get_logger().info(f'Web: following {"ENABLED" if enabled else "DISABLED"} ({why})')
        self.web_enabled = enabled
        if not enabled:
            self.halt()

    def halt(self):
        """웹 명령으로 즉시 정지 (다음 control_loop 를 기다리지 않음)."""
        self.reset_association()
        self.cam = None
        self.last_worker_seen = None
        self.stop_all()

    def reply_callback(self, msg: String):
        r = msg.data.strip().lower()
        if r == 'y':
            self.set_web_enabled(True, "worker confirmed 'y'")
        elif r == 'n':
            self.set_web_enabled(False, "worker rejected 'n'")

    def retry_callback(self, msg: Bool):
        if msg.data:
            self.set_web_enabled(False, 'recognition retry')

    def cancel_callback(self, msg: Bool):
        if msg.data:
            self.set_web_enabled(False, 'recognition cancel / zone move / job end')

    def pause_callback(self, msg: Bool):
        paused = bool(msg.data)
        if paused != self.paused:
            self.get_logger().info(f'Web: {"PAUSED" if paused else "RESUMED"}')
        self.paused = paused
        if paused:
            self.halt()

    def follow_allowed(self) -> Optional[str]:
        """None 이면 추종 가능, 아니면 멈춰 있는 이유."""
        if not self.web_enabled:
            return 'waiting for worker confirmation on web'
        if self.paused:
            return 'paused by web server'
        if self.is_docked:
            return 'robot is docked'
        return None

    # ================= Worker (camera) input =================
    def worker_callback(self, msg: Float32MultiArray):
        now = time.monotonic()
        d = list(msg.data)
        if len(d) < 6:
            self.get_logger().warning(f'Worker message too short ({len(d)})',
                                      throttle_duration_sec=5.0)
            return
        self.last_cam_msg = now
        found = d[0] >= 0.5
        registered = d[10] >= 0.5 if len(d) >= 11 else True
        if not (found and registered) or d[5] < self.min_match_score:
            return  # not seen in this frame -> self.cam ages out
        cx, cy, w, h = d[1], d[2], d[3], d[4]
        if not all(math.isfinite(v) for v in (cx, cy, w, h)) or not 0.0 <= cx <= 1.0:
            return
        img_w = int(d[8]) if len(d) >= 9 and d[8] > 0 else self.image_width_default
        self.cam = {'cx': cx, 'cy': cy, 'w': w, 'h': h, 'score': d[5],
                    'img_w': img_w, 'recv': now}
        self.last_worker_seen = now

    def camera_bearing(self, cam) -> Tuple[float, float]:
        """(bearing in base_link [rad, left +], half bbox width angle)."""
        x = cam['cx'] - 0.5
        if self.cam_mirrored:
            x = -x
        bearing = math.atan(-2.0 * x * self.tan_half_hfov) + self.cam_yaw_offset
        half_w = math.atan(max(cam['w'], 0.0) * self.tan_half_hfov)
        return bearing, half_w

    def pixel_error(self, now) -> Optional[float]:
        cam = self.cam
        if cam is None or now - cam['recv'] > self.heading_max_age:
            return None
        err = (cam['cx'] - 0.5) * cam['img_w']   # + : worker right of centre
        return -err if self.cam_mirrored else err

    # ================= Startup: localize while docked -> undock =================
    def dock_status_callback(self, msg: DockStatus):
        docked = bool(msg.is_docked)
        if self.is_docked is None or docked != self.is_docked:
            self.get_logger().info(f'Dock status: {"DOCKED" if docked else "UNDOCKED"}')
        self.is_docked = docked

    def map_localized(self, now) -> bool:
        try:
            self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except TransformException:
            self.map_ready_since = None
            self.get_logger().info(
                f'Waiting for map localization ({self.map_frame}->{self.base_frame}) while '
                'docked... (localization running? initial pose set?)', throttle_duration_sec=3.0)
            return False
        if self.map_ready_since is None:
            self.map_ready_since = now
            self.get_logger().info('Map pose available; checking it is stable...')
        return now - self.map_ready_since >= self.map_ready_stable_sec

    def startup_step(self, now) -> bool:
        if self.phase == PHASE_FOLLOW:
            return True
        if self.phase == PHASE_FAILED:
            self.get_logger().error('Startup failed (undock). Robot stays stopped; restart.',
                                    throttle_duration_sec=5.0)
            return False
        if self.phase == PHASE_UNDOCKING:
            if now - self.undock_sent_at > self.undock_timeout:
                self.get_logger().error('Undock timed out.')
                self.cancel_undock()
                self.phase = PHASE_FAILED
            return False
        if not self.map_localized(now):
            return False
        if self.wait_nav2_before_undock and not self.nav_client.server_is_ready():
            self.get_logger().info(f'Map localized; waiting for Nav2 ({self.action_name})...',
                                   throttle_duration_sec=3.0)
            return False
        if not self.auto_undock:
            self.get_logger().info('auto_undock=false; following enabled.')
            self.phase = PHASE_FOLLOW
            return True
        if self.is_docked is None:
            # 도킹 여부를 모르면 언독하지 않고 기다림 (언독 상태에서 언독 → 실패 방지)
            if now - self.node_start >= self.dock_status_wait_sec:
                self.get_logger().warning(
                    f'No dock status yet on {self.dock_status_topic}; waiting...',
                    throttle_duration_sec=3.0)
            return False
        if self.is_docked is False:
            self.get_logger().info('Robot is already undocked; skipping undock.')
            self.phase = PHASE_FOLLOW
            return True
        if not self.undock_client.server_is_ready():
            self.get_logger().warning(f'Undock server not ready: {self.undock_action}',
                                      throttle_duration_sec=3.0)
            return False
        self.phase = PHASE_UNDOCKING
        self.undock_sent_at = now
        self.get_logger().info(f'Map localized while docked -> undocking {self.robot_ns}')
        self.undock_client.send_goal_async(Undock.Goal()).add_done_callback(self.undock_response)
        return False

    def undock_response(self, future):
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'Undock request failed: {exc}')
            self.phase = PHASE_FAILED
            return
        if not handle.accepted:
            self.get_logger().error('Undock goal rejected.')
            self.phase = PHASE_FAILED
            return
        self.undock_goal_handle = handle
        handle.get_result_async().add_done_callback(self.undock_result)

    def undock_result(self, future):
        self.undock_goal_handle = None
        if self.phase != PHASE_UNDOCKING:
            return
        try:
            res = future.result()
            ok = res.status == GoalStatus.STATUS_SUCCEEDED and not res.result.is_docked
        except Exception as exc:
            self.get_logger().error(f'Undock result error: {exc}')
            ok = False
        if ok:
            self.get_logger().info('Undock complete -> worker following ENABLED.')
            self.reset_association()
            self.phase = PHASE_FOLLOW
        else:
            self.get_logger().error('Undock did not succeed.')
            self.phase = PHASE_FAILED

    def cancel_undock(self):
        if self.undock_goal_handle is not None:
            try:
                self.undock_goal_handle.cancel_goal_async()
            except Exception:
                pass
            self.undock_goal_handle = None

    # ================= LiDAR =================
    def make_clusters(self, scan: LaserScan):
        clusters, current, previous = [], [], None
        for i, r in enumerate(scan.ranges):
            valid = (math.isfinite(r) and scan.range_min <= r <= scan.range_max
                     and self.min_range <= r <= self.max_range)
            if not valid:
                if current:
                    clusters.append(current)
                    current = []
                previous = None
                continue
            a = scan.angle_min + i * scan.angle_increment
            p = (r * math.cos(a), r * math.sin(a))
            if previous is not None and math.hypot(p[0] - previous[0],
                                                   p[1] - previous[1]) > self.cluster_gap:
                if current:
                    clusters.append(current)
                current = []
            current.append(p)
            previous = p
        if current:
            clusters.append(current)
        return clusters

    def get_candidates(self, clusters, base_from_scan, map_from_base, angle_limit):
        out = []
        for cl in clusters:
            if len(cl) < self.min_points:
                continue
            xs = [p[0] for p in cl]
            ys = [p[1] for p in cl]
            width = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            if not self.min_width <= width <= self.max_width:
                continue
            sx, sy = sum(xs) / len(xs), sum(ys) / len(ys)
            bx, by, _ = apply_transform_xyz(base_from_scan, (sx, sy, 0.0))
            if abs(math.atan2(by, bx)) > angle_limit:   # filter in base_link frame
                continue
            mx, my, _ = apply_transform_xyz(map_from_base, (bx, by, 0.0))
            out.append({'bx': bx, 'by': by, 'mx': mx, 'my': my, 'n': 1})
        return out

    def merge_legs(self, cands):
        merged = []
        for c in sorted(cands, key=lambda c: math.hypot(c['bx'], c['by'])):
            for m in merged:
                if math.hypot(c['bx'] - m['bx'], c['by'] - m['by']) <= self.leg_merge_distance:
                    n = m['n']
                    for k in ('bx', 'by', 'mx', 'my'):
                        m[k] = (m[k] * n + c[k]) / (n + 1)
                    m['n'] = n + 1
                    break
            else:
                merged.append(dict(c))
        return merged

    def bearing_error(self, c, bearing):
        ox, oy = self.cam_xy
        return abs(wrap_angle(math.atan2(c['by'] - oy, c['bx'] - ox) - bearing))

    def reset_association(self):
        self.prev_target_map = None
        self.filtered_target = None
        self.track = None
        self.last_target_seen = None
        self.goal_target_ref = None
        self.target_hist.clear()
        self.mismatch_count = 0

    def scan_callback(self, scan: LaserScan):
        now = time.monotonic()
        self.last_scan_received = now
        if self.phase != PHASE_FOLLOW or self.follow_allowed() is not None:
            return
        try:
            cam = self.cam
            if cam is None or now - cam['recv'] > self.camera_lost_timeout:
                return   # no worker from camera -> follow nothing
            scan_frame = scan.header.frame_id.strip()
            if not scan_frame:
                self.warn_tf('LaserScan header.frame_id is empty.')
                return
            try:
                base_from_scan = self.tf_buffer.lookup_transform(
                    self.base_frame, scan_frame, Time())
                map_from_base = self.tf_buffer.lookup_transform(
                    self.map_frame, self.base_frame, Time())
            except TransformException as exc:
                self.warn_tf(f'TF chain not ready ({self.map_frame} <- {self.base_frame} '
                             f'<- {scan_frame}): {exc}')
                return

            tracking = self.prev_target_map is not None
            cands = self.get_candidates(self.make_clusters(scan), base_from_scan, map_from_base,
                                        self.track_angle if tracking else self.front_angle)
            cam_fresh = now - cam['recv'] <= self.camera_assoc_max_age
            bearing, half_w = self.camera_bearing(cam)
            tol = min(self.assoc_angle_tol + self.assoc_bbox_factor * half_w,
                      self.assoc_angle_tol_max)

            if not tracking:
                # Acquire ONLY on the camera ray of the worker; nearest along the ray
                # (what the camera sees in front, not the wall behind).
                if not cam_fresh:
                    return
                gated = [c for c in cands if self.bearing_error(c, bearing) <= tol]
                merged = self.merge_legs(gated)
                if not merged:
                    return
                target = min(merged, key=lambda m: math.hypot(m['bx'], m['by']))
                self.get_logger().info(
                    f'Worker associated with LiDAR cluster at '
                    f'{math.hypot(target["bx"], target["by"]):.2f} m, '
                    f'{math.degrees(math.atan2(target["by"], target["bx"])):.0f} deg')
            else:
                px, py = self.prev_target_map
                gated = [c for c in cands
                         if math.hypot(c['mx'] - px, c['my'] - py) <= self.match_distance]
                merged = self.merge_legs(gated)
                if not merged:
                    return
                target = min(merged, key=lambda m: math.hypot(m['mx'] - px, m['my'] - py))
                # The camera is the authority: if the tracked cluster keeps
                # disagreeing with the bbox direction, drop it and re-associate.
                if cam_fresh:
                    if self.bearing_error(target, bearing) > 2.0 * tol:
                        self.mismatch_count += 1
                    else:
                        self.mismatch_count = 0
                    if self.mismatch_count >= self.reassoc_mismatch_scans:
                        self.get_logger().warning(
                            'Tracked LiDAR cluster disagrees with camera bbox; re-associating.')
                        self.reset_association()
                        return

            rx = map_from_base.transform.translation.x
            ry = map_from_base.transform.translation.y
            ryaw = yaw_from_transform(map_from_base)
            tx, ty = target['mx'], target['my']
            if not all(math.isfinite(v) for v in (tx, ty, rx, ry, ryaw)):
                self.warn_tf('Non-finite map coordinate; skipping scan.')
                return
            self.last_tf_success = now

            if self.filtered_target is None:
                self.filtered_target = (tx, ty)
            else:
                a = self.filter_alpha
                fx, fy = self.filtered_target
                self.filtered_target = (a * tx + (1 - a) * fx, a * ty + (1 - a) * fy)
            self.prev_target_map = (tx, ty)
            self.last_target_seen = now

            # Walking speed of the worker (map frame)
            self.target_hist.append((now, tx, ty))
            while now - self.target_hist[0][0] > self.velocity_window:
                self.target_hist.popleft()
            t0, x0, y0 = self.target_hist[0]
            vel = ((tx - x0) / (now - t0), (ty - y0) / (now - t0)) \
                if len(self.target_hist) >= 3 and now - t0 >= 0.3 else (0.0, 0.0)

            self.track = {
                'map': self.filtered_target,
                'robot': (rx, ry, ryaw),
                'range': math.hypot(target['bx'], target['by']),
                'vel': vel,
                'moving': math.hypot(*vel) > self.target_moving_speed,
            }
            pt = PointStamped()
            pt.header.frame_id = self.map_frame
            pt.header.stamp = self.get_clock().now().to_msg()
            pt.point.x, pt.point.y = self.filtered_target
            self.debug_pub.publish(pt)
            self.get_logger().info(
                f'Worker range={self.track["range"]:.2f} m '
                f'speed={math.hypot(*vel):.2f} m/s mode={self.mode}',
                throttle_duration_sec=1.0)
        except Exception as exc:
            self.get_logger().error(f'Scan callback error (scan skipped): {exc}',
                                    throttle_duration_sec=2.0)

    # ================= Control =================
    def control_loop(self):
        now = time.monotonic()
        if not self.startup_step(now):
            self.stop_all()
            return
        if self.last_scan_received == 0.0:
            return

        scan_lost = now - self.last_scan_received > self.scan_lost_timeout
        cam_dead = now - self.last_cam_msg > self.camera_lost_timeout
        worker_lost = (self.last_worker_seen is None
                       or now - self.last_worker_seen > self.camera_lost_timeout)
        target_lost = (self.last_target_seen is not None
                       and now - self.last_target_seen > self.target_lost_timeout)
        tf_lost = (self.last_target_seen is not None and
                   (self.last_tf_success is None
                    or now - self.last_tf_success > self.tf_lost_timeout))

        reason = self.follow_allowed()
        if reason is not None:
            pass
        elif scan_lost:
            reason = 'scan timeout'
        elif cam_dead:
            reason = f'no messages on {self.worker_topic}'
        elif worker_lost:
            reason = 'worker not seen by camera'
        elif tf_lost:
            reason = 'TF timeout'
        if reason is not None:
            self.reset_association()
            if (self.goal_active or self.rotating) and now - self.last_lost_log > 2.0:
                self.get_logger().warning(f'{reason}; stopping (smooth).')
                self.last_lost_log = now
            elif now - self.last_lost_log > 5.0:
                self.get_logger().info(f'Waiting: {reason}')
                self.last_lost_log = now
            self.stop_all()
            return

        if target_lost:
            self.get_logger().warning('LiDAR lost the worker; re-associating from camera.')
            self.reset_association()

        track = self.track
        if track is None:
            # Camera sees the worker but LiDAR has no match yet: do not drive,
            # but keep the worker centred by rotating in place.
            self.want_goal = False
            self.cancel_current_goal()
            self.mode = MODE_IDLE
            self.handle_centering(now)
            return

        self.update_mode(track['range'], track['moving'])
        if self.mode == MODE_APPROACH:
            self.stop_rotation()
            self.handle_approach(track, now, allow_new=True)
        elif self.mode == MODE_HOLD:
            # Do NOT cancel: let Nav2 finish the goal (smooth stop). Only fix a
            # goal that still contains the walking lead after the worker stopped.
            if self.goal_active:
                self.handle_approach(track, now, allow_new=False)
            self.handle_centering(now)
        elif self.mode == MODE_TOO_CLOSE:
            self.want_goal = False
            self.cancel_current_goal()   # smoother decelerates
            self.handle_centering(now)

    def update_mode(self, d, moving):
        td = self.target_distance
        prev = self.mode
        if d < td - self.too_close_margin:
            self.mode = MODE_TOO_CLOSE
        elif self.mode == MODE_TOO_CLOSE and d < td - self.too_close_margin + 0.10:
            pass
        elif d > td + self.approach_margin or (moving and d > td + self.moving_follow_margin):
            self.mode = MODE_APPROACH
        elif (self.mode == MODE_APPROACH and not moving
              and (d <= td + self.hold_margin or self.last_goal_succeeded)):
            self.mode = MODE_HOLD
        elif self.mode in (MODE_IDLE, MODE_TOO_CLOSE):
            self.mode = MODE_HOLD
        if prev != self.mode:
            self.last_goal_succeeded = False
            self.get_logger().info(f'Mode {prev} -> {self.mode} (distance {d:.2f} m)')

    def handle_approach(self, track, now, allow_new):
        tx, ty = track['map']
        rx, ry, _ = track['robot']
        if track['moving']:
            vx, vy = track['vel']
            lx, ly = vx * self.lead_time, vy * self.lead_time
            lead = math.hypot(lx, ly)
            if lead > self.max_lead_distance > 0.0:
                lx, ly = lx * self.max_lead_distance / lead, ly * self.max_lead_distance / lead
            tx, ty = tx + lx, ty + ly
            upd_d, upd_t = self.goal_update_distance_moving, self.goal_update_period_moving
        else:
            upd_d, upd_t = self.goal_update_distance, self.goal_update_period

        if self.goal_active and self.goal_target_ref is not None:
            moved = math.hypot(tx - self.goal_target_ref[0], ty - self.goal_target_ref[1])
            if moved < upd_d or now - self.last_goal_sent < upd_t:
                return
        elif not allow_new:
            return

        goal = self.create_goal(tx, ty, rx, ry)
        if goal is None:
            if self.goal_active and not allow_new:
                self.cancel_current_goal()   # already close enough; smoother decelerates
            return
        if self.nav2_ready(now):
            self.send_goal(goal, (tx, ty))

    def create_goal(self, tx, ty, rx, ry) -> Optional[PoseStamped]:
        dx, dy = tx - rx, ty - ry
        dist = math.hypot(dx, dy)
        if dist <= self.target_distance or dist < 1e-3:
            return None
        travel = dist - self.target_distance
        gx, gy = rx + dx / dist * travel, ry + dy / dist * travel
        yaw = math.atan2(ty - gy, tx - gx)       # face the worker
        if not all(math.isfinite(v) for v in (gx, gy, yaw)):
            return None
        goal = PoseStamped()
        goal.header.frame_id = self.map_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y = gx, gy
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        return goal

    # ---------- Centering rotation (mutually exclusive with Nav2) ----------
    def handle_centering(self, now):
        if self.goal_active or self.goal_request_pending or self.cancel_pending:
            self.stop_rotation()
            return
        err = self.pixel_error(now)
        if err is None:
            self.centering = False
            self.stop_rotation()
            return
        if self.centering:
            if abs(err) <= self.center_release_px:
                self.centering = False
        elif abs(err) > self.center_tol_px:
            self.centering = True
        if not self.centering:
            self.stop_rotation()
            return
        w = -self.rotation_kp * err          # worker left (err<0) -> turn left (+z)
        w = max(-self.max_rot, min(self.max_rot, w))
        if abs(w) < self.min_rot:
            w = math.copysign(self.min_rot, w)
        self.publish_cmd(0.0, w)
        self.rotating = True

    def publish_cmd(self, vx, wz):
        if self.cmd_vel_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.base_frame
            msg.twist.linear.x, msg.twist.angular.z = float(vx), float(wz)
        else:
            msg = Twist()
            msg.linear.x, msg.angular.z = float(vx), float(wz)
        self.cmd_pub.publish(msg)

    def stop_rotation(self):
        if self.rotating:
            self.publish_cmd(0.0, 0.0)   # velocity_smoother ramps this down
            self.rotating = False

    def stop_all(self):
        self.want_goal = False
        self.mode = MODE_IDLE
        self.centering = False
        self.stop_rotation()
        self.cancel_current_goal()

    # ================= Nav2 action management =================
    def nav2_ready(self, now) -> bool:
        if self.goal_request_pending or self.cancel_pending or now < self.nav_retry_after:
            return False
        if not self.nav_client.server_is_ready():
            self.get_logger().warning(f'Nav2 action server not ready: {self.action_name}',
                                      throttle_duration_sec=3.0)
            return False
        return True

    def send_goal(self, goal: PoseStamped, target_ref):
        self.stop_rotation()
        msg = NavigateToPose.Goal()
        msg.pose = goal
        self.goal_seq += 1
        seq = self.goal_seq
        self.goal_request_pending = True
        self.want_goal = True
        self.goal_target_ref = target_ref
        self.last_goal_sent = time.monotonic()
        # A new goal preempts the running one in bt_navigator (no stop-and-go).
        self.nav_client.send_goal_async(msg).add_done_callback(
            lambda f, s=seq: self.goal_response(f, s))
        self.get_logger().info(
            f'Nav2 goal: ({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f})',
            throttle_duration_sec=1.0)

    def drop_old_goal(self):
        if self.goal_active and self.goal_handle is not None:
            try:
                self.goal_handle.cancel_goal_async()
            except Exception:
                pass
        self.goal_active = False
        self.goal_handle = None
        self.goal_target_ref = None

    def goal_response(self, future, seq):
        self.goal_request_pending = False
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'Goal request failed: {exc}')
            self.drop_old_goal()
            self.register_nav_failure()
            return
        if not handle.accepted:
            self.get_logger().warning('Nav2 rejected goal.')
            self.drop_old_goal()
            self.register_nav_failure()
            return
        self.goal_handle = handle
        self.goal_active = True
        handle.get_result_async().add_done_callback(lambda f, s=seq: self.goal_result(f, s))
        now = time.monotonic()
        fresh = (self.last_target_seen is not None
                 and now - self.last_target_seen <= self.target_lost_timeout
                 and now - self.last_scan_received <= self.scan_lost_timeout)
        if not (fresh and self.want_goal and seq == self.goal_seq):
            self.get_logger().warning('Late goal acceptance after stop; cancelling.')
            self.cancel_current_goal()

    def goal_result(self, future, seq):
        if seq != self.goal_seq:
            return   # result of a preempted older goal
        status = None
        try:
            status = future.result().status
        except Exception as exc:
            self.get_logger().warning(f'Nav2 result error: {exc}')
        self.goal_active = False
        self.goal_handle = None
        self.goal_target_ref = None
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.nav_fail_count = 0
            self.last_goal_succeeded = True
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warning('Nav2 aborted the goal.')
            self.register_nav_failure()

    def register_nav_failure(self):
        self.nav_fail_count += 1
        delay = min(self.nav2_retry_delay * self.nav_fail_count, self.nav2_max_retry_delay)
        self.nav_retry_after = time.monotonic() + delay
        self.get_logger().warning(f'Nav2 failure #{self.nav_fail_count}; retry in {delay:.1f}s.')

    def cancel_current_goal(self):
        if not self.goal_active or self.goal_handle is None or self.cancel_pending:
            return
        self.cancel_pending = True
        try:
            self.goal_handle.cancel_goal_async().add_done_callback(self.cancel_done)
        except Exception as exc:
            self.cancel_pending = False
            self.get_logger().error(f'Cancel request could not be sent: {exc}')

    def cancel_done(self, future):
        self.cancel_pending = False
        try:
            future.result()
        except Exception as exc:
            self.get_logger().warning(f'Cancel request failed: {exc}')


# ================= main =================
def select_robots(argv) -> Optional[list]:
    """CLI argument (1/2/robot6/robot11/all) or terminal prompt -> list of robots."""
    by_name = {r['name']: r for r in ROBOTS.values()}

    def parse(c):
        c = c.strip()
        if c.lower() in ('a', 'all'):
            return list(ROBOTS.values())
        robot = ROBOTS.get(c) or by_name.get(c)
        return [robot] if robot else None

    user_args = remove_ros_args(args=argv)[1:]
    if user_args:
        robots = parse(user_args[0])
        if robots is None:
            print(f'Unknown robot "{user_args[0]}". Use 1, 2, all, or {list(by_name)}')
        return robots
    if not sys.stdin.isatty():
        print('No terminal; pass the robot, e.g. "ros2 run my_package worker_follow all"')
        return None
    menu = '  '.join(f'[{k}] {r["name"]}' for k, r in ROBOTS.items()) + '  [a] all'
    while True:
        try:
            c = input(f'Select robot -> {menu}  (q: quit): ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if c.lower() == 'q':
            return None
        robots = parse(c)
        if robots:
            return robots
        print('Please enter 1, 2 or a.')


def main(args=None):
    argv = sys.argv if args is None else args
    robots = select_robots(argv)
    if not robots:
        return
    multi = len(robots) > 1
    for r in robots:
        if r['domain_id'] is not None:
            os.environ['ROS_DOMAIN_ID'] = str(r['domain_id'])
    print(f'>>> Selected {[r["name"] for r in robots]} '
          f'(ROS_DOMAIN_ID={os.environ.get("ROS_DOMAIN_ID", "0")})', flush=True)
    rclpy.init(args=argv)
    nodes = []
    executor = rclpy.executors.SingleThreadedExecutor()
    try:
        for r in robots:
            # 켜기만 해서는 언독하지 않음 (웹에서 로봇 선택 → web_relay 가 언독)
            node = WorkerFollower(r['name'])
            nodes.append(node)
            executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for node in nodes:
            try:
                node.cancel_undock()
                node.stop_rotation()
                node.cancel_current_goal()
            except Exception:
                pass
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
