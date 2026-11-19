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

터틀봇 추종 모드 (robot): B로봇 호출로 A 앞에 도착한 로봇 B 가 로봇 A 를 따라감
  인식: ros2 run my_package worker_detect tb      (B 의 OAK-D 로 A 인식 → /<B>/tb_detect/target)
  추종: ros2 run my_package worker_follow robot all
  - 호출 결과(web_feedback ok)가 오면 시작, A 의 작업 종료·인식 취소·구역 이동·도킹 또는
    B 에 다른 이동/도킹/언독 명령이 오면 멈춤 (RobotFollower 클래스 설명 참고)
  - 노드 이름 robot_follow_<로봇>, 정지 기록 ~/follow_stop_log_<로봇>_robot.csv
"""

import math
import os
import sys
import time
import csv
import json
import datetime
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
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, LaserScan
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
    def __init__(self, robot_ns: str = 'robot6', default_auto_undock: bool = False,
                 node_prefix: str = 'worker_follow', param_defaults: Optional[Dict] = None):
        ns = robot_ns.strip('/')
        super().__init__(f'{node_prefix}_{ns}')
        self.robot_ns = ns

        # param_defaults: 기본값만 바꿈 (터틀봇 추종용). --ros-args -p 로 준 값이 항상 우선.
        defaults = param_defaults or {}
        dp = lambda name, value: self.declare_parameter(name, defaults.get(name, value))  # noqa: E731
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
        # 작업자 소실 복구: 이 시간 이상 못 보면 마지막 좌표로 이동 → 천천히 한 바퀴 → 못 찾으면 일시정지
        dp('lost_recovery_enabled', True)
        dp('lost_recovery_after_sec', 5.0)
        dp('lost_spin_speed', 0.3)           # rad/s (한 바퀴 약 21초)
        dp('lost_move_timeout', 40.0)        # 마지막 좌표 이동 최대 시간 (초)
        dp('pause_topic', f'/{ns}/pause')

        # Webcam geometry (웹캠이 로봇 위에 정면으로 달려 있다고 가정)
        dp('camera_hfov_deg', 50.0)          # 웹캠 가로 화각 (50도로 변경)
        dp('camera_yaw_offset_deg', 0.0)     # 웹캠이 로봇 정면에서 돌아간 각도 (왼쪽 +)
        dp('camera_x_offset', 0.0)           # base_link 기준 웹캠 위치 (m)
        dp('camera_y_offset', 0.0)
        dp('camera_mirrored', False)         # 좌우 반전 영상이면 true
        dp('image_width', 640)               # 메시지에 영상폭이 없을 때만 사용
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
        self.lost_recovery_enabled = bool(g('lost_recovery_enabled'))
        self.lost_recovery_after = float(g('lost_recovery_after_sec'))
        self.lost_spin_speed = abs(float(g('lost_spin_speed')))
        self.lost_move_timeout = float(g('lost_move_timeout'))
        self.recover = None              # None | {'stage': 'move'|'spin', ...}
        self.recover_done_for = None     # 이미 복구를 시도한 소실 (last_worker_seen 값)
        self.last_worker_map = None      # 마지막으로 추종한 작업자 map 좌표 (복구용, 연결 초기화와 별개)
        self.lost_paused = False         # 한 바퀴 탐색 실패 → 일시정지 (resume / y 로 해제)
        # 웹·콘솔 '재개'(pause false) 를 직접 받아 탐색 실패 일시정지를 푼다
        self.create_subscription(String, f'/{ns}/web_command', self.web_command_callback, 10)
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

        # ---- 정지 원인 기록: odom 으로 '실제로 움직이다 멈춘 순간'을 잡아 그때의 원인을 남긴다 ----
        self.stop_cause = ('시작 대기', '')          # (분류, 상세) — control_loop 가 매 주기 갱신
        self.stop_snapshot = None                    # 감속 시작 시점의 원인 (정지 확정 때 기록)
        self.motion_state = 'stopped'
        self.moving_since = self.quiet_since = self.motion_started = None
        self.stop_events = []
        self.stop_log_path = os.path.expanduser(f'~/follow_stop_log_{ns}.csv')
        self.create_subscription(Odometry, f'/{ns}/odom', self.odom_stop_callback,
                                 qos_profile_sensor_data)
        self.get_logger().info(f'Stop-cause log: {self.stop_log_path}')

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
            self.clear_lost_pause("worker confirmed 'y'")
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
            if not paused:
                self.clear_lost_pause('web resume')
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
        # 점수 기준(min_match_score)은 사용하지 않음: 같은 사람 판단은 인식 노드(MATCH_THRESHOLD)가 함
        if not (found and registered):
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

            self.last_worker_map = self.filtered_target
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
    # ================= 정지 원인 기록 =================
    @staticmethod
    def classify_block(reason, scan_lost, cam_dead, worker_lost, tf_lost):
        if reason.startswith('waiting for worker confirmation'):
            return ('작업자 확인 전 (y 전 / n / 재인식)', reason)
        if reason.startswith('paused after lost-worker'):
            return ('대상 소실 - 한 바퀴 탐색 실패 → 일시정지', reason)
        if reason.startswith('paused'):
            return ('일시정지 (웹·콘솔 pause/수동조작)', reason)
        if reason.startswith('robot is docked'):
            return ('도크 위', reason)
        if scan_lost or tf_lost:
            return ('센서·TF 끊김', reason)
        if cam_dead:
            return ('카메라 메시지 없음 (인식 노드 끊김)', reason)
        if worker_lost:
            return ('대상 소실 - 카메라', '카메라에서 등록 작업자를 못 찾음')
        return ('기타', reason)

    def odom_stop_callback(self, msg: Odometry):
        now = time.monotonic()
        v = msg.twist.twist
        lin = abs(v.linear.x)
        if lin > 0.05:                                   # 직진 중 (제자리 회전은 제외)
            self.quiet_since = None
            self.stop_snapshot = None
            if self.moving_since is None:
                self.moving_since = now
            if self.motion_state == 'stopped' and now - self.moving_since >= 0.3:
                self.motion_state, self.motion_started = 'moving', now
            return
        self.moving_since = None
        if lin < 0.02:
            if self.quiet_since is None:
                self.quiet_since = now
                self.stop_snapshot = self.stop_cause      # 감속이 끝난 시점의 원인
            if self.motion_state == 'moving' and now - self.quiet_since >= 0.5:
                self.motion_state = 'stopped'
                self.record_stop(now)
        elif self.stop_snapshot is None:
            self.stop_snapshot = self.stop_cause          # 감속 중

    def record_stop(self, now):
        cat, detail = self.stop_snapshot or self.stop_cause
        dist = f'{self.track["range"]:.2f}' if self.track is not None else ''
        cam_age = (f'{now - self.last_worker_seen:.1f}' if self.last_worker_seen is not None else '')
        moved = f'{(self.quiet_since or now) - (self.motion_started or now):.1f}'
        row = [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.robot_ns, cat, detail,
               dist, f'{self.target_distance:.2f}', cam_age, self.mode, moved]
        self.stop_events.append(row)
        self.get_logger().warning(
            f'[정지] {cat} | {detail} | 거리 {dist or "-"} m | 카메라 마지막 인식 {cam_age or "-"} 초 전 '
            f'| 모드 {self.mode} | 주행 {moved} 초 후')
        try:
            new = not os.path.exists(self.stop_log_path)
            with open(self.stop_log_path, 'a', newline='') as fp:
                w = csv.writer(fp)
                if new:
                    w.writerow(['time', 'robot', 'cause', 'detail', 'lidar_distance_m',
                                'target_distance_m', 'camera_last_seen_s', 'mode', 'moved_s'])
                w.writerow(row)
        except OSError as exc:
            self.get_logger().error(f'stop log write failed: {exc}')

    def print_stop_summary(self):
        if not self.stop_events:
            return
        counts = {}
        for r in self.stop_events:
            counts[r[2]] = counts.get(r[2], 0) + 1
        lines = [f'\n===== [{self.robot_ns}] 이번 실행 정지 원인 ({len(self.stop_events)}회) =====',
                 '| 원인 | 횟수 |', '|---|---|']
        lines += [f'| {k} | {v} |' for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
        lines.append(f'전체 기록: {self.stop_log_path}  (표: ros2 run my_package stop_report)')
        print('\n'.join(lines), flush=True)

    # ================= 작업자 소실 복구 =================
    def web_command_callback(self, msg: String):
        try:
            c = json.loads(msg.data)
        except ValueError:
            return
        if isinstance(c, dict) and c.get('command') == 'pause' and c.get('value') is False:
            self.clear_lost_pause('web/console resume')

    def clear_lost_pause(self, why):
        if self.lost_paused:
            self.get_logger().info(f'Lost-worker pause cleared ({why})')
        self.lost_paused = False

    def current_yaw(self):
        try:
            return yaw_from_transform(
                self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time()))
        except TransformException:
            return None

    def abort_recovery(self, why):
        r, self.recover = self.recover, None
        if r is None:
            return
        if r['stage'] == 'move':
            self.want_goal = False
            self.cancel_current_goal()
        self.stop_rotation()
        self.get_logger().info(f'Lost-worker search stopped ({why})')

    def recovery_tick(self, now) -> bool:
        """작업자를 lost_recovery_after 초 이상 못 봤을 때: 마지막 좌표로 이동 → 천천히 한 바퀴.
        True 를 돌려주면 이번 주기 동작을 복구가 맡는다."""
        if (not self.lost_recovery_enabled or self.last_worker_seen is None
                or self.last_worker_map is None):
            return False
        r = self.recover
        if r is None:
            if (now - self.last_worker_seen < self.lost_recovery_after
                    or self.recover_done_for == self.last_worker_seen):
                return False
            self.recover_done_for = self.last_worker_seen
            r = self.recover = {'stage': 'move', 'start': now, 'sent': False}
            x, y = self.last_worker_map
            self.get_logger().warning(
                f'Worker lost for {now - self.last_worker_seen:.1f}s -> moving to last position '
                f'({x:.2f}, {y:.2f}), then one slow turn')
        x, y = self.last_worker_map
        if r['stage'] == 'move':
            self.stop_cause = ('대상 소실 - 마지막 위치로 이동 중', f'({x:.2f}, {y:.2f})')
            if not r['sent']:
                if now - r['start'] > self.lost_move_timeout:
                    r['stage'] = 'spin'
                elif self.nav2_ready(now):
                    try:
                        t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
                        rx, ry = t.transform.translation.x, t.transform.translation.y
                    except TransformException:
                        rx, ry = x, y
                    yaw = math.atan2(y - ry, x - rx) if math.hypot(x - rx, y - ry) > 0.05 else 0.0
                    goal = PoseStamped()
                    goal.header.frame_id = self.map_frame
                    goal.header.stamp = self.get_clock().now().to_msg()
                    goal.pose.position.x, goal.pose.position.y = x, y
                    goal.pose.orientation.z = math.sin(yaw / 2.0)
                    goal.pose.orientation.w = math.cos(yaw / 2.0)
                    self.send_goal(goal, (x, y))
                    r['sent'] = True
                return True
            if self.goal_request_pending or self.goal_active:
                if now - r['start'] <= self.lost_move_timeout:
                    return True
                self.want_goal = False
                self.cancel_current_goal()
            self.get_logger().info('Lost-worker search: at last position -> slow 360 deg turn')
            r.update(stage='spin', turned=0.0, last_yaw=None, spin_start=now)
        # ---- spin ----
        if r.get('spin_start') is None:
            r.update(turned=0.0, last_yaw=None, spin_start=now)
        if self.goal_active or self.goal_request_pending or self.cancel_pending:
            return True                      # Nav2 와 동시에 회전 명령을 내지 않음
        yaw = self.current_yaw()
        if yaw is not None:
            if r['last_yaw'] is not None:
                r['turned'] += abs(wrap_angle(yaw - r['last_yaw']))
            r['last_yaw'] = yaw
        spin_limit = 2.0 * (2.0 * math.pi / max(self.lost_spin_speed, 0.05)) + 5.0
        self.stop_cause = ('대상 소실 - 제자리 한 바퀴 탐색 중', f'{math.degrees(r["turned"]):.0f} deg')
        if r['turned'] >= 2.0 * math.pi or now - r['spin_start'] > spin_limit:
            self.stop_rotation()
            self.recover = None
            self.lost_paused = True
            self.get_logger().warning(
                f'Lost-worker search failed ({math.degrees(r["turned"]):.0f} deg turned) -> PAUSED. '
                "Resume on web/console or confirm the worker again ('y').")
            return False
        self.publish_cmd(0.0, self.lost_spin_speed)
        self.rotating = True
        return True

    def control_loop(self):
        now = time.monotonic()
        if not self.startup_step(now):
            self.stop_cause = ('시작 대기 (위치 인식/언독 전)', str(self.phase))
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
        if reason is None and self.lost_paused:
            reason = 'paused after lost-worker search (resume or y to continue)'
        if reason is not None:
            if self.recover is not None:
                self.abort_recovery(f'blocked: {reason}')
        elif scan_lost:
            reason = 'scan timeout'
        elif cam_dead:
            reason = f'no messages on {self.worker_topic}'
        elif worker_lost:
            reason = 'worker not seen by camera'
        elif tf_lost:
            reason = 'TF timeout'
        if reason is None and self.recover is not None:
            # 작업자를 다시 찾음 → 복구 중단, 아래 일반 추종 로직으로 이어감
            self.get_logger().info('Lost-worker search: worker found again -> continue following')
            self.abort_recovery('worker found')
        if reason == 'worker not seen by camera' and self.recovery_tick(now):
            return
        if reason is not None:
            self.stop_cause = self.classify_block(reason, scan_lost, cam_dead, worker_lost, tf_lost)
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
            self.stop_cause = ('대상 소실 - LiDAR', 'LiDAR 클러스터를 놓침 (카메라에는 보임)')
            self.reset_association()

        track = self.track
        if track is None:
            # Camera sees the worker but LiDAR has no match yet: do not drive,
            # but keep the worker centred by rotating in place.
            if not self.stop_cause[0].startswith('대상 소실 - LiDAR'):
                self.stop_cause = ('대상 소실 - LiDAR', '카메라 방향에 맞는 LiDAR 물체 없음')
            self.want_goal = False
            self.cancel_current_goal()
            self.mode = MODE_IDLE
            self.handle_centering(now)
            return

        self.update_mode(track['range'], track['moving'])
        td = self.target_distance
        if self.mode == MODE_HOLD:
            self.stop_cause = ('가까워서 정지 - 목표 거리 도달', f'{track["range"]:.2f} m (목표 {td:.2f} m)')
        elif self.mode == MODE_TOO_CLOSE:
            self.stop_cause = ('가까워서 정지 - 너무 가까움', f'{track["range"]:.2f} m (목표 {td:.2f} m)')
        else:
            self.stop_cause = ('추종 중 멈춤 - Nav2 (경로 실패/장애물/재시도 대기)',
                               f'{track["range"]:.2f} m, Nav2 실패 {self.nav_fail_count}회')
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
        recovering = self.recover is not None and self.recover.get('stage') == 'move'
        if not ((fresh or recovering) and self.want_goal and seq == self.goal_seq):
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


# ================= 터틀봇 추종 (로봇 B 가 로봇 A 를 따라감) =================
class RobotFollower(WorkerFollower):
    """[터틀봇 추종] B로봇 호출로 A 앞에 도착한 로봇 B 가 로봇 A 를 따라간다.

    - 인식: worker_detect tb 모드가 B 의 OAK-D 카메라로 A 를 찾아 /<B>/tb_detect/target 발행
      (형식이 작업자 인식과 같아서 추종 로직(LiDAR 연결·Nav2·회전·소실 복구)은 WorkerFollower 그대로)
    - 카메라: 호출을 받고 언독되면 바로 /<B>/robot_follow/active=True → tb 인식 시작 (가는 중에도 인식)
    - 넘김: 가는 중 터틀봇이 3프레임 연속 인식되면 /<B>/robot_follow/found 발행
            → web_relay 가 좌표 이동을 멈추고 호출을 성공 처리 (web_feedback ok)
    - 켜짐: B 의 web_command 로 온 호출(navigate, purpose "call")의 결과가 web_feedback ok 로 오면
            (가는 중 인식해서 넘겨받았거나, 끝까지 못 보고 A 앞에 도착했거나)
    - 꺼짐: B 에 다른 이동·도킹·언독 명령 / B 의 user_register/cancel /
            부른 로봇 A 의 user_register/cancel(작업 종료·인식 취소·구역 이동) 또는 A 의 dock 명령
    - 가는 중이거나 추종 중일 때만 /<B>/robot_follow/active=True → tb 인식이 그동안만 카메라를 받음
    """

    # WorkerFollower 를 터틀봇 추종용 기본값으로 만들고 호출·카메라 관련 토픽을 추가한다.
    def __init__(self, robot_ns: str):
        ns = robot_ns.strip('/')
        super().__init__(ns, node_prefix='robot_follow', param_defaults={
            'worker_topic': f'/{ns}/tb_detect/target',          # tb 인식 결과
            'debug_target_topic': f'/{ns}/robot_follow/target',
            'require_web_confirm': True,                         # 호출 도착 전에는 추종 안 함
            'target_distance': 0.8,                              # 로봇 A 와 유지할 거리 (m)
            'camera_hfov_deg': 69.0,                             # OAK-D RGB (camera_info 오면 그 값으로)
        })
        self.caller = None           # 따라갈 로봇 A (호출한 로봇)
        self.call_cmd_id = None      # 결과를 기다리는 호출 명령 id (= A 에게 가는 중)
        self.tb_seen_count = 0       # 가는 중 터틀봇이 연속으로 인식된 프레임 수
        self.hfov_logged = False
        self.found_pub = self.create_publisher(Bool, f'/{ns}/robot_follow/found', 10)
        self.stop_log_path = os.path.expanduser(f'~/follow_stop_log_{ns}_robot.csv')
        self.create_subscription(String, f'/{ns}/web_feedback', self.own_feedback_callback, 10)
        self.create_subscription(CameraInfo, f'/{ns}/oakd/rgb/camera_info',
                                 self.camera_info_callback, qos_profile_sensor_data)
        self.active_pub = self.create_publisher(Bool, f'/{ns}/robot_follow/active', 10)
        self.create_timer(0.5, self.publish_active)
        for r in ROBOTS.values():
            other = r['name']
            if other == ns:
                continue
            self.create_subscription(Bool, f'/{other}/user_register/cancel',
                                     lambda m, o=other: self.caller_cancel_callback(o, m), 10)
            self.create_subscription(String, f'/{other}/web_command',
                                     lambda m, o=other: self.caller_command_callback(o, m), 10)
        self.get_logger().info(f'Robot follow for {ns}: follows the calling robot after a call '
                               f'arrives (input {self.worker_topic}, stop log {self.stop_log_path})')

    # 사람 등록 응답(y/n)·재인식은 터틀봇 추종과 상관없으므로 무시.
    def reply_callback(self, msg: String):
        pass

    # (위와 같음) 재인식 무시.
    def retry_callback(self, msg: Bool):
        pass

    # 추종 가능하면 None, 아니면 이유 (호출 도착 전이면 그 이유로).
    def follow_allowed(self) -> Optional[str]:
        if not self.web_enabled:
            return 'waiting for call arrival (robot follow off)'
        return super().follow_allowed()

    # 정지 원인 분류: 호출 도착 전 이유를 추가, 나머지는 작업자 추종과 같음.
    @staticmethod
    def classify_block(reason, scan_lost, cam_dead, worker_lost, tf_lost):
        if reason.startswith('waiting for call arrival'):
            return ('호출 도착 전 (터틀봇 추종 꺼짐)', reason)
        return WorkerFollower.classify_block(reason, scan_lost, cam_dead, worker_lost, tf_lost)

    # B 의 web_command: 호출이면 결과를 기다리고, 다른 이동·도킹·언독이면 터틀봇 추종을 끈다.
    def web_command_callback(self, msg: String):
        super().web_command_callback(msg)
        try:
            c = json.loads(msg.data)
        except ValueError:
            return
        if not isinstance(c, dict) or c.get('robot') not in (None, self.robot_ns):
            return
        command = c.get('command')
        if command == 'navigate' and c.get('purpose') == 'call':
            self.call_cmd_id, self.caller = c.get('id'), c.get('caller')
            if self.web_enabled:
                self.set_web_enabled(False, f'new call from {self.caller}')
            self.get_logger().info(f'Call from {self.caller} received; will follow it after arrival')
        elif command in ('navigate', 'dock', 'undock'):
            self.call_cmd_id = None
            if self.web_enabled:
                self.set_web_enabled(False, f'web {command} command')

    # B 의 web_feedback: 기다리던 호출이 성공하면 터틀봇 추종 시작.
    def own_feedback_callback(self, msg: String):
        try:
            f = json.loads(msg.data)
        except ValueError:
            return
        if (not isinstance(f, dict) or self.call_cmd_id is None
                or f.get('id') != self.call_cmd_id or f.get('command') != 'navigate'):
            return
        self.call_cmd_id = None
        if f.get('ok'):
            self.clear_lost_pause('call arrived')
            self.set_web_enabled(True, f'arrived at {self.caller} -> following {self.caller}')
        else:
            self.get_logger().warning(f'Call from {self.caller} failed ({f.get("message")}); not following')

    # 부른 로봇 A 의 user_register/cancel (작업 종료·인식 취소·구역 이동) → 터틀봇 추종 끔.
    def caller_cancel_callback(self, other, msg: Bool):
        if msg.data and other == self.caller and self.web_enabled:
            self.set_web_enabled(False, f'{other} job end / recognition cancel / zone move')

    # 부른 로봇 A 에 dock 명령 → 터틀봇 추종 끔.
    def caller_command_callback(self, other, msg: String):
        if other != self.caller or not self.web_enabled:
            return
        try:
            c = json.loads(msg.data)
        except ValueError:
            return
        if isinstance(c, dict) and c.get('command') == 'dock':
            self.set_web_enabled(False, f'{other} is docking')

    # OAK-D camera_info: 초점거리로 실제 가로 화각을 계산해 방향 계산에 사용.
    def camera_info_callback(self, msg: CameraInfo):
        fx, width = float(msg.k[0]), int(msg.width)
        if fx <= 0.0 or width <= 0:
            return
        self.tan_half_hfov = (width / 2.0) / fx
        if not self.hfov_logged:
            self.hfov_logged = True
            self.get_logger().info(f'OAK-D hfov from camera_info: '
                                   f'{math.degrees(2.0 * math.atan(self.tan_half_hfov)):.1f} deg')

    # 카메라(터틀봇 인식)가 켜져 있어야 하는지: 추종 중이거나, 호출로 가는 중이고 언독된 상태.
    def camera_wanted(self) -> bool:
        going = self.call_cmd_id is not None and self.is_docked is False
        return bool(self.web_enabled or going)

    # (0.5초) 카메라 켜짐 여부 발행 → worker_detect tb 모드가 그동안만 인식.
    def publish_active(self):
        self.active_pub.publish(Bool(data=self.camera_wanted()))

    # tb 인식 결과: 원래 처리 + (A 에게 가는 중이면) 터틀봇이 FOUND_FRAMES 연속 인식되면
    # /<B>/robot_follow/found 발행 → web_relay 가 좌표 이동을 멈추고 호출 성공 처리 → 추종 시작.
    FOUND_FRAMES = 3

    def worker_callback(self, msg):
        super().worker_callback(msg)
        if self.call_cmd_id is None or self.web_enabled:
            self.tb_seen_count = 0
            return
        found = len(msg.data) >= 1 and msg.data[0] >= 0.5
        self.tb_seen_count = self.tb_seen_count + 1 if found else 0
        if self.tb_seen_count >= self.FOUND_FRAMES:
            if self.tb_seen_count == self.FOUND_FRAMES:
                self.get_logger().info(f'TurtleBot detected on the way to {self.caller} '
                                       '-> asking web_relay to hand over to robot follow')
            self.found_pub.publish(Bool(data=True))


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
    argv = list(sys.argv if args is None else args)
    # 첫 인자가 robot / tb 면 터틀봇 추종 모드 (나머지 인자는 로봇 선택 그대로)
    user_args = remove_ros_args(args=argv)[1:]
    robot_mode = bool(user_args) and user_args[0].lower() in ('robot', 'tb', 'turtlebot')
    if robot_mode:
        argv.pop(argv.index(user_args[0], 1))
        print('>>> Robot follow mode: the called robot follows the calling robot (tb_detect)',
              flush=True)
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
            node = RobotFollower(r['name']) if robot_mode else WorkerFollower(r['name'])
            nodes.append(node)
            executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for node in nodes:
            try:
                node.print_stop_summary()
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
