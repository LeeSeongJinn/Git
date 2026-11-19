#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# worker_detect_node.py — 인식 노드 (두 가지 모드)
#
#  [사람 모드] 기본 — 웹캠 PC 의 웹캠으로 작업자 인식, pose_clothing_test_v3.py 로직
#     - 검출용 영상만 밝게 보정: CLAHE + 감마 (모드 2 는 형광등 영역 회색 처리까지)
#     - 의상 색 특징(HSV 히스토그램)은 보정 전 원본에서 추출 → 색이 바뀌지 않음
#     - 오검출 필터: 보이는 관절 수(MIN_KP) + 같은 추적 ID 가 MIN_HITS 프레임 이상 (ByteTrack)
#     - 손 머리 위로 RAISE_FRAMES 프레임 / r → 등록, 상의 0.6 + 하의 0.4 유사도로 재식별
#     발행: /<robot>/worker_detect/target (Float32MultiArray) 매 프레임
#             [찾음, 중심x, 중심y, 폭, 높이(0~1), 유사도, -1, 0, 영상폭, 영상높이, 등록여부]
#           /<robot>/user_register/image (등록 사진), /<robot>/user_track/following (보이는지)
#     구독: /<robot>/worker_detect/command ('register'|'clear'),
#           /<robot>/user_register/retry, /<robot>/user_register/cancel → 등록 취소
#     키:   r 등록 / c 취소 / [ ] 같은 사람 기준 / - = 검출 신뢰도 / e 보정 모드 / d 보정 영상 / q 종료
#     실행: ros2 run my_package worker_detect 1 --device 0 --model ~/turtlebot4_ws/yolov8n-pose.pt
#           (1 = robot6, 2 = robot11 / --cam 웹캠 번호, --no-show, --enhance 0~2)
#
#  [터틀봇 모드] 첫 인자 tb — 로봇 B 의 OAK-D 로 로봇 A(TurtleBot) 인식, tb_detect_node(1).py 로직
#     - YOLO(turtlebot_v8n_best.pt) 박스 + 거리(Depth / 박스 크기), 'c' 키 보정
#     - 기본은 /<B>/robot_follow/active 가 True 일 때만 카메라 구독·인식
#       (worker_follow robot 모드가 '호출 도착 후 추종 중' 일 때만, --always 면 항상)
#     발행: /<B>/tb_detect/target  [찾음, 중심x, 중심y, 폭, 높이, 신뢰도, 거리, 거리출처, 영상폭, 영상높이, 1]
#           /<B>/tb_detect/image/compressed
#     실행: ros2 run my_package worker_detect tb                     # 두 로봇 모두 대기
#           ros2 run my_package worker_detect tb 2 --show --always   # robot11 카메라, 항상 (시험·보정)
#
#  필요: pip install ultralytics lap   (lap: ByteTrack 추적)
# =============================================================================

import argparse
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Bool, Float32MultiArray, String
from ultralytics import YOLO

ROBOTS = {'1': 'robot6', '2': 'robot11'}

# #############################################################################
#  [사람 모드] pose_clothing_test_v3
# #############################################################################
# ---------------------------------------------------------------------------
# 설정값 (pose_clothing_test_v3.py 와 동일)
# ---------------------------------------------------------------------------
IMGSZ = 640              # 사람이 작게 보이면 960 (FPS 떨어짐)
DET_CONF = 0.25          # 사람 박스 신뢰도
DET_MIN, DET_MAX = 0.05, 0.8
KP_CONF = 0.5            # 관절 신뢰도가 이 값 이하면 안 보이는 관절
MIN_KP = 5               # 보이는 관절이 이 개수 미만이면 사람 아님
MIN_HITS = 3             # 같은 추적 ID 가 이 프레임 수 이상 잡혀야 사람으로 인정

ENHANCE_MODE = 1         # 0 없음, 1 CLAHE+감마, 2 CLAHE+감마+형광등 가리기
GAMMA = 0.6              # 1보다 작을수록 어두운 부분을 밝게
SAT_V = 245              # 이 밝기 이상이면 형광등(포화 영역)

MATCH_THRESHOLD = 0.5    # 이 점수 이상이면 등록 작업자
MATCH_MIN, MATCH_MAX = 0.3, 0.9
UPDATE_THRESHOLD = 0.75  # 이 점수 이상일 때만 등록 특징 갱신
UPDATE_RATE = 0.1
RAISE_FRAMES = 15        # 손을 이만큼 연속 프레임 들어야 등록 (약 0.5초)

# COCO 관절 번호
L_SH, R_SH = 5, 6
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KN, R_KN = 13, 14

CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
GAMMA_LUT = np.array([((i / 255.0) ** GAMMA) * 255 for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# 0. 검출용 영상 보정
# ---------------------------------------------------------------------------
# 검출용 영상 보정: mode 0 원본 / 1 CLAHE+감마 / 2 형광등(포화) 영역을 회색으로 채운 뒤 1번.
def enhance(frame, mode):
    if mode == 0:
        return frame
    out = frame
    if mode == 2:
        v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2]
        mask = (v >= SAT_V).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))  # 번짐(글레어)까지 포함
        out = frame.copy()
        out[mask > 0] = 114
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lab[..., 0] = CLAHE.apply(lab[..., 0])                    # 밝기 채널만 CLAHE
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.LUT(out, GAMMA_LUT)                            # 감마로 어두운 부분 끌어올리기


# ---------------------------------------------------------------------------
# 1. 관절로 상의·하의 영역 구하기
# ---------------------------------------------------------------------------
# 지정한 관절들이 모두 보이는지 (신뢰도 > KP_CONF).
def visible(conf, *ids):
    return all(conf[i] > KP_CONF for i in ids)


# 사각형을 영상 안으로 자르고, 너무 작으면(10px 미만) None.
def clip_rect(x1, y1, x2, y2, w, h):
    x1, x2 = sorted((int(x1), int(x2)))
    y1, y2 = sorted((int(y1), int(y2)))
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return x1, y1, x2, y2


# 관절 위치로 상의(어깨~골반)와 하의(골반~무릎) 사각형을 구한다. 안 보이면 None.
def get_regions(kp, conf, shape):
    h, w = shape[:2]
    upper = lower = None
    if visible(conf, L_SH, R_SH):
        sx1, sx2 = sorted((kp[L_SH][0], kp[R_SH][0]))
        sw = sx2 - sx1
        sy = (kp[L_SH][1] + kp[R_SH][1]) / 2
        if visible(conf, L_HIP, R_HIP):
            hy = (kp[L_HIP][1] + kp[R_HIP][1]) / 2
        else:
            hy = sy + sw * 1.2
        margin_y = (hy - sy) * 0.1
        upper = clip_rect(sx1 + sw * 0.1, sy + margin_y, sx2 - sw * 0.1, hy - margin_y, w, h)
    if visible(conf, L_HIP, R_HIP, L_KN, R_KN):
        hx1, hx2 = sorted((kp[L_HIP][0], kp[R_HIP][0]))
        hy = (kp[L_HIP][1] + kp[R_HIP][1]) / 2
        ky = (kp[L_KN][1] + kp[R_KN][1]) / 2
        lower = clip_rect(hx1, hy + (ky - hy) * 0.1, hx2, ky, w, h)
    return upper, lower


# ---------------------------------------------------------------------------
# 2. 의상 특징: HSV 색 히스토그램 (원본 프레임에서)
# ---------------------------------------------------------------------------
# 사각형 영역의 HSV 3차원 히스토그램 (합 1로 정규화). 영역이 없으면 None.
def color_hist(frame, rect):
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
    return hist


# 등록 특징과 후보 특징의 유사도 0~1 (상의 0.6 + 하의 0.4, 하의가 없으면 상의만).
def similarity(target, feat):
    if target['upper'] is None or feat['upper'] is None:
        return 0.0
    du = cv2.compareHist(target['upper'], feat['upper'], cv2.HISTCMP_BHATTACHARYYA)
    if target['lower'] is not None and feat['lower'] is not None:
        dl = cv2.compareHist(target['lower'], feat['lower'], cv2.HISTCMP_BHATTACHARYYA)
        d = 0.6 * du + 0.4 * dl
    else:
        d = du
    return 1.0 - d


# 등록 특징을 새 특징 쪽으로 UPDATE_RATE 만큼 갱신 (조명 변화 따라가기).
def blend(old, new):
    if old is None:
        return None if new is None else new.copy()
    if new is None:
        return old
    out = (1 - UPDATE_RATE) * old + UPDATE_RATE * new
    cv2.normalize(out, out, 1, 0, cv2.NORM_L1)
    return out


# 손목이 어깨보다 어깨너비의 절반 이상 위에 있으면 손을 든 것.
def hand_raised(kp, conf):
    if not visible(conf, L_SH, R_SH):
        return False
    sy = min(kp[L_SH][1], kp[R_SH][1])
    sw = abs(kp[L_SH][0] - kp[R_SH][0])
    for wr in (L_WR, R_WR):
        if conf[wr] > KP_CONF and kp[wr][1] < sy - sw * 0.5:
            return True
    return False


# 사람 박스 넓이 (가장 큰 사람 고르기용).
def box_area(p):
    x1, y1, x2, y2 = p['box']
    return (x2 - x1) * (y2 - y1)


# ---------------------------------------------------------------------------
# 3. 검출 + 추적 + 오검출 필터
# ---------------------------------------------------------------------------
# 보정 영상으로 YOLO Pose + ByteTrack 실행 → (사람 목록, 필터에 걸린 박스 목록).
#   hits: 추적 ID 별 연속 검출 횟수 (이번 프레임에 없는 ID 는 지움)
def detect_people(model, frame, det_img, det_conf, imgsz, device, hits):
    kw = {} if device is None else {'device': device}
    r = model.track(det_img, persist=True, conf=det_conf, imgsz=imgsz,
                    tracker='bytetrack.yaml', verbose=False, **kw)[0]
    people, rejected, seen_ids = [], [], set()
    if len(r.boxes) > 0 and r.keypoints is not None and r.keypoints.conf is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        kps = r.keypoints.xy.cpu().numpy()
        kcs = r.keypoints.conf.cpu().numpy()
        ids = (r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else [-1] * len(boxes))
        for box, kp, kc, tid in zip(boxes, kps, kcs, ids):
            tid = int(tid)
            if tid >= 0:
                seen_ids.add(tid)
                hits[tid] += 1
            n_kp = int((kc > KP_CONF).sum())
            if n_kp < MIN_KP or (tid >= 0 and hits[tid] < MIN_HITS):
                rejected.append(box)
                continue
            up, lo = get_regions(kp, kc, frame.shape)
            people.append({
                'box': box, 'kp': kp, 'kc': kc, 'up': up, 'lo': lo, 'id': tid,
                # 색 특징은 원본 frame 에서 → 보정 때문에 색이 바뀌지 않음
                'feat': {'upper': color_hist(frame, up), 'lower': color_hist(frame, lo)},
                'raised': hand_raised(kp, kc),
            })
    for tid in list(hits):
        if tid not in seen_ids:
            del hits[tid]
    return people, rejected


# 등록 작업자와 가장 비슷한 사람과 점수. 점수가 높으면 등록 특징 갱신, 기준 미만이면 사람은 None.
def match_worker(target, people, match_thr):
    best, best_score = None, 0.0
    if target is None:
        return None, 0.0
    for p in people:
        p['score'] = similarity(target, p['feat'])
        if p['score'] > best_score:
            best, best_score = p, p['score']
    if best is not None and best_score >= UPDATE_THRESHOLD:
        target['upper'] = blend(target['upper'], best['feat']['upper'])
        target['lower'] = blend(target['lower'], best['feat']['lower'])
    if best_score < match_thr:
        best = None
    return best, best_score


# 화면에 박스·관절·상하의 영역·상태 글자를 그린다.
def draw(view, people, rejected, best, status_line, robot):
    h, w = view.shape[:2]
    for box in rejected:                        # 필터에 걸린 박스: 빨간 얇은 선
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 0, 180), 1)
    for p in people:
        x1, y1, x2, y2 = map(int, p['box'])
        is_target = p is best
        color = (0, 200, 0) if is_target else (150, 150, 150)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 3 if is_target else 1)
        if p['up']:
            cv2.rectangle(view, p['up'][:2], p['up'][2:], (255, 120, 0), 1)   # 상의: 파랑
        if p['lo']:
            cv2.rectangle(view, p['lo'][:2], p['lo'][2:], (0, 220, 255), 1)   # 하의: 노랑
        for (x, y), c in zip(p['kp'], p['kc']):
            if c > KP_CONF:
                cv2.circle(view, (int(x), int(y)), 3, (0, 0, 255), -1)
        label = f"#{p['id']}" + (' WORKER' if is_target else '')
        if 'score' in p:
            label += f" {p['score']:.2f}"
        cv2.putText(view, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(view, status_line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(view, f'-> {robot}', (w - 130, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# ---------------------------------------------------------------------------
# ROS 연결 (worker_detect_node.py 와 같음)
# ---------------------------------------------------------------------------
# 인식 결과 발행 + 등록/취소 명령 수신을 맡는 ROS 노드.
class WorkerDetectRos(Node):

    # 로봇별 발행자·구독자 생성.
    def __init__(self, robot):
        super().__init__(f'worker_detect_{robot}')
        ns = f'/{robot}'
        self.robot = robot
        self.pub_target = self.create_publisher(Float32MultiArray, f'{ns}/worker_detect/target', 10)
        self.pub_reg = self.create_publisher(CompressedImage, f'{ns}/user_register/image', 10)
        self.pub_follow = self.create_publisher(Bool, f'{ns}/user_track/following', 10)
        self.cmd = None   # 'register' | 'clear' (다음 프레임에서 처리)
        self.create_subscription(String, f'{ns}/worker_detect/command', self.on_command, 10)
        self.create_subscription(Bool, f'{ns}/user_register/retry', self.on_clear, 10)
        self.create_subscription(Bool, f'{ns}/user_register/cancel', self.on_clear, 10)
        self.last_following = None
        self.last_following_pub = 0.0

    # worker_detect/command 수신: 'register' / 'clear' 를 다음 프레임에서 처리하도록 저장.
    def on_command(self, msg):
        c = msg.data.strip().lower()
        if c in ('register', 'clear'):
            self.cmd = c

    # 웹 재인식·취소 수신: 등록 취소 예약.
    def on_clear(self, msg):
        if msg.data:
            self.cmd = 'clear'

    # 예약된 명령을 꺼내고 비운다.
    def take_cmd(self):
        c, self.cmd = self.cmd, None
        return c

    # 인식 결과 발행 (worker_follow 형식) + 작업자 보임 여부 (바뀔 때 + 1초마다).
    def publish_result(self, best, best_score, registered, w, h):
        out = Float32MultiArray()
        if best is not None:
            x1, y1, x2, y2 = [float(v) for v in best['box']]
            out.data = [1.0, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h,
                        float(best_score), -1.0, 0.0, float(w), float(h), 1.0]
        else:
            out.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, float(w), float(h),
                        1.0 if registered else 0.0]
        self.pub_target.publish(out)
        if registered:
            seen = best is not None
            now = time.time()
            if seen != self.last_following or now - self.last_following_pub >= 1.0:
                self.pub_follow.publish(Bool(data=seen))
                self.last_following, self.last_following_pub = seen, now
        else:
            self.last_following = None

    # 등록 순간 사진(박스 표시)을 JPEG 로 발행 → 웹/콘솔에서 맞는지 확인.
    def publish_register_photo(self, frame, person):
        snap = frame.copy()
        x1, y1, x2, y2 = map(int, person['box'])
        cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 200, 0), 3)
        img = CompressedImage()
        img.header.stamp = self.get_clock().now().to_msg()
        img.format = 'jpeg'
        img.data = cv2.imencode('.jpg', snap, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        self.pub_reg.publish(img)


# ---------------------------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------------------------
# [사람 모드] 명령행 인자 해석.
def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('robot', nargs='?', default='', help='1(robot6) / 2(robot11)')
    ap.add_argument('--cam', type=int, default=0, help='웹캠 번호')
    ap.add_argument('--model', default='yolov8n-pose.pt')
    ap.add_argument('--device', default=None, help="GPU면 '0' (없으면 ultralytics 기본)")
    ap.add_argument('--imgsz', type=int, default=IMGSZ)
    ap.add_argument('--conf', type=float, default=DET_CONF, help='사람 박스 신뢰도')
    ap.add_argument('--enhance', type=int, choices=[0, 1, 2], default=ENHANCE_MODE,
                    help='보정 모드 0 없음 / 1 CLAHE+감마 / 2 +형광등 가리기')
    ap.add_argument('--no-show', dest='show', action='store_false', help='화면 창 끄기')
    ap.add_argument('--show', dest='show', action='store_true', help='(기본) 화면 창 표시')
    ap.set_defaults(show=True)
    return ap.parse_args(argv)


# 인자 또는 터미널 입력으로 결과를 보낼 로봇을 정한다 (q 면 None).
def choose_robot(arg):
    robot = ROBOTS.get(arg, arg if arg in ROBOTS.values() else None)
    while robot is None:
        c = input('결과를 보낼 로봇 -> [1] robot6  [2] robot11  (q: 종료): ').strip()
        if c.lower() == 'q':
            return None
        robot = ROBOTS.get(c)
    return robot


# 키 입력으로 실행 중 설정 조절 (기준·신뢰도·보정 모드·보정 영상 보기). 바뀐 설정 dict 반환.
def handle_tuning_keys(key, s):
    if key == ord('['):
        s['match'] = max(MATCH_MIN, round(s['match'] - 0.05, 2))
        print(f"같은 사람 기준 {s['match']:.2f} (낮을수록 쉽게 찾음)")
    elif key == ord(']'):
        s['match'] = min(MATCH_MAX, round(s['match'] + 0.05, 2))
        print(f"같은 사람 기준 {s['match']:.2f} (높을수록 엄격)")
    elif key == ord('-'):
        s['det'] = max(DET_MIN, round(s['det'] - 0.05, 2))
        print(f"검출 신뢰도 {s['det']:.2f}")
    elif key == ord('='):
        s['det'] = min(DET_MAX, round(s['det'] + 0.05, 2))
        print(f"검출 신뢰도 {s['det']:.2f}")
    elif key == ord('e'):
        s['mode'] = (s['mode'] + 1) % 3
        print(f"보정 모드 {s['mode']} (0 없음 / 1 CLAHE+감마 / 2 +형광등 가리기)")
    elif key == ord('d'):
        s['show_det'] = not s['show_det']
    return s


# [사람 모드] 웹캠 → 보정 → 검출·추적 → 등록·매칭 → ROS 발행 → 화면.
def person_main(argv):
    a = parse_args(argv)
    robot = choose_robot(a.robot)
    if robot is None:
        return
    model = YOLO(a.model)
    rclpy.init()
    ros = WorkerDetectRos(robot)
    ros.get_logger().info(f'pose_clothing_test_v3 로직으로 인식 → /{robot}/worker_detect/target '
                          f'(카메라 {a.cam}, 모델 {a.model}, 보정 모드 {a.enhance})')

    s = {'match': MATCH_THRESHOLD, 'det': a.conf, 'mode': a.enhance, 'show_det': False}
    hits = defaultdict(int)          # 추적 ID 별 연속 검출 횟수
    cap = cv2.VideoCapture(a.cam)
    target = None
    raise_count = 0
    prev_t = time.time()

    try:
        while rclpy.ok():
            ok, frame = cap.read()
            if not ok:
                print('카메라를 열 수 없습니다.')
                break
            h, w = frame.shape[:2]

            # --- 보정 + 검출 + 추적 ---
            det_img = enhance(frame, s['mode'])
            people, rejected = detect_people(model, frame, det_img, s['det'], a.imgsz, a.device, hits)

            key = (cv2.waitKey(1) & 0xFF) if a.show else 255
            rclpy.spin_once(ros, timeout_sec=0.0)
            cmd = ros.take_cmd()

            # --- 등록 / 취소 ---
            usable = [p for p in people if p['feat']['upper'] is not None]
            raisers = [p for p in usable if p['raised']]
            raise_count = raise_count + 1 if (raisers and target is None) else 0
            chosen = None
            if raise_count >= RAISE_FRAMES:
                chosen = max(raisers, key=box_area)
            elif (key == ord('r') or cmd == 'register') and usable:
                chosen = max(usable, key=box_area)
            if chosen is not None:
                target = {k: (v.copy() if v is not None else None) for k, v in chosen['feat'].items()}
                raise_count = 0
                print('작업자 등록 완료')
                ros.publish_register_photo(frame, chosen)
            if key == ord('c') or cmd == 'clear':
                target = None
                print('등록 취소')
            s = handle_tuning_keys(key, s)

            # --- 매칭 + ROS 발행 ---
            best, best_score = match_worker(target, people, s['match'])
            ros.publish_result(best, best_score, target is not None, w, h)

            # --- 화면 ---
            if a.show:
                now = time.time()
                fps = 1.0 / max(now - prev_t, 1e-6)
                prev_t = now
                if target is None:
                    status = f'WAITING ({raise_count}/{RAISE_FRAMES}) raise hand or r'
                else:
                    status = 'TRACKING' if best is not None else 'LOST'
                view = det_img.copy() if s['show_det'] else frame
                draw(view, people, rejected, best,
                     f"{status} | det {s['det']:.2f} | enh {s['mode']} | match {s['match']:.2f} | FPS {fps:.1f}",
                     robot)
                cv2.imshow('worker detect (person, v3)', view)
                if key == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        ros.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# #############################################################################
#  [터틀봇 모드] tb_detect_node(1)
# #############################################################################
MODEL_CANDIDATES = ['~/yolo_train/turtlebot_v8n_best.pt', '~/Downloads/turtlebot_v8n_best.pt']
ACTIVE_TIMEOUT = 3.0     # robot_follow/active 가 이 시간(초) 안 오면 꺼진 것으로 봄


# ROS 메시지 header 의 시각(초).
def stamp_sec(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


# compressedDepth (png) → uint16(mm) 또는 float32(m). 앞 12바이트는 설정 헤더.
def decode_depth_compressed(msg):
    raw = np.frombuffer(msg.data, np.uint8)
    for start in (12, 0):
        img = cv2.imdecode(raw[start:], cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    return None


# 압축 안 된 Depth Image → numpy 배열 (16UC1 / 32FC1 만).
def decode_depth_raw(msg):
    if msg.encoding.lower() in ('16uc1', 'mono16'):
        return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
    if msg.encoding.lower() == '32fc1':
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.step // 4)[:, :msg.width]
    return None


# --model 이 비었으면 알려진 위치에서 TurtleBot 모델 파일을 찾는다.
def find_model(path):
    if path:
        return os.path.expanduser(path)
    for p in MODEL_CANDIDATES:
        if os.path.exists(os.path.expanduser(p)):
            return os.path.expanduser(p)
    return os.path.expanduser(MODEL_CANDIDATES[0])


# 로봇 한 대의 카메라로 TurtleBot 을 인식하는 노드.
class TbDetect(Node):

    # 발행자·보정값 준비. 카메라 구독은 active 일 때만 (또는 --always 면 바로).
    def __init__(self, a, ns, model):
        ns = ns.rstrip('/')
        self.ns = ns
        self.tag = ns.strip('/').replace('/', '_') or 'robot'
        super().__init__(f'tb_detect_{self.tag}')
        self.a = a
        self.model = model
        self.rgb_topic = a.topic or f'{ns}/oakd/rgb/image_raw/compressed'
        self.dep_topic = a.depth_topic or f'{ns}/oakd/stereo/image_raw/compressedDepth'
        self.info_topic = a.info_topic or f'{ns}/oakd/rgb/camera_info'
        self.pub_t = self.create_publisher(Float32MultiArray, f'{ns}/tb_detect/target', 10)
        self.pub_i = self.create_publisher(CompressedImage, f'{ns}/tb_detect/image/compressed', 10)
        self.create_subscription(Bool, f'{ns}/robot_follow/active', self.on_active, 10)
        self.camera_subs = []
        self.active_until = 0.0
        self.create_timer(0.5, self.update_camera_subscription)

        self.depth, self.depth_t = None, 0.0
        self.fx, self.fy, self.info_w, self.info_h = None, None, None, None
        self.calib_file = os.path.expanduser(a.calib_file or f'~/yolo_train/tb_dist_calib_{self.tag}.txt')
        self.tb_h, self.tb_w = a.tb_height, a.tb_width
        self.load_calibration()
        self.busy = False
        self.t_prev = time.time()
        self.fps = 0.0
        self.logged = False
        self.last_box_px = None
        self.get_logger().info(f'[{self.tag}] RGB: {self.rgb_topic} | Depth: {self.dep_topic} | '
                               f'conf {a.conf} | 거리 {a.dist_mode} | '
                               f'{"항상 인식" if a.always else "robot_follow 추종 중일 때만 인식"}')

    # 보정 파일(로봇별, 없으면 예전 공용 파일)에서 로봇 높이·폭 보정값을 읽는다.
    def load_calibration(self):
        load_from = self.calib_file
        old = os.path.expanduser('~/yolo_train/tb_dist_calib.txt')
        if not os.path.exists(load_from) and os.path.exists(old):
            load_from = old
            self.get_logger().warn(f'{self.tag} 전용 보정 파일이 없어 공용 파일을 사용해요: {old}')
        if os.path.exists(load_from):
            try:
                vals = [float(v) for v in open(load_from).read().split()]
                self.tb_h = vals[0]
                if len(vals) > 1:
                    self.tb_w = vals[1]
                self.get_logger().info(f'보정값 불러옴: 높이 {self.tb_h:.4f} m, 폭 {self.tb_w:.4f} m ({load_from})')
            except (ValueError, IndexError):
                pass

    # ---------- 켜기 / 끄기 ----------
    # robot_follow/active 수신: True 면 ACTIVE_TIMEOUT 동안 켜진 것으로 본다.
    def on_active(self, msg):
        if msg.data:
            self.active_until = time.monotonic() + ACTIVE_TIMEOUT
        else:
            self.active_until = 0.0

    # (0.5초 타이머) 켜져야 하면 카메라 토픽 구독 시작, 꺼져야 하면 구독 해제.
    def update_camera_subscription(self):
        want = self.a.always or time.monotonic() < self.active_until
        if want and not self.camera_subs:
            self.subscribe_camera()
            self.get_logger().info(f'[{self.tag}] 인식 시작 (카메라 구독)')
        elif not want and self.camera_subs:
            for s in self.camera_subs:
                self.destroy_subscription(s)
            self.camera_subs = []
            self.depth = None
            self.get_logger().info(f'[{self.tag}] 인식 중지 (카메라 구독 해제)')

    # RGB · camera_info · (필요하면) Depth 구독 생성.
    def subscribe_camera(self):
        c = self.create_subscription
        self.camera_subs = [
            c(CompressedImage, self.rgb_topic, self.on_image, qos_profile_sensor_data),
            c(CameraInfo, self.info_topic, self.on_info, qos_profile_sensor_data),
        ]
        if self.a.dist_mode != 'box':
            if 'compressed' in self.dep_topic:
                self.camera_subs.append(c(CompressedImage, self.dep_topic, self.on_depth_c,
                                          qos_profile_sensor_data))
            else:
                self.camera_subs.append(c(Image, self.dep_topic, self.on_depth_r,
                                          qos_profile_sensor_data))

    # ---------- 입력 ----------
    # camera_info 수신: 초점거리 fx, fy 와 원본 영상 크기 저장.
    def on_info(self, msg):
        if msg.k[4] > 0:
            self.fx, self.fy = float(msg.k[0]), float(msg.k[4])
            self.info_w, self.info_h = int(msg.width), int(msg.height)

    # 압축 Depth 수신.
    def on_depth_c(self, msg):
        d = decode_depth_compressed(msg)
        if d is not None:
            self.depth, self.depth_t = d, stamp_sec(msg.header)

    # 압축 안 된 Depth 수신.
    def on_depth_r(self, msg):
        d = decode_depth_raw(msg)
        if d is not None:
            self.depth, self.depth_t = d, stamp_sec(msg.header)

    # ---------- 초점거리 (영상 크기에 맞춤) ----------
    # 현재 영상 높이에 맞춘 fy (camera_info 가 없으면 세로 시야각으로 추정).
    def fy_for(self, img_h):
        if self.fy and self.info_h:
            return self.fy * img_h / self.info_h
        return (img_h / 2) / np.tan(np.radians(self.a.vfov / 2))

    # 현재 영상 크기에 맞춘 fx (없으면 fy 와 같다고 봄).
    def fx_for(self, img_w, img_h):
        if self.fx and self.info_w:
            return self.fx * img_w / self.info_w
        return self.fy_for(img_h)

    # ---------- Depth 거리 ----------
    # Depth 를 읽을 영역 (박스 아랫부분 띠 또는 가운데), 0~1 비율 좌표.
    def roi_rect(self, box_n):
        x1, y1, x2, y2 = box_n
        bw, bh = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2
        hw = bw * self.a.roi / 2
        if self.a.roi_pos == 'bottom':
            ry2 = y2 - bh * self.a.bottom_margin
            ry1 = ry2 - bh * self.a.roi_h
        else:
            cy, hh = (y1 + y2) / 2, bh * self.a.roi / 2
            ry1, ry2 = cy - hh, cy + hh
        return cx - hw, ry1, cx + hw, ry2

    # Depth 영역의 중앙값 거리(m). Depth 가 없거나 오래됐거나 값이 부족하면 -1.
    def depth_distance(self, box_n, rgb_t):
        if self.depth is None or abs(rgb_t - self.depth_t) > self.a.max_dt:
            return -1.0
        dh, dw = self.depth.shape[:2]
        rx1, ry1, rx2, ry2 = self.roi_rect(box_n)
        u1, u2 = int(max(0, rx1 * dw)), int(min(dw, rx2 * dw))
        v1, v2 = int(max(0, ry1 * dh)), int(min(dh, ry2 * dh))
        if u2 - u1 < 2 or v2 - v1 < 2:
            return -1.0
        roi = self.depth[v1:v2, u1:u2].astype(np.float32)
        vals = roi[roi > 0]
        if vals.size < 10:
            return -1.0
        d = float(np.median(vals))
        if self.depth.dtype == np.uint16:
            d /= 1000.0
        return d

    # ---------- 박스 크기 거리 ----------
    # 박스 높이로 거리 계산, 위아래가 잘렸으면 폭으로. 반환 (거리, 'H'|'W'), 안 되면 (-1, '').
    def box_distance(self, box, img_w, img_h):
        x1, y1, x2, y2 = box
        m = self.a.edge
        cut_v = y1 <= m or y2 >= img_h - m
        cut_h = x1 <= m or x2 >= img_w - m
        if not cut_v and (y2 - y1) >= 5:
            return self.tb_h * self.fy_for(img_h) / (y2 - y1), 'H'
        if not cut_h and (x2 - x1) >= 5:
            return self.tb_w * self.fx_for(img_w, img_h) / (x2 - x1), 'W'
        return -1.0, ''

    # 'c' 키 보정: 지금 박스 크기와 --calib-dist 로 로봇 높이·폭을 계산해 파일에 저장.
    def calibrate(self):
        if self.last_box_px is None:
            self.get_logger().warn('보정 실패: 로봇 A가 인식되지 않았어요.')
            return
        x1, y1, x2, y2, img_w, img_h = self.last_box_px
        m = self.a.edge
        if y1 <= m or y2 >= img_h - m or x1 <= m or x2 >= img_w - m:
            self.get_logger().warn('보정 실패: 로봇이 화면 끝에 잘려 있어요. 화면 안에 다 들어오게 놓고 다시 c')
            return
        d = self.a.calib_dist
        self.tb_h = d * (y2 - y1) / self.fy_for(img_h)
        self.tb_w = d * (x2 - x1) / self.fx_for(img_w, img_h)
        os.makedirs(os.path.dirname(self.calib_file), exist_ok=True)
        with open(self.calib_file, 'w') as f:
            f.write(f'{self.tb_h:.5f} {self.tb_w:.5f}\n')
        self.get_logger().info(f'보정 완료 ({d} m 기준): 높이 {self.tb_h:.4f} m, 폭 {self.tb_w:.4f} m → {self.calib_file}')

    # 박스 하나의 거리와 출처 (Depth 우선, 안 되면 박스 크기). 반환 (거리, 출처 0/1/2, 'H'|'W'|'').
    def measure(self, box, box_n, w, h, rgb_t):
        dist, src, how = -1.0, 0, ''
        if self.a.dist_mode in ('auto', 'depth'):
            d = self.depth_distance(box_n, rgb_t)
            if d >= 0:
                dist, src = d, 1
        if src == 0 and self.a.dist_mode in ('auto', 'box'):
            d, how = self.box_distance(box, w, h)
            if d >= 0:
                dist, src = d, 2
        if dist >= 0:
            dist = max(0.0, dist - self.a.offset)
        return dist, src, how

    # ---------- 메인 처리 ----------
    # RGB 프레임 수신: 검출 → 거리 → target 발행 → 디버그 영상 발행 (처리 중이면 이 프레임은 건너뜀).
    def on_image(self, msg):
        if self.busy:
            return
        self.busy = True
        try:
            frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return
            h, w = frame.shape[:2]
            if not self.logged:
                src = 'camera_info' if self.fy else f'시야각 {self.a.vfov}° 추정'
                self.get_logger().info(f'[{self.tag}] RGB {w}x{h}, 초점거리 fy {self.fy_for(h):.1f}px ({src})')
                self.logged = True
            r = self.model.predict(frame, conf=self.a.conf, imgsz=self.a.imgsz, device=self.a.device,
                                   max_det=5, verbose=False)[0]
            out = Float32MultiArray()
            if len(r.boxes) > 0:
                i = int(r.boxes.conf.argmax())
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                c = float(r.boxes.conf[i])
                box_n = (x1 / w, y1 / h, x2 / w, y2 / h)
                self.last_box_px = (x1, y1, x2, y2, w, h)
                dist, src, how = self.measure((x1, y1, x2, y2), box_n, w, h, stamp_sec(msg.header))
                out.data = [1.0, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h,
                            c, dist, float(src), float(w), float(h), 1.0]
                self.draw_box(frame, (x1, y1, x2, y2), box_n, c, dist, src, how)
            else:
                self.last_box_px = None
                out.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, float(w), float(h), 1.0]
            self.pub_t.publish(out)
            self.publish_debug_image(msg, frame)
        finally:
            self.busy = False

    # 화면에 박스·Depth 영역·거리 글자를 그린다.
    def draw_box(self, frame, box, box_n, c, dist, src, how):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        if src == 1:
            rx1, ry1, rx2, ry2 = self.roi_rect(box_n)
            cv2.rectangle(frame, (int(rx1 * w), int(ry1 * h)), (int(rx2 * w), int(ry2 * h)),
                          (255, 200, 0), 2)
        tag = {1: '(depth)', 2: f'(box-{how})'}.get(src, '')
        txt = f'turtlebot {c:.2f}  ' + (f'{dist:.2f} m {tag}' if dist >= 0 else 'dist ?')
        cv2.putText(frame, txt, (int(x1), max(15, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 0), 2)

    # FPS 표시 후 디버그 영상(JPEG) 발행, --show 면 창에 표시 ('c' 보정, 'q' 종료).
    def publish_debug_image(self, msg, frame):
        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self.t_prev, 1e-6))
        self.t_prev = now
        dep_ok = 'depth OK' if self.depth is not None else 'NO DEPTH'
        cv2.putText(frame, f'{self.tag} | FPS {self.fps:.1f} | {dep_ok} | c: calib {self.a.calib_dist}m',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        img = CompressedImage()
        img.header = msg.header
        img.format = 'jpeg'
        img.data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        self.pub_i.publish(img)
        if self.a.show:
            cv2.imshow(f'tb_detect {self.tag} (q: quit, c: calib)', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                self.calibrate()
            elif key == ord('q'):
                raise KeyboardInterrupt


# [터틀봇 모드] 명령행 인자 해석 (tb_detect_node(1).py 와 같은 옵션 + robot, --always).
def tb_parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('robot', nargs='?', default='all', help='1 / 2 / robot6 / robot11 / all (카메라를 쓰는 로봇 B)')
    ap.add_argument('--model', default='', help='기본: ~/yolo_train 또는 ~/Downloads 의 turtlebot_v8n_best.pt')
    ap.add_argument('--topic', default='', help='RGB 토픽 직접 지정 (로봇 1대일 때)')
    ap.add_argument('--depth-topic', default='', help='Depth 토픽 직접 지정 (로봇 1대일 때)')
    ap.add_argument('--info-topic', default='', help='camera_info 토픽 직접 지정 (로봇 1대일 때)')
    ap.add_argument('--conf', type=float, default=0.7)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0', help="GPU 번호, CPU면 'cpu'")
    ap.add_argument('--dist-mode', choices=['auto', 'depth', 'box'], default='auto')
    ap.add_argument('--roi-pos', choices=['bottom', 'center'], default='bottom')
    ap.add_argument('--roi', type=float, default=0.5)
    ap.add_argument('--roi-h', type=float, default=0.2)
    ap.add_argument('--bottom-margin', type=float, default=0.05)
    ap.add_argument('--max-dt', type=float, default=0.6)
    ap.add_argument('--tb-height', type=float, default=0.35, help='보정 전 기본 로봇 높이(m)')
    ap.add_argument('--tb-width', type=float, default=0.34, help='보정 전 기본 로봇 폭(m)')
    ap.add_argument('--edge', type=int, default=3, help='화면 끝에서 이 픽셀 안이면 잘린 것으로 봄')
    ap.add_argument('--vfov', type=float, default=55.0, help='camera_info 가 없을 때 쓸 세로 시야각(도)')
    ap.add_argument('--calib-dist', type=float, default=1.0, help="'c' 보정 때 실제 거리(m)")
    ap.add_argument('--calib-file', default='', help='기본: ~/yolo_train/tb_dist_calib_<로봇>.txt')
    ap.add_argument('--offset', type=float, default=0.0, help='카메라~로봇 B 앞면 거리(m)')
    ap.add_argument('--always', action='store_true', help='robot_follow 와 상관없이 항상 인식 (시험·보정용)')
    ap.add_argument('--show', action='store_true')
    return ap.parse_args(argv)


# 'all' / 1 / 2 / 이름 → 로봇 이름 목록 (모르는 값이면 None).
def robots_from_arg(arg):
    if arg.lower() in ('a', 'all'):
        return list(ROBOTS.values())
    r = ROBOTS.get(arg, arg if arg in ROBOTS.values() or arg.startswith('robot') else None)
    return [r] if r else None


# [터틀봇 모드] 로봇마다 TbDetect 노드를 만들어 한 executor 로 실행 (모델은 공유).
def tb_main(argv):
    a = tb_parse_args(argv)
    robots = robots_from_arg(a.robot)
    if robots is None:
        print(f'알 수 없는 로봇 "{a.robot}" (1, 2, robot6, robot11, all)')
        return
    if len(robots) > 1 and (a.topic or a.depth_topic or a.info_topic):
        print('--topic / --depth-topic / --info-topic 은 로봇 1대일 때만 쓸 수 있어요.')
        return
    model_path = find_model(a.model)
    model = YOLO(model_path)
    print(f'>>> TurtleBot 모델: {model_path} | 로봇: {robots}', flush=True)
    rclpy.init()
    nodes = [TbDetect(a, f'/{r}', model) for r in robots]
    ex = SingleThreadedExecutor()
    for n in nodes:
        ex.add_node(n)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        for n in nodes:
            n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# 실행 진입점: 첫 인자가 tb / turtlebot 이면 터틀봇 모드, 아니면 사람 모드.
def main():
    argv = remove_ros_args(sys.argv)[1:]
    if argv and argv[0].lower() in ('tb', 'turtlebot'):
        tb_main(argv[1:])
    else:
        person_main(argv)


if __name__ == '__main__':
    main()
