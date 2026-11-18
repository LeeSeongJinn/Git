#!/usr/bin/env python3
"""
[웹캠 PC에서 실행] 웹캠으로 작업자를 인식(YOLO Pose + 상·하의 색 특징)하고 결과를 로봇으로 보내는 노드

- 인식 로직은 pose_clothing_test.py 와 같다 (관절로 상의/하의 영역 → HSV 히스토그램 → 재식별).
- 결과 형식은 tb_detect_node.py 의 target 형식을 그대로 따르고, 뒤에 3칸을 덧붙였다.

발행:  <ns>/worker_detect/target   (std_msgs/Float32MultiArray)
          data = [찾음(1/0), 중심x, 중심y, 폭, 높이 (0~1 비율), 유사도점수, 거리(-1: 모름), 거리출처(0),
                  영상폭(px), 영상높이(px), 작업자등록여부(1/0)]
          → 매 프레임 발행 (못 찾아도 찾음=0 으로 발행: 받는 쪽이 카메라 살아있음을 앎)
          → 찾음/등록여부는 웹에서 '맞음'을 누른(확정된) 작업자일 때만 1
       <ns>/worker_detect/image/compressed  (박스가 그려진 영상, 디버그용)
       <ns>/user_register/image   (sensor_msgs/CompressedImage)  등록 순간 사진 → 웹 페이지3
       <ns>/user_track/following  (std_msgs/Bool)  확정된 작업자가 보이는지 → 웹 '인식 중/미인식'
구독:  <ns>/web_command             (std_msgs/String JSON) 자동 모드: 'undock' = 웹에서 이 로봇 선택
       <ns>/worker_detect/command   (std_msgs/String)  'register' | 'clear'
       <ns>/user_register/reply   (std_msgs/String)  웹 '맞음'='y' → 확정 / '아님'='n' → 거절
       <ns>/user_register/retry   (std_msgs/Bool)    웹 재인식 → 다시 등록 대기
       <ns>/user_register/cancel  (std_msgs/Bool)    웹 취소·구역 이동·작업 종료 → 등록 해제

로봇 자동 선택 (번호 없이 실행 = 자동 모드)
  robot6 / robot11 의 토픽을 모두 듣다가, 웹에서 로봇을 선택하면(web_command 'undock')
  그 로봇을 '현재 로봇'으로 정하고 결과를 그 로봇의 <ns> 로만 발행한다.
  웹에서 재인식(retry) 이 온 로봇도 현재 로봇으로 바뀐다.
  선택 전에는 등록하지 않는다 (화면: WAITING FOR ROBOT SELECTION).
  번호(1/2)를 주면 그 로봇에 고정 (이전 동작).

등록 흐름 (웹 연동)
  등록 대기(READY) --손 들기/r--> 사진 발행, 확인 대기(PENDING)
     --'y'--> 확정(CONFIRMED: 이때부터 추종)   --'n'--> 거절(REJECTED: 재인식/취소 대기)
     --confirm_timeout 초 안에 응답 없음--> 등록 대기
  cancel / retry / clear 는 언제든 등록 대기로 되돌림
  --no-web 이면 웹 확인 없이 바로 확정 (이전 동작, 단독 테스트용)

등록 방법
  - 손을 머리 위로 약 0.5초 들기
  - 화면 창에서 r (가장 큰 사람 등록), c (등록 취소), q (종료)   ※ --show 일 때
  - 토픽: ros2 topic pub --once /robot6/worker_detect/command std_msgs/String "{data: register}"

실행 (웹캠 PC, 로봇과 같은 ROS_DOMAIN_ID / Discovery 설정 필요)
  python3 worker_detect_node.py --show              # 자동: 웹에서 선택한 로봇으로 발행
  python3 worker_detect_node.py 2 --show --cam 0    # robot11 고정
  python3 worker_detect_node.py 1 --show --no-web    # 웹 없이 단독 테스트
"""
import argparse
import json
import sys
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32MultiArray, String
from ultralytics import YOLO

ROBOTS = {'1': 'robot6', '2': 'robot11'}

# --- pose_clothing_test.py 와 같은 설정값 ---
KP_CONF = 0.5            # 관절 신뢰도가 이 값보다 낮으면 안 보이는 것으로 처리
MATCH_THRESHOLD = 0.6    # 이 점수 이상이어야 작업자로 판단
UPDATE_THRESHOLD = 0.75  # 이 점수 이상일 때만 등록 특징을 갱신
UPDATE_RATE = 0.1        # 갱신 비율
RAISE_FRAMES = 15        # 손을 이만큼 연속 프레임 들어야 등록 (약 0.5초)

L_SH, R_SH = 5, 6
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KN, R_KN = 13, 14

# 등록 상태
ST_READY = 'READY'            # 등록 대기 (손 들기 / r)
ST_PENDING = 'PENDING'        # 사진 보냄, 웹 확인 대기
ST_REJECTED = 'REJECTED'      # 웹에서 '아님' → 재인식/취소 대기
ST_CONFIRMED = 'CONFIRMED'    # 웹에서 '맞음' → 추종 대상


# ---------------------------------------------------------------------------
# pose_clothing_test.py 의 함수 (동일)
# ---------------------------------------------------------------------------
def visible(conf, *ids):
    return all(conf[i] > KP_CONF for i in ids)


def clip_rect(x1, y1, x2, y2, w, h):
    x1, x2 = sorted((int(x1), int(x2)))
    y1, y2 = sorted((int(y1), int(y2)))
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return x1, y1, x2, y2


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


def color_hist(frame, rect):
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
    return hist


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


def blend(old, new):
    if old is None:
        return None if new is None else new.copy()
    if new is None:
        return old
    out = (1 - UPDATE_RATE) * old + UPDATE_RATE * new
    cv2.normalize(out, out, 1, 0, cv2.NORM_L1)
    return out


def hand_raised(kp, conf):
    if not visible(conf, L_SH, R_SH):
        return False
    sy = min(kp[L_SH][1], kp[R_SH][1])
    sw = abs(kp[L_SH][0] - kp[R_SH][0])
    for wr in (L_WR, R_WR):
        if conf[wr] > KP_CONF and kp[wr][1] < sy - sw * 0.5:
            return True
    return False


def box_area(p):
    x1, y1, x2, y2 = p['box']
    return (x2 - x1) * (y2 - y1)


# ---------------------------------------------------------------------------
class WorkerDetect(Node):
    def __init__(self, a, robots, auto):
        super().__init__('worker_detect' if auto else f'worker_detect_{robots[0]}')
        self.a = a
        self.auto = auto
        self.robots = list(robots)
        self.active = None if auto else self.robots[0]   # 결과를 보낼 로봇
        self.model = YOLO(a.model)
        self.pubs = {}
        for r in self.robots:
            ns = f'/{r}'
            self.pubs[r] = {
                't': self.create_publisher(Float32MultiArray, f'{ns}/worker_detect/target', 10),
                'i': self.create_publisher(
                    CompressedImage, f'{ns}/worker_detect/image/compressed', 10),
                'reg': self.create_publisher(CompressedImage, f'{ns}/user_register/image', 10),
                'follow': self.create_publisher(Bool, f'{ns}/user_track/following', 10),
            }
            self.create_subscription(String, f'{ns}/worker_detect/command',
                                     lambda m, r=r: self.on_command(r, m), 10)
            self.create_subscription(String, f'{ns}/user_register/reply',
                                     lambda m, r=r: self.on_reply(r, m), 10)
            self.create_subscription(Bool, f'{ns}/user_register/retry',
                                     lambda m, r=r: self.on_retry(r, m), 10)
            self.create_subscription(Bool, f'{ns}/user_register/cancel',
                                     lambda m, r=r: self.on_cancel(r, m), 10)
            if auto:
                self.create_subscription(String, f'{ns}/web_command',
                                         lambda m, r=r: self.on_web_command(r, m), 10)
        self.feat = None              # 등록된 작업자 특징 (PENDING / CONFIRMED)
        self.state = ST_READY
        self.pending_since = 0.0
        self.raise_count = 0
        self.pending_cmd = None
        self.last_following = None
        self.last_following_pub = 0.0
        self.get_logger().info(
            f'{"자동 모드: 웹에서 선택한 로봇으로 발행 " + str(self.robots) if auto else "로봇 고정: " + self.active}'
            f'  (카메라 {a.cam}, 모델 {a.model}, 웹 확인 {"끔" if a.no_web else "켬"})')

    # ---------- 현재 로봇 ----------
    def select_robot(self, r, why):
        if r == self.active:
            return
        old = self.active
        if old is not None:
            # 이전 로봇에는 '작업자 없음' 을 한 번 보내 즉시 멈추게 함
            out = Float32MultiArray()
            out.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]
            self.pubs[old]['t'].publish(out)
        self.active = r
        self.set_state(ST_READY, f'(로봇 변경 {old} -> {r}: {why})')
        self.get_logger().info(f'현재 로봇: {r} ({why})')

    def on_web_command(self, r, msg):
        try:
            c = json.loads(msg.data)
        except ValueError:
            return
        if isinstance(c, dict) and c.get('command') == 'undock':
            self.select_robot(r, '웹에서 로봇 선택')

    # ---------- 등록 상태 ----------
    def set_state(self, st, why=''):
        if st != self.state:
            self.get_logger().info(f'등록 상태 {self.state} -> {st} {why}')
        self.state = st
        if st in (ST_READY, ST_REJECTED):
            self.feat = None
            self.raise_count = 0
        if st != ST_CONFIRMED:
            self.last_following = None

    def on_command(self, robot, msg):
        cmd = msg.data.strip().lower()
        if robot == self.active and cmd in ('register', 'clear'):
            self.pending_cmd = cmd

    def on_reply(self, robot, msg):
        r = msg.data.strip().lower()
        if robot != self.active:
            self.get_logger().warning(f'{robot} 웹 응답 "{r}" 무시 (현재 로봇 {self.active})')
            return
        if self.state != ST_PENDING:
            self.get_logger().warning(f'웹 응답 "{r}" 무시 (상태 {self.state})')
            return
        if r == 'y':
            self.set_state(ST_CONFIRMED, '(웹: 맞음)')
        elif r == 'n':
            self.set_state(ST_REJECTED, '(웹: 아님)')

    def on_retry(self, robot, msg):
        if not msg.data:
            return
        if self.auto and robot != self.active:
            self.select_robot(robot, '웹 재인식')
        elif robot == self.active:
            self.set_state(ST_READY, '(웹: 재인식)')

    def on_cancel(self, robot, msg):
        if msg.data and robot == self.active:
            self.set_state(ST_READY, '(웹: 취소)')

    def publish_jpeg(self, pub, frame, quality=60):
        img = CompressedImage()
        img.header.stamp = self.get_clock().now().to_msg()
        img.format = 'jpeg'
        img.data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
        pub.publish(img)

    def register(self, frame, person):
        """작업자 등록: 특징 저장 + 등록 순간 사진을 웹으로 발행."""
        self.feat = {k: (v.copy() if v is not None else None) for k, v in person['feat'].items()}
        self.raise_count = 0
        if self.a.no_web:
            self.set_state(ST_CONFIRMED, '(웹 확인 없이 확정)')
            return
        snap = frame.copy()
        x1, y1, x2, y2 = map(int, person['box'])
        cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 200, 0), 3)
        self.publish_jpeg(self.pubs[self.active]['reg'], snap, 80)
        self.pending_since = time.time()
        self.set_state(ST_PENDING, '(사진 발행, 웹 확인 대기)')

    def process(self, frame, key):
        h, w = frame.shape[:2]
        now = time.time()
        r = self.model(frame, conf=self.a.conf, device=self.a.device, verbose=False)[0]
        people = []
        if len(r.boxes) > 0 and r.keypoints is not None and r.keypoints.conf is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            kps = r.keypoints.xy.cpu().numpy()
            kcs = r.keypoints.conf.cpu().numpy()
            for box, kp, kc in zip(boxes, kps, kcs):
                up, lo = get_regions(kp, kc, frame.shape)
                people.append({
                    'box': box, 'kp': kp, 'kc': kc, 'up': up, 'lo': lo,
                    'feat': {'upper': color_hist(frame, up), 'lower': color_hist(frame, lo)},
                    'raised': hand_raised(kp, kc),
                })

        # --- 등록 / 취소 ---
        if self.state == ST_PENDING and now - self.pending_since > self.a.confirm_timeout:
            self.set_state(ST_READY, '(웹 응답 시간 초과)')
        if key == ord('c') or self.pending_cmd == 'clear':
            self.set_state(ST_READY, '(등록 취소)')
        if self.state == ST_READY and self.active is not None:
            usable = [p for p in people if p['feat']['upper'] is not None]
            raisers = [p for p in usable if p['raised']]
            self.raise_count = self.raise_count + 1 if raisers else 0
            chosen = None
            if self.raise_count >= RAISE_FRAMES:
                chosen = max(raisers, key=box_area)
            elif (key == ord('r') or self.pending_cmd == 'register') and usable:
                chosen = max(usable, key=box_area)
            if chosen is not None:
                self.register(frame, chosen)
        self.pending_cmd = None

        # --- 매칭 (확인 대기 중에도 화면에는 표시) ---
        best, best_score = None, 0.0
        if self.feat is not None:
            for p in people:
                p['score'] = similarity(self.feat, p['feat'])
                if p['score'] > best_score:
                    best, best_score = p, p['score']
            if (self.state == ST_CONFIRMED and best is not None
                    and best_score >= UPDATE_THRESHOLD):
                self.feat['upper'] = blend(self.feat['upper'], best['feat']['upper'])
                self.feat['lower'] = blend(self.feat['lower'], best['feat']['lower'])
            if best_score < MATCH_THRESHOLD:
                best = None

        # --- 발행 (tb_detect 형식 + 영상크기 + 등록여부) : 확정된 작업자만 ---
        confirmed = self.state == ST_CONFIRMED
        out = Float32MultiArray()
        if confirmed and best is not None:
            x1, y1, x2, y2 = [float(v) for v in best['box']]
            out.data = [1.0, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h,
                        float(best_score), -1.0, 0.0, float(w), float(h), 1.0]
        else:
            out.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, float(w), float(h),
                        1.0 if confirmed else 0.0]
        if self.active is not None:
            self.pubs[self.active]['t'].publish(out)

        # --- 추종 중 여부 (웹): 확정 상태에서 바뀔 때 + 1초마다 ---
        if confirmed:
            seen = best is not None
            if seen != self.last_following or now - self.last_following_pub >= 1.0:
                m = Bool()
                m.data = seen
                self.pubs[self.active]['follow'].publish(m)
                self.last_following, self.last_following_pub = seen, now

        # --- 화면 ---
        for p in people:
            x1, y1, x2, y2 = map(int, p['box'])
            is_t = p is best
            color = ((0, 200, 0) if confirmed else (0, 200, 255)) if is_t else (150, 150, 150)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_t else 1)
            label = ('WORKER' if is_t else '') + (f" {p['score']:.2f}" if 'score' in p else '')
            cv2.putText(frame, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 0, 255), 1)   # 화면 중앙선
        if self.active is None:
            status = 'WAITING FOR ROBOT SELECTION ON WEB'
        elif self.state == ST_READY:
            status = f'WAITING - raise hand ({self.raise_count}/{RAISE_FRAMES}) or r'
        elif self.state == ST_PENDING:
            left = max(0, int(self.a.confirm_timeout - (now - self.pending_since)))
            status = f'CONFIRM ON WEB ({left}s)'
        elif self.state == ST_REJECTED:
            status = 'REJECTED - waiting retry/cancel on web'
        else:
            status = 'TRACKING' if best is not None else 'LOST'
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if self.active is not None:
            cv2.putText(frame, f'-> {self.active}', (w - 150, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if self.a.pub_image and self.active is not None:
            self.publish_jpeg(self.pubs[self.active]['i'], frame)
        return frame


def choose_robot(arg):
    """'' / 'auto' → None (자동 모드), 1/2/이름 → 그 로봇, 잘못된 값 → False."""
    if not arg or arg.lower() in ('auto', 'a'):
        return None
    ns = ROBOTS.get(arg, arg if arg in ROBOTS.values() else None)
    if ns is None:
        print(f'알 수 없는 로봇 "{arg}". 1, 2, {list(ROBOTS.values())} 또는 auto')
        return False
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('robot', nargs='?', default='',
                    help='1 / 2 / robot6 / robot11 (없으면 자동: 웹에서 선택한 로봇)')
    ap.add_argument('--cam', type=int, default=0, help='웹캠 번호')
    ap.add_argument('--model', default='yolov8n-pose.pt')
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--device', default='cpu', help="GPU면 '0'")
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--no-image', dest='pub_image', action='store_false', help='영상 토픽 발행 안 함')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--no-web', dest='no_web', action='store_true',
                    help='웹 확인 없이 등록 즉시 확정 (단독 테스트)')
    ap.add_argument('--confirm-timeout', type=float, default=15.0,
                    help='웹 맞음/아님 응답을 기다리는 시간(초), 지나면 등록 대기로')
    a = ap.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    ns = choose_robot(a.robot)
    if ns is False:
        return
    auto = ns is None
    if auto and a.no_web:
        print('--no-web 은 로봇 번호가 필요합니다. 예: python3 worker_detect_node.py 1 --show --no-web')
        return
    cap = cv2.VideoCapture(a.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        print(f'웹캠 {a.cam} 을 열 수 없습니다.')
        return

    rclpy.init()
    node = WorkerDetect(a, list(ROBOTS.values()) if auto else [ns], auto)
    t_prev = time.time()
    try:
        while rclpy.ok():
            ok, frame = cap.read()
            if not ok:
                node.get_logger().error('카메라 프레임을 읽을 수 없습니다.')
                break
            key = (cv2.waitKey(1) & 0xFF) if a.show else 255
            frame = node.process(frame, key)
            rclpy.spin_once(node, timeout_sec=0.0)
            if a.show:
                now = time.time()
                fps = 1.0 / max(now - t_prev, 1e-6)
                t_prev = now
                cv2.putText(frame, f'FPS {fps:.1f}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow('worker detect (r: register, c: clear, q: quit)', frame)
                if key == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
