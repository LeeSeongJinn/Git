#!/usr/bin/env python3
"""추종 중 '물리적 정지' 기록을 표로 정리 (worker_follow 가 남긴 ~/follow_stop_log_<robot>.csv)

  ros2 run my_package stop_report            # robot6
  ros2 run my_package stop_report robot11
  ros2 run my_package stop_report robot6 --last 20   # 최근 20건만
"""
import argparse
import csv
import os
import sys

# 원인 분류 → 큰 묶음 (가까워서 / 대상을 잃어서 / 그 밖)
GROUP = [
    ('가까워서 정지', '가까워서'),
    ('대상 소실', '대상을 잃어서'),
]


def group_of(cause):
    for prefix, g in GROUP:
        if cause.startswith(prefix):
            return g
    return '그 밖 (일시정지·센서·Nav2 등)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('robot', nargs='?', default='robot6')
    ap.add_argument('--last', type=int, default=0, help='최근 N 건만 (0 = 전체)')
    a = ap.parse_args([x for x in sys.argv[1:] if not x.startswith('--ros-args')])
    path = os.path.expanduser(f'~/follow_stop_log_{a.robot}.csv')
    if not os.path.exists(path):
        print(f'기록 없음: {path}  (worker_follow 를 실행하고 추종하다 멈추면 생김)')
        return
    with open(path, newline='') as fp:
        rows = list(csv.DictReader(fp))
    if a.last > 0:
        rows = rows[-a.last:]
    if not rows:
        print('기록이 비어 있습니다.')
        return

    print(f'\n## {a.robot} 추종 정지 기록 ({len(rows)}건)  — {path}\n')
    print('| # | 시각 | 큰 분류 | 정지 원인 | 상세 | LiDAR 거리(m) | 카메라 마지막 인식(초 전) | 모드 | 주행 후(초) |')
    print('|---|---|---|---|---|---|---|---|---|')
    for i, r in enumerate(rows, 1):
        print(f"| {i} | {r['time']} | {group_of(r['cause'])} | {r['cause']} | {r['detail']} | "
              f"{r['lidar_distance_m'] or '-'} | {r['camera_last_seen_s'] or '-'} | {r['mode']} | {r['moved_s']} |")

    print('\n### 원인별 횟수\n')
    print('| 큰 분류 | 정지 원인 | 횟수 | 비율 |')
    print('|---|---|---|---|')
    counts = {}
    for r in rows:
        k = (group_of(r['cause']), r['cause'])
        counts[k] = counts.get(k, 0) + 1
    for (g, c), n in sorted(counts.items(), key=lambda kv: (kv[0][0], -kv[1])):
        print(f'| {g} | {c} | {n} | {100.0 * n / len(rows):.0f}% |')
    near = sum(n for (g, _), n in counts.items() if g == '가까워서')
    lost = sum(n for (g, _), n in counts.items() if g == '대상을 잃어서')
    print(f'\n요약: 가까워서 정지 {near}회 / 대상을 잃어서 정지 {lost}회 / 그 밖 {len(rows) - near - lost}회')


if __name__ == '__main__':
    main()
