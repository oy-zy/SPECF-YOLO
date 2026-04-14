# Z-depth debug version generated from stm32_qt_python_udp7(4).py
import cv2
import time
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from stereoo import camera_configs
import struct
import socket
import json
import torch
import os
from datetime import datetime


# =========================
# 可调参数
# =========================
REQUEST_WIDTH = 1280           # 参照 stm32_1 的摄像头请求宽度
REQUEST_HEIGHT = 480           # 参照 stm32_1 的摄像头请求高度
REQUEST_FPS = 30               # 参照 stm32_1 的摄像头请求帧率
DEPTH_UPDATE_INTERVAL = 1      # 精度优先：长度/速度测量建议每帧更新点云
FULL_POINTCLOUD_REFRESH_INTERVAL = 5  # 每隔几帧强制整帧重算一次点云，避免 ROI 累积误差
JPEG_QUALITY = 30
TRACKER_CFG = "bytetrack.yaml"  # 比 botsort 更快
MODEL_IMGSZ = 640
CONF_THRES = 0.25
IOU_THRES = 0.45

# =========================
# 测量参数（新增）
# =========================
PC_SAMPLE_RADIUS = 2           # 3D 点取样窗口半径，减小单点噪声
PC_MIN_ABS_Z = 80.0            # 稳定深度版：按近距离场景收紧有效 Z 下限（mm）
PC_MAX_ABS_Z = 800.0           # 稳定深度版：按近距离场景收紧有效 Z 上限（mm）
LENGTH_SAMPLE_COUNT = 21        # 长度沿线采样点数
LENGTH_SCAN_Y_RATIOS = (0.20, 0.35, 0.50, 0.65, 0.80)   # 多条扫描线，降低框偏移影响
LENGTH_MIN_VALID_POINTS = 4
LENGTH_SMOOTH_ALPHA = 0.35     # 长度 EMA 平滑
LENGTH_JUMP_THRESHOLD_CM = 10.0  # 长度突变阈值，超过则沿用上一帧长度
MAX_SEGMENT_MM = 250.0         # 相邻采样点允许的最大 3D 跳变
NEAR_SHRINK_RATIO = 0.05       # 近距离时左右收缩比例
MID_SHRINK_RATIO = 0.03
FAR_SHRINK_RATIO = 0.02
VERY_FAR_SHRINK_RATIO = 0.01
LENGTH_BOX_TOP_BOTTOM_SHRINK = 0.05  # 上下轻微收缩，减少背景带入
SPEED_HISTORY_SEC = 0.35       # 速度估计窗口
SPEED_MIN_DT = 0.08            # 速度估计最小时间间隔
SPEED_SMOOTH_ALPHA = 0.35      # 速度 EMA 平滑
TRACK_POS_SMOOTH_ALPHA = 0.45  # 3D 位置 EMA 平滑
MAX_SPEED_CM_S = 400.0         # 速度异常值抑制
DIRECTION_MIN_DX = 3.0         # 左右方向判定最小像素位移，抑制抖动

# =========================
# 本地存储参数（新增）
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
DETECTION_LOG_PATH = os.path.join(STORAGE_DIR, "crossing_events.txt")
COUNT_LOG_PATH = os.path.join(STORAGE_DIR, "object_counts_realtime.txt")
SAVE_ONLY_WHEN_CROSSING_CENTER = True   # 只有目标框碰到中线时才保存视频帧
TRIGGER_LINE_COLOR = (0, 255, 255)
TRIGGER_LINE_THICKNESS = 2
CENTER_DETECT_BAND_HALF_WIDTH = 100  # 只让 YOLO 看中线附近的窄区域
POINTCLOUD_UPDATE_LINE_MARGIN = 60    # 目标靠近中线多少像素时，触发点云更新
POINTCLOUD_ROI_MIN_WIDTH = 256        # 稳定深度版：适当放大 ROI 最小宽度，降低局部匹配失稳
POINTCLOUD_ROI_MIN_HEIGHT = 96        # ROI 立体匹配最小高度
DEPTH_SAMPLE_GRID_X = 5                # 框内深度采样列数
DEPTH_SAMPLE_GRID_Y = 3                # 框内深度采样行数
DEPTH_BOX_SHRINK_X = 0.20              # 深度取样时左右收缩，避开背景
DEPTH_BOX_SHRINK_Y_TOP = 0.20          # 深度取样时上边收缩比例
DEPTH_BOX_SHRINK_Y_BOTTOM = 0.10       # 深度取样时下边收缩比例
DEPTH_MIN_VALID_SAMPLES = 4            # 至少需要多少个有效采样点
DEPTH_SMOOTH_ALPHA = 0.30              # 深度 EMA 平滑
DEPTH_JUMP_THRESHOLD_CM = 15.0         # 突变抑制阈值（cm）



def ensure_storage_dir():
    os.makedirs(STORAGE_DIR, exist_ok=True)


def ensure_detection_log_header(log_path=DETECTION_LOG_PATH):
    ensure_storage_dir()
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write("time\tcamera_id\ttracker_id\tclass\tsize_label\tdirection\tdist_cm\tspeed_cm_s\tlength_cm\ttrigger_line_x\tevent\n")


def append_detection_log(camera_id, tracker_id, cls_name, size_label, direction, dist, speed, length, trigger_line_x, event='cross_center_line', log_path=DETECTION_LOG_PATH):
    ensure_detection_log_header(log_path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(
            f"{now_str}\t{camera_id}\t{tracker_id}\t{cls_name}\t{size_label}\t{direction}\t{dist}\t{speed}\t{length}\t{trigger_line_x}\t{event}\n"
        )


def write_realtime_count_log(current_number, total_number, log_path=COUNT_LOG_PATH):
    ensure_storage_dir()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"update_time: {now_str}\n")
        f.write("[Current Counts]\n")
        if current_number:
            for item in current_number:
                f.write(f"{item['class']}: {item['number']}\n")
        else:
            f.write("None: 0\n")

        f.write("\n[Total Counts]\n")
        if total_number:
            for item in total_number:
                f.write(f"{item['class']}: {item['number']}\n")
        else:
            f.write("None: 0\n")

def bbox_crosses_vertical_line(p1, p2, line_x):
    return int(p1[0]) <= int(line_x) <= int(p2[0])


def box_near_center_line(p1, p2, line_x, margin=POINTCLOUD_UPDATE_LINE_MARGIN):
    return int(p1[0]) <= int(line_x + margin) and int(p2[0]) >= int(line_x - margin)


def get_stereo_roi_box(p1, p2, frame_w, frame_h, min_w=POINTCLOUD_ROI_MIN_WIDTH, min_h=POINTCLOUD_ROI_MIN_HEIGHT):
    x1, y1 = safe_xy(p1[0], p1[1], frame_w, frame_h)
    x2, y2 = safe_xy(p2[0], p2[1], frame_w, frame_h)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    roi_w = max(x2 - x1 + 1, min_w)
    roi_h = max(y2 - y1 + 1, min_h)

    rx1 = int(round(cx - roi_w / 2.0))
    ry1 = int(round(cy - roi_h / 2.0))
    rx2 = rx1 + roi_w - 1
    ry2 = ry1 + roi_h - 1

    if rx1 < 0:
        rx2 -= rx1
        rx1 = 0
    if ry1 < 0:
        ry2 -= ry1
        ry1 = 0
    if rx2 >= frame_w:
        shift = rx2 - frame_w + 1
        rx1 -= shift
        rx2 = frame_w - 1
    if ry2 >= frame_h:
        shift = ry2 - frame_h + 1
        ry1 -= shift
        ry2 = frame_h - 1

    rx1 = max(0, rx1)
    ry1 = max(0, ry1)
    rx2 = min(frame_w - 1, rx2)
    ry2 = min(frame_h - 1, ry2)

    return rx1, ry1, rx2, ry2


def getPointClouds_roi(images_left_rectified, images_right_rectified, stereo, roi_box):
    rx1, ry1, rx2, ry2 = roi_box
    left_roi = images_left_rectified[ry1:ry2 + 1, rx1:rx2 + 1]
    right_roi = images_right_rectified[ry1:ry2 + 1, rx1:rx2 + 1]

    if left_roi.size == 0 or right_roi.size == 0:
        return None

    imgL = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(imgL, imgR)

    q_roi = camera_configs.Q.copy().astype(np.float32)
    q_roi[0, 3] += float(rx1)
    q_roi[1, 3] += float(ry1)

    pointClouds_roi = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16.0, q_roi)
    return pointClouds_roi


def update_pointcloud_rois(cached_pointclouds, images_left_rectified, images_right_rectified, stereo, update_boxes):
    if cached_pointclouds is None:
        return cached_pointclouds

    frame_h, frame_w = cached_pointclouds.shape[:2]

    for p1, p2 in update_boxes:
        # 这里统一按 numpy 切片的左闭右开规则处理，避免 +1 带来的尺寸错位
        x1 = max(0, min(int(p1[0]), frame_w - 1))
        y1 = max(0, min(int(p1[1]), frame_h - 1))
        x2 = max(x1 + 1, min(int(p2[0]), frame_w))
        y2 = max(y1 + 1, min(int(p2[1]), frame_h))

        if x2 <= x1 or y2 <= y1:
            continue

        roi_box = get_stereo_roi_box((x1, y1), (x2 - 1, y2 - 1), frame_w, frame_h)
        roi_pointclouds = getPointClouds_roi(images_left_rectified, images_right_rectified, stereo, roi_box)
        if roi_pointclouds is None:
            continue

        rx1, ry1, rx2, ry2 = roi_box
        roi_h, roi_w = roi_pointclouds.shape[:2]

        local_x1 = max(0, x1 - rx1)
        local_y1 = max(0, y1 - ry1)
        local_x2 = min(roi_w, x2 - rx1)
        local_y2 = min(roi_h, y2 - ry1)

        copy_w = min(x2 - x1, local_x2 - local_x1)
        copy_h = min(y2 - y1, local_y2 - local_y1)
        if copy_w <= 0 or copy_h <= 0:
            continue

        cached_pointclouds[y1:y1 + copy_h, x1:x1 + copy_w] = roi_pointclouds[local_y1:local_y1 + copy_h, local_x1:local_x1 + copy_w]

    return cached_pointclouds

def get_size_label(length_cm):
    if length_cm < 20:
        return "Small"
    elif length_cm < 40:
        return "Medium"
    else:
        return "Large"


def get_direction_label(id, p1, p2, Cnt):
    if id == "Tracking":
        return "Unknown"

    cx = float((p1[0] + p2[0]) / 2.0)
    state = Cnt.setdefault(id, {})
    prev_cx = state.get('prev_center_x', None)
    direction = state.get('direction', 'Unknown')

    if prev_cx is not None:
        dx = cx - prev_cx
        if dx >= DIRECTION_MIN_DX:
            direction = "Upstream"
        elif dx <= -DIRECTION_MIN_DX:
            direction = "Downstream"

    state['prev_center_x'] = cx
    state['direction'] = direction
    return direction

def convert_to_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def safe_xy(x, y, w, h):
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))
    return x, y


def is_valid_3d_point(pt):
    if pt is None:
        return False
    if not np.isfinite(pt).all():
        return False
    z = abs(float(pt[2]))
    if z < PC_MIN_ABS_Z or z > PC_MAX_ABS_Z:
        return False
    return True


def robust_pointcloud_value(PointClouds, x, y, radius=PC_SAMPLE_RADIUS):
    if PointClouds is None:
        return None

    h, w = PointClouds.shape[:2]
    x, y = safe_xy(x, y, w, h)

    x1 = max(0, x - radius)
    x2 = min(w - 1, x + radius)
    y1 = max(0, y - radius)
    y2 = min(h - 1, y + radius)

    roi = PointClouds[y1:y2 + 1, x1:x2 + 1].reshape(-1, 3)
    if roi.size == 0:
        return None

    valid_mask = np.isfinite(roi).all(axis=1)
    if not np.any(valid_mask):
        return None

    pts = roi[valid_mask]
    z_abs = np.abs(pts[:, 2])
    pts = pts[(z_abs >= PC_MIN_ABS_Z) & (z_abs <= PC_MAX_ABS_Z)]
    if len(pts) < 3:
        return None

    median_pt = np.median(pts, axis=0)
    diff = np.linalg.norm(pts - median_pt, axis=1)
    mad = np.median(diff)

    if mad > 1e-6:
        pts = pts[diff <= max(2.5 * mad, 30.0)]
        if len(pts) < 3:
            return median_pt.astype(np.float32)

    return np.median(pts, axis=0).astype(np.float32)


def safe_pointcloud_value(PointClouds, x, y):
    val = robust_pointcloud_value(PointClouds, x, y, radius=PC_SAMPLE_RADIUS)
    if val is None:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return val


def rectify_stereo_pair(images_left, images_right):
    img1_rectified = cv2.remap(
        images_left,
        camera_configs.left_map1,
        camera_configs.left_map2,
        cv2.INTER_LINEAR
    )
    img2_rectified = cv2.remap(
        images_right,
        camera_configs.right_map1,
        camera_configs.right_map2,
        cv2.INTER_LINEAR
    )
    return img1_rectified, img2_rectified


def split_stereo_frame_dynamic(frame):
    h, w = frame.shape[:2]
    if w < 2:
        return None, None

    mid = w // 2
    left = frame[:, :mid]
    right = frame[:, mid:]

    # 两边宽度不一致时，裁到相同宽度，避免校正/视差错位
    common_w = min(left.shape[1], right.shape[1])
    if common_w <= 0:
        return None, None

    left = left[:, :common_w]
    right = right[:, :common_w]
    return left, right




def get_depth_adaptive_shrink_ratio(center_pt):
    if center_pt is None:
        return FAR_SHRINK_RATIO

    z_mm = abs(float(center_pt[2]))
    if z_mm < 800:
        return NEAR_SHRINK_RATIO
    elif z_mm < 1200:
        return MID_SHRINK_RATIO
    elif z_mm < 2000:
        return FAR_SHRINK_RATIO
    else:
        return VERY_FAR_SHRINK_RATIO


def shrink_box_for_length(p1, p2, PointClouds):
    x1, y1 = p1
    x2, y2 = p2
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return p1, p2

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    center_pt = robust_pointcloud_value(PointClouds, cx, cy)

    shrink_ratio = get_depth_adaptive_shrink_ratio(center_pt)
    dx = int(w * shrink_ratio)
    dy = int(h * LENGTH_BOX_TOP_BOTTOM_SHRINK)

    nx1 = x1 + dx
    nx2 = x2 - dx
    ny1 = y1 + dy
    ny2 = y2 - dy

    if nx2 <= nx1:
        nx1, nx2 = x1, x2
    if ny2 <= ny1:
        ny1, ny2 = y1, y2

    return (nx1, ny1), (nx2, ny2)


def should_update_length(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False

    ratio = w / float(h + 1e-6)
    if ratio < 1.2 or ratio > 8.0:
        return False
    return True


def centerPointDist(p1, p2, PointClouds):
    if PointClouds is None:
        return 0
    x = int((p1[0] + p2[0]) / 2)
    y = int((p1[1] + p2[1]) / 2)
    pt = robust_pointcloud_value(PointClouds, x, y)
    if pt is None:
        return 0
    return round(float(np.linalg.norm(pt)) / 10)


def shrink_box_for_depth(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return p1, p2

    nx1 = x1 + int(DEPTH_BOX_SHRINK_X * w)
    nx2 = x2 - int(DEPTH_BOX_SHRINK_X * w)
    ny1 = y1 + int(DEPTH_BOX_SHRINK_Y_TOP * h)
    ny2 = y2 - int(DEPTH_BOX_SHRINK_Y_BOTTOM * h)

    if nx2 <= nx1 or ny2 <= ny1:
        return p1, p2
    return (nx1, ny1), (nx2, ny2)


def centerPointDepth(p1, p2, PointClouds, state_dict=None, track_id=None):
    if PointClouds is None:
        return 0

    dp1, dp2 = shrink_box_for_depth(p1, p2)
    x1, y1 = dp1
    x2, y2 = dp2
    if x2 <= x1 or y2 <= y1:
        x1, y1 = p1
        x2, y2 = p2

    xs = np.linspace(x1, x2, DEPTH_SAMPLE_GRID_X)
    ys = np.linspace(y1, y2, DEPTH_SAMPLE_GRID_Y)

    z_list = []
    for yy in ys:
        for xx in xs:
            pt = robust_pointcloud_value(PointClouds, xx, yy)
            if pt is None:
                continue
            z_mm = abs(float(pt[2]))
            if PC_MIN_ABS_Z <= z_mm <= PC_MAX_ABS_Z:
                z_list.append(z_mm)

    if len(z_list) < DEPTH_MIN_VALID_SAMPLES:
        if state_dict is not None and track_id in state_dict and 'depth_z_cm' in state_dict[track_id]:
            return round(state_dict[track_id]['depth_z_cm'], 2)
        return 0

    raw_z_cm = float(np.median(z_list)) / 10.0

    if state_dict is not None and track_id is not None:
        st = state_dict.setdefault(track_id, {})
        prev_z = st.get('depth_z_cm', raw_z_cm)

        if abs(raw_z_cm - prev_z) > DEPTH_JUMP_THRESHOLD_CM:
            z_cm = prev_z
        else:
            z_cm = prev_z * (1.0 - DEPTH_SMOOTH_ALPHA) + raw_z_cm * DEPTH_SMOOTH_ALPHA

        st['depth_z_cm'] = z_cm
        return round(z_cm, 2)

    return round(raw_z_cm, 2)


def _estimate_length_by_scanline(p1, p2, PointClouds, y_ratio):
    x1, y1 = p1
    x2, y2 = p2
    if x2 <= x1 or y2 <= y1:
        return None

    y = int(y1 + (y2 - y1) * y_ratio)
    xs = np.linspace(x1, x2, LENGTH_SAMPLE_COUNT)

    samples = []
    for x in xs:
        pt = robust_pointcloud_value(PointClouds, x, y)
        samples.append((x, pt))

    valid_samples = [(x, pt) for x, pt in samples if pt is not None]
    if len(valid_samples) < LENGTH_MIN_VALID_POINTS:
        return None

    first_pt = valid_samples[0][1]
    last_pt = valid_samples[-1][1]
    total_length = 0.0
    prev_pt = None

    for _, pt in valid_samples:
        if prev_pt is not None:
            seg = float(np.linalg.norm(pt - prev_pt))
            if 0.0 < seg <= MAX_SEGMENT_MM:
                total_length += seg
        prev_pt = pt

    if total_length <= 0.0:
        total_length = float(np.linalg.norm(last_pt - first_pt))

    return total_length if total_length > 0 else None


def coord2dist(id, p1, p2, Cnt, PointClouds):
    if PointClouds is None:
        return 0

    candidates = []
    for ratio in LENGTH_SCAN_Y_RATIOS:
        length_mm = _estimate_length_by_scanline(p1, p2, PointClouds, ratio)
        if length_mm is not None:
            candidates.append(length_mm)

    if not candidates:
        x1, y1 = p1
        x2, y2 = p2
        y = int((y1 + y2) / 2)
        pt1 = robust_pointcloud_value(PointClouds, x1, y)
        pt2 = robust_pointcloud_value(PointClouds, x2, y)
        if pt1 is None or pt2 is None:
            if id != "Tracking" and id in Cnt and 'length' in Cnt[id]:
                return round(Cnt[id]['length'], 2)
            return 0
        raw_length_cm = float(np.linalg.norm(pt1 - pt2)) / 10.0
    else:
        candidates_cm = [c / 10.0 for c in candidates if 5.0 <= (c / 10.0) <= 200.0]

        if not candidates_cm:
            if id != "Tracking" and id in Cnt and 'length' in Cnt[id]:
                return round(Cnt[id]['length'], 2)
            return 0

        candidates_cm = sorted(candidates_cm)

        # 按要求：优先取 75 分位；候选太少时取最大稳定值
        if len(candidates_cm) >= 3:
            raw_length_cm = float(np.percentile(candidates_cm, 75))
        else:
            raw_length_cm = float(max(candidates_cm))

    if id == "Tracking":
        return round(raw_length_cm, 2)

    state = Cnt.setdefault(id, {})
    prev_length = state.get('length', raw_length_cm)

    # 按要求修改：如果当前长度相对上一帧突变超过 10 cm，则直接沿用上一帧长度
    if abs(raw_length_cm - prev_length) > LENGTH_JUMP_THRESHOLD_CM:
        raw_length_cm = prev_length

    smoothed_length = prev_length * (1.0 - LENGTH_SMOOTH_ALPHA) + raw_length_cm * LENGTH_SMOOTH_ALPHA
    state['length'] = smoothed_length
    return round(smoothed_length, 2)


def coord2speed(id, p1, p2, Cnt, PointClouds):
    if PointClouds is None:
        return 0

    p = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
    cur_pos_raw = robust_pointcloud_value(PointClouds, p[0], p[1])

    if id == "Tracking":
        return 0

    cur_time = time.time()
    state = Cnt.setdefault(id, {})

    if cur_pos_raw is None:
        state['time'] = cur_time
        return round(state.get('speed', 0.0))

    if 'pos' not in state:
        state['time'] = cur_time
        state['pos'] = cur_pos_raw
        state['history'] = [(cur_time, cur_pos_raw)]
        state['speed'] = 0.0
        return 0

    last_pos = state['pos']
    cur_pos = last_pos * (1.0 - TRACK_POS_SMOOTH_ALPHA) + cur_pos_raw * TRACK_POS_SMOOTH_ALPHA
    state['pos'] = cur_pos
    state['time'] = cur_time

    history = state.setdefault('history', [])
    history.append((cur_time, cur_pos.copy()))
    state['history'] = [(t, pos) for t, pos in history if cur_time - t <= max(SPEED_HISTORY_SEC, 1.0)]

    ref_item = None
    for t, pos in state['history']:
        if cur_time - t >= SPEED_MIN_DT:
            ref_item = (t, pos)
            break

    if ref_item is None:
        return round(state.get('speed', 0.0))

    ref_time, ref_pos = ref_item
    dt = cur_time - ref_time
    if dt <= 0:
        return round(state.get('speed', 0.0))

    move_cm = float(np.linalg.norm(cur_pos - ref_pos)) / 10.0
    raw_speed = move_cm / dt

    prev_speed = state.get('speed', 0.0)
    if raw_speed > MAX_SPEED_CM_S:
        raw_speed = prev_speed

    smoothed_speed = prev_speed * (1.0 - SPEED_SMOOTH_ALPHA) + raw_speed * SPEED_SMOOTH_ALPHA
    state['speed'] = smoothed_speed
    return round(smoothed_speed)


def getPointClouds(images_left_rectified, images_right_rectified, stereo):
    imgL = cv2.cvtColor(images_left_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(images_right_rectified, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(imgL, imgR)
    pointClouds = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16.0, camera_configs.Q)
    return pointClouds

def create_camera_context(camera_id, model_path, cls_names):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
    # 按 stm32_1 的方式设置摄像头读取参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REQUEST_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[Camera {camera_id}] 实际分辨率: {actual_width} x {actual_height}")
    print(f"[Camera {camera_id}] 驱动返回FPS: {actual_fps}")
    # 每个相机各自一个模型，避免 persist=True 时多路串轨迹
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to('cuda:0')
    # 只创建一次 SGBM
    stereo = cv2.StereoSGBM.create(
          minDisparity=0,
          numDisparities=176,
          blockSize=9,
          P1=8 * 3 * 9 * 9,
          P2=32 * 3 * 9 * 9,
          uniquenessRatio=10,
          speckleWindowSize=50,
          speckleRange=2,
          disp12MaxDiff=1
    )
    ensure_storage_dir()
    video_path = os.path.join(STORAGE_DIR, f'output_camera_{camera_id}.avi')
    videoWriter = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        10,
        (640, 480)
    )
    return {
        'camera_id': camera_id,
        'cap': cap,
        'driver_fps': float(actual_fps) if actual_fps and actual_fps > 0 else 0.0,
        'model': model,
        'stereo': stereo,
        'cls_name_temp_total': {},
        'track_ids_max': 0,
        'cls_ids_total': [0] * len(cls_names),
        'speed_cnt': Counter(),
        'videoWriter': videoWriter,
        'video_path': video_path,
        'log_path': DETECTION_LOG_PATH,
        'count_log_path': COUNT_LOG_PATH,
        'passed_class_counts': Counter(),
        'tracker_passed_info': {},
        'frame_idx': 0,
        'cached_pointClouds': None,
        'process_fps_ema': 0.0,
        'line_cross_state': {},
        'saved_frame_count': 0,
    }
def read_one_camera(camera_item):
    camera_id, camera_ctx = camera_item
    success, frame = camera_ctx['cap'].read()
    return camera_id, success, frame
def process_one_camera_frame(camera_ctx, frame, box_annotator, sock, server_address, cls_names):
    loop_start = cv2.getTickCount()

    model = camera_ctx['model']
    stereo = camera_ctx['stereo']
    cls_name_temp_total = camera_ctx['cls_name_temp_total']
    track_ids_max = camera_ctx['track_ids_max']
    cls_ids_total = camera_ctx['cls_ids_total']
    speed_cnt = camera_ctx['speed_cnt']
    videoWriter = camera_ctx['videoWriter']
    camera_id = camera_ctx['camera_id']
    log_path = camera_ctx['log_path']
    count_log_path = camera_ctx['count_log_path']
    passed_class_counts = camera_ctx['passed_class_counts']
    tracker_passed_info = camera_ctx['tracker_passed_info']
    prev_line_cross_state = camera_ctx.get('line_cross_state', {})
    current_line_cross_state = {}
    camera_fps = camera_ctx.get('driver_fps', 0.0)
    camera_ctx['frame_idx'] += 1
    frame_idx = camera_ctx['frame_idx']
    frame1_raw, frame2_raw = split_stereo_frame_dynamic(frame)
    if frame1_raw is None or frame2_raw is None:
        print(f"[Camera {camera_id}] 输入帧宽度异常，无法动态切分左右图")
        return
    # 先做双目校正：检测坐标与三维点云坐标必须在同一坐标系下，否则深度会偏
    frame1, frame2 = rectify_stereo_pair(frame1_raw, frame2_raw)
    line_x = frame1.shape[1] // 2

    cls_name_temp = {}
    Tabel_Of_Data = list()
    Current_Number = list()
    Total_Number = list()
    json_txt1 = {
        'Current_Number': [],
        'Total_Number': [],
    }

    # 为了保证长度测量准确，这里恢复为整帧检测；
    # 但只保留真正穿过中线的目标用于显示、保存和记录。
    results = model.track(
        source=frame1,
        tracker=TRACKER_CFG,
        persist=True,
        verbose=False,
        imgsz=MODEL_IMGSZ,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=0 if torch.cuda.is_available() else "cpu"
    )

    cls_id = [0] * len(cls_names)
    cnt = Counter()
    labels = []
    label_sets = set()

    result0 = results[0]
    boxes = result0.boxes

    cls_ids = []
    track_ids = 0
    detections = None
    position_ids = []
    update_boxes = []

    if boxes is not None and len(boxes) > 0 and boxes.xyxy is not None:
        xyxy = boxes.xyxy.cpu().numpy().copy()

        for idx in range(len(xyxy)):
            p1_all = (int(xyxy[idx][0]), int(xyxy[idx][1]))
            p2_all = (int(xyxy[idx][2]), int(xyxy[idx][3]))
            if box_near_center_line(p1_all, p2_all, line_x):
                update_boxes.append((p1_all, p2_all))

        cls_ids_all = boxes.cls.int().cpu().tolist() if boxes.cls is not None and len(boxes.cls) > 0 else []
        if boxes.is_track and boxes.id is not None:
            track_ids_all = boxes.id.int().cpu().tolist()
        else:
            track_ids_all = None

        keep_indices = []
        for idx in range(len(xyxy)):
            p1_keep = (int(xyxy[idx][0]), int(xyxy[idx][1]))
            p2_keep = (int(xyxy[idx][2]), int(xyxy[idx][3]))
            if bbox_crosses_vertical_line(p1_keep, p2_keep, line_x):
                keep_indices.append(idx)

        if keep_indices:
            filtered_xyxy = xyxy[keep_indices]
            cls_ids = [cls_ids_all[idx] for idx in keep_indices]

            for i in cls_ids:
                cls_id[i] += 1

            if track_ids_all is not None:
                track_ids = [track_ids_all[idx] for idx in keep_indices]
            else:
                track_ids = 0

            detections = sv.Detections(
                xyxy=filtered_xyxy,
                class_id=np.array(cls_ids, dtype=int)
            )
            if track_ids != 0 and len(track_ids) > 0:
                detections.tracker_id = np.array(track_ids, dtype=int)
            position_ids = filtered_xyxy

    # 按要求修改：每一帧都整帧全图更新点云
    # 不再使用“每5帧整帧刷新”策略，也不再使用“中线附近目标局部 ROI 更新”策略
    camera_ctx['cached_pointClouds'] = getPointClouds(frame1, frame2, stereo)
    pointClouds = camera_ctx['cached_pointClouds']

    depth_total = Counter()
    save_this_frame = False

    for i in range(len(cls_ids)):
        p1 = (int(position_ids[i][0]), int(position_ids[i][1]))
        p2 = (int(position_ids[i][2]), int(position_ids[i][3]))

        tracker_id = track_ids[i] if track_ids != 0 else "Tracking"
        depth = centerPointDepth(p1, p2, pointClouds, speed_cnt, tracker_id)

        # 显示稳定 Z 深度 + 长度；速度/方向仍保持调试占位
        dist = 0
        speed = 0
        size_label = "Debug"
        direction = "Debug"

        if should_update_length(p1, p2):
            lp1, lp2 = shrink_box_for_length(p1, p2, pointClouds)
            length = coord2dist(tracker_id, lp1, lp2, speed_cnt, pointClouds)
        else:
            if tracker_id != "Tracking" and tracker_id in speed_cnt and 'length' in speed_cnt[tracker_id]:
                length = round(speed_cnt[tracker_id]['length'], 2)
            else:
                length = 0

        cls = cls_names[cls_ids[i]]
        label_sets.add(cls)
        cnt[cls] += 1

        depth_total[cls] += depth

        tracker_key = f"{cls}_{tracker_id}" if tracker_id != "Tracking" else f"Tracking_{frame_idx}_{i}"
        is_crossing_line = bbox_crosses_vertical_line(p1, p2, line_x)
        prev_crossing = prev_line_cross_state.get(tracker_key, False)
        current_line_cross_state[tracker_key] = is_crossing_line

        if is_crossing_line:
            save_this_frame = True

        class_count = 0
        if tracker_id != "Tracking":
            if tracker_key in tracker_passed_info:
                class_count = tracker_passed_info[tracker_key]['count']
            if is_crossing_line and not prev_crossing and tracker_key not in tracker_passed_info:
                passed_class_counts[cls] += 1
                class_count = passed_class_counts[cls]
                tracker_passed_info[tracker_key] = {
                    'class': cls,
                    'count': class_count,
                }
                append_detection_log(camera_id, tracker_id, cls, size_label, direction, dist, speed, length, line_x, 'cross_center_line', log_path)
        else:
            class_count = cnt[cls]

        trigger_flag = " HIT" if is_crossing_line else ""
        labels.append("#{} {} count:{} Z:{}cm L:{}cm{}".format(tracker_id, cls, class_count, depth, length, trigger_flag))

        Tabel_Of_Data.append({
            "class": cls,
            "position": [p1, p2],
            "tracker": tracker_id,
            "class_count": class_count,
            "depth_z_cm": depth,
            "length_cm": length,
            "cross_center_line": is_crossing_line
        })

    for cls in label_sets:
        Current_Number.append({
            "class": cls,
            "number": cnt[cls],
            "depth_z_average_cm": depth_total[cls] / cnt[cls] if cnt[cls] > 0 else 0
        })

    for cls_name, passed_num in passed_class_counts.items():
        if passed_num > 0:
            Total_Number.append({
                "class": cls_name,
                "number": passed_num
            })

    json_txt1['Current_Number'] = Current_Number
    json_txt1['Total_Number'] = Total_Number
    write_realtime_count_log(Current_Number, Total_Number, count_log_path)

    for i in range(len(cls_id)):
        if cls_id[i] > 0:
            cls_name_temp[cls_names[i]] = cls_id[i]

    cls_name_temp_total.clear()
    for item in Total_Number:
        cls_name_temp_total[item['class']] = item['number']

    if track_ids != 0 and len(track_ids) > 0:
        track_ids_max = max(track_ids)

    # 保持你原来的控制台输出不变
    print(Current_Number)
    print(Total_Number)
    print(json_txt1)

    # 保持原来的 UDP 数据发送逻辑
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    result_encode, img_encode = cv2.imencode('.jpg', frame1, encode_param)
    if result_encode:
        data = img_encode.tobytes()
        size = len(data)

        json_bytes = json.dumps(json_txt1, default=convert_to_default).encode('utf-8')

        packet = bytearray()
        packet.extend(b'\xA1\xA2\xA3')
        packet.extend(struct.pack('!I', size))
        packet.extend(json_bytes)
        packet.extend(b'\xB1\xB2\xB3')
        sock.sendto(packet, server_address)

    annotated_frame = frame1.copy()

    loop_time = cv2.getTickCount() - loop_start
    total_time = loop_time / cv2.getTickFrequency()
    process_fps = (1.0 / total_time) if total_time > 0 else 0.0

    prev_process_fps = camera_ctx.get('process_fps_ema', 0.0)
    if prev_process_fps <= 0:
        camera_ctx['process_fps_ema'] = process_fps
    else:
        camera_ctx['process_fps_ema'] = prev_process_fps * 0.8 + process_fps * 0.2

    camera_fps_text = f"Camera FPS: {camera_fps:.2f}"
    process_fps_text = f"Process FPS: {camera_ctx['process_fps_ema']:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (0, 0, 255)
    camera_fps_position = (10, 30)
    process_fps_position = (10, 60)

    curItems = ""
    for key, val in cnt.items():
        curItems = curItems + "{} {},".format(val, key)

    cv2.putText(annotated_frame, curItems[:-2], (25, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200), 1)

    if detections is not None and len(labels) > 0:
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

    cv2.putText(annotated_frame, "Stable Z-depth + Length", (10, 95), font, 0.6, text_color, 2)
    y0, dy = 120, 20
    for idx, (key, value) in enumerate(cls_name_temp.items()):
        text = f"{key}: {value}"
        y = y0 + idx * dy
        cv2.putText(annotated_frame, text, (10, y), font, 0.5, text_color, 1)

    y0 += (len(cls_name_temp) + 1) * dy
    cv2.putText(annotated_frame, "total", (10, y0), font, 0.5, text_color, 1)
    y0 += dy
    for idx, (key, value) in enumerate(cls_name_temp_total.items()):
        text = f"{key}: {value}"
        y = y0 + idx * dy
        cv2.putText(annotated_frame, text, (10, y), font, 0.5, text_color, 1)

    cur_time = time.time()
    for key, dicts in list(speed_cnt.items()):
        if cur_time - dicts.get('time', cur_time) > 5:
            del speed_cnt[key]

    cv2.line(
        annotated_frame,
        (line_x, 0),
        (line_x, annotated_frame.shape[0] - 1),
        TRIGGER_LINE_COLOR,
        TRIGGER_LINE_THICKNESS
    )

    save_state_text = "SAVE: ON" if save_this_frame else "SAVE: OFF"
    cv2.putText(annotated_frame, save_state_text, (430, 30), font, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"L/R width: {frame1.shape[1]}", (430, 60), font, 0.6, (255, 255, 0), 2)
    cv2.putText(annotated_frame, camera_fps_text, camera_fps_position, font, font_scale, text_color, font_thickness)
    cv2.putText(annotated_frame, process_fps_text, process_fps_position, font, font_scale, text_color, font_thickness)
    cv2.imshow(f'img_{camera_id}', annotated_frame)

    if (not SAVE_ONLY_WHEN_CROSSING_CENTER) or save_this_frame:
        videoWriter.write(annotated_frame)
        camera_ctx['saved_frame_count'] = camera_ctx.get('saved_frame_count', 0) + 1

    camera_ctx['line_cross_state'] = current_line_cross_state
    camera_ctx['track_ids_max'] = track_ids_max


if __name__ == '__main__':
    model_path = "./COCO/YOLOv8n/weights/best.pt"

    cls_names = {
        0: "Grass carp",
        1: "Black carp",
        2: "Sliver carp",
        3: "Bighead carp",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush"
    }

    ensure_storage_dir()
    ensure_detection_log_header()

    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    server_address = ("127.0.0.1", 8686)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    camera_ids = [0]
    camera_contexts = {}
    for camera_id in camera_ids:
        camera_contexts[camera_id] = create_camera_context(camera_id, model_path, cls_names)

    with ThreadPoolExecutor(max_workers=len(camera_ids)) as executor:
        while True:
            read_results = list(executor.map(read_one_camera, camera_contexts.items()))

            for camera_id, success, frame in read_results:
                if not success or frame is None:
                    continue
                process_one_camera_frame(
                    camera_ctx=camera_contexts[camera_id],
                    frame=frame,
                    box_annotator=box_annotator,
                    sock=sock,
                    server_address=server_address,
                    cls_names=cls_names
                )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    for camera_ctx in camera_contexts.values():
        camera_ctx['cap'].release()
        camera_ctx['videoWriter'].release()
        
    cv2.destroyAllWindows()
