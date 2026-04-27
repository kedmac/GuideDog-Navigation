#!/usr/bin/env python3
"""
Final Optimized Terrain-Aware Navigation
- YOLO + simple tracking (IoU matching, age-based)
- Forbidden mask for edge suppression inside tracked objects
- Road mask from trained model (used for ground plane & obstacle restriction)
- Stair detection with upstairs/downstairs
- Frame skipping, FP16, video saving
"""

import cv2
import torch
import numpy as np
import time
import os
from ultralytics import YOLO
from scipy.signal import find_peaks
from model import LightweightUNet
from config import IMAGE_SIZE, MODEL_SAVE_PATH

# ============================================================
#  TRACKING HELPERS
# ============================================================

class TrackedObject:
    def __init__(self, box, class_id, frame_id):
        self.box = box  # (x1, y1, x2, y2)
        self.class_id = class_id
        self.last_seen = frame_id
        self.age = 0  # frames since last detection

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def update_tracks(tracks, detections, frame_id, iou_thresh=0.3, max_age=2):
    used = [False] * len(detections)
    for track in tracks:
        best_iou = 0
        best_idx = -1
        for i, det in enumerate(detections):
            if used[i]:
                continue
            iou_val = iou(track.box, det['box'])
            if iou_val > best_iou and iou_val > iou_thresh:
                best_iou = iou_val
                best_idx = i
        if best_idx >= 0:
            track.box = detections[best_idx]['box']
            track.last_seen = frame_id
            track.age = 0
            used[best_idx] = True
    # Add new detections
    for i, det in enumerate(detections):
        if not used[i]:
            tracks.append(TrackedObject(det['box'], det['class_id'], frame_id))
    # Remove stale tracks
    tracks = [t for t in tracks if frame_id - t.last_seen <= max_age]
    return tracks

# ============================================================
#  ENHANCED STAIR DETECTOR (with direction)
# ============================================================

class StairDetector:
    def __init__(self):
        self.consecutive_stair_frames = 0
        self.consecutive_flat_frames = 0
        self.stair_direction = "unknown"
        self.direction_buffer = []

    def detect_stairs(self, depth_map):
        h, w = depth_map.shape
        roi_start = int(h * 0.3)
        roi_end = int(h * 0.8)
        depth_roi = depth_map[roi_start:roi_end, :]

        scores = []
        direction_scores = {"upstairs": 0, "downstairs": 0}

        # Horizontal edge periodicity
        sobel_y = cv2.Sobel(depth_roi, cv2.CV_64F, 0, 1, ksize=3)
        edge_profile = np.mean(np.abs(sobel_y), axis=1)
        try:
            peaks, _ = find_peaks(edge_profile, height=np.percentile(edge_profile, 60),
                                  distance=max(5, h//30))
            if len(peaks) >= 3:
                spacings = np.diff(peaks)
                if len(spacings) >= 2:
                    regularity = 1 - min(1.0, np.std(spacings)/np.mean(spacings))
                    if regularity > 0.5:
                        scores.append(0.8 * regularity)
        except:
            pass

        # Depth step pattern & direction
        depth_profile = np.median(depth_roi, axis=1)
        depth_diff = np.diff(depth_profile)
        step_thresh = np.std(depth_diff) * 1.2
        significant = depth_diff[np.abs(depth_diff) > step_thresh]
        if len(significant) >= 3:
            pos = np.sum(significant > 0)
            neg = np.sum(significant < 0)
            if pos > neg:
                direction_scores["upstairs"] += 0.7 * (pos/len(significant))
            else:
                direction_scores["downstairs"] += 0.7 * (neg/len(significant))
            scores.append(0.7 * (1 - min(1.0, np.std(significant)/np.mean(np.abs(significant)))))

        # Vertical gradient
        vert_grad = np.mean(np.abs(sobel_y))
        if vert_grad > 0.15:
            scores.append(min(1.0, vert_grad/0.3) * 0.6)

        # Horizontal line density
        edges = cv2.Canny((depth_map*255).astype(np.uint8), 50, 150)
        horiz_kernel = np.ones((1,30), np.uint8)
        horiz_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horiz_kernel)
        line_density = np.sum(horiz_lines[roi_start:roi_end,:]) / (h*w)
        if line_density > 0.03:
            scores.append(min(1.0, line_density/0.1)*0.5)

        confidence = sum(scores)/len(scores) if scores else 0
        is_stair = confidence > 0.55

        if is_stair:
            self.consecutive_stair_frames += 1
            self.consecutive_flat_frames = 0
            self.direction_buffer.append("upstairs" if direction_scores["upstairs"] > direction_scores["downstairs"] else "downstairs")
            if len(self.direction_buffer) > 5:
                self.direction_buffer.pop(0)
            if self.direction_buffer:
                self.stair_direction = max(set(self.direction_buffer), key=self.direction_buffer.count)
        else:
            self.consecutive_flat_frames += 1
            self.consecutive_stair_frames = 0
            self.direction_buffer.clear()

        confirmed = self.consecutive_stair_frames >= 3
        return confirmed, confidence, self.stair_direction

# ============================================================
#  GROUND PLANE DETECTOR (using road mask)
# ============================================================

class GroundPlaneDetector:
    def __init__(self):
        self.ground_depth = None
        self.last_update = 0

    def detect_ground_plane(self, depth_map, road_mask, frame_num, update_rate=5):
        h, w = depth_map.shape
        if frame_num - self.last_update >= update_rate and road_mask is not None:
            # Bottom 30% of image
            bottom = depth_map[int(h*0.7):, :]
            mask_region = road_mask[int(h*0.7):, :]
            valid = bottom[mask_region > 0.3]  # soft threshold
            if len(valid) > 100:
                self.ground_depth = np.percentile(valid, 25)
                self.last_update = frame_num
            elif self.ground_depth is None:
                self.ground_depth = np.median(valid) if len(valid) > 0 else 0.4
        if self.ground_depth is None:
            self.ground_depth = 0.4
        return self.ground_depth

    def compute_residual(self, depth_map, ground_depth):
        return ground_depth - depth_map

# ============================================================
#  EDGE/CONTOUR OBSTACLE DETECTOR (with forbidden mask)
# ============================================================

class EdgeObstacleDetector:
    def __init__(self, canny_low=50, canny_high=150, min_area=300):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area = min_area

    def detect(self, rgb_image, road_mask, forbidden_mask, scale_factor=0.5):
        h, w = rgb_image.shape[:2]
        small_h, small_w = int(h*scale_factor), int(w*scale_factor)
        small_rgb = cv2.resize(rgb_image, (small_w, small_h))
        small_road = cv2.resize(road_mask, (small_w, small_h)) if road_mask is not None else None
        small_forbidden = cv2.resize(forbidden_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST) if forbidden_mask is not None else None

        gray = cv2.cvtColor(small_rgb, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 1.5)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Suppress edges outside road and inside forbidden regions
        if small_road is not None:
            edges[small_road < 0.3] = 0
        if small_forbidden is not None:
            edges[small_forbidden > 0] = 0

        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacle_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            # Scale back to original size
            cnt_scaled = (cnt * (1.0/scale_factor)).astype(np.int32)
            obstacle_contours.append(cnt_scaled)
        return obstacle_contours

# ============================================================
#  MAIN NAVIGATION SYSTEM WITH TRACKING
# ============================================================

class OptimizedNavigation:
    def __init__(self, road_model_path=MODEL_SAVE_PATH, yolo_model='yolov8n.pt',
                 skip_yolo=3, skip_midas=2, use_fp16=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'

        # YOLO
        self.yolo = YOLO(yolo_model)
        if self.use_fp16:
            self.yolo.model.half()

        # Road model
        self.road_model = LightweightUNet().to(self.device)
        if os.path.exists(road_model_path):
            self.road_model.load_state_dict(torch.load(road_model_path, map_location=self.device))
            print(f"Loaded road model from {road_model_path}")
        else:
            print(f"Warning: road model not found at {road_model_path}")
            self.road_model = None
        self.road_model.eval()
        if self.use_fp16 and self.road_model:
            self.road_model.half()

        self.stair_detector = StairDetector()
        self.ground_plane = GroundPlaneDetector()
        self.edge_obstacle = EdgeObstacleDetector(min_area=200)

        self.skip_yolo = skip_yolo
        self.skip_midas = skip_midas
        self.frame_counter = 0

        # Tracking
        self.tracks = []  # list of TrackedObject

        # Cached results
        self.last_depth = None
        self.last_road_mask = None
        self.last_midas_frame = -1
        self.last_yolo_frame = -1

    def get_road_mask(self, rgb_image):
        if self.road_model is None:
            return None
        h, w = rgb_image.shape[:2]
        resized = cv2.resize(rgb_image, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).permute(2,0,1).unsqueeze(0).to(self.device)
        if self.use_fp16:
            tensor = tensor.half()
        with torch.no_grad():
            out = self.road_model(tensor)
            mask = out.cpu().numpy()[0,0]
            mask = cv2.resize(mask, (w, h))
        return mask

    def get_depth(self, rgb_image, midas_model, transform):
        # Crop bottom 60% ROI, resize to 256x256 for MiDaS small
        h, w = rgb_image.shape[:2]
        roi_h = int(h * 0.6)
        cropped = rgb_image[-roi_h:, :]
        small = cv2.resize(cropped, (256, 256))
        img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb)
        if tensor.dim() == 5:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        if self.use_fp16:
            tensor = tensor.half()
        with torch.no_grad():
            pred = midas_model(tensor)
            pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=small.shape[:2],
                                                   mode='bicubic', align_corners=False).squeeze()
        depth = pred.cpu().numpy().astype(np.float32)
        dmin, dmax = depth.min(), depth.max()
        if dmax - dmin > 1e-6:
            depth = (depth - dmin) / (dmax - dmin)
        depth = cv2.medianBlur((depth*255).astype(np.uint8), 5) / 255.0
        # Resize to full frame (only bottom part used)
        full_depth = np.zeros((h, w), dtype=np.float32)
        full_depth[-roi_h:, :] = cv2.resize(depth, (w, roi_h))
        return full_depth

    def process_frame(self, rgb_image, midas_model, transform):
        self.frame_counter += 1
        frame_id = self.frame_counter

        # 1. Depth estimation (skip_midas)
        if frame_id - self.last_midas_frame >= self.skip_midas:
            depth = self.get_depth(rgb_image, midas_model, transform)
            self.last_depth = depth
            self.last_midas_frame = frame_id
        else:
            depth = self.last_depth

        # 2. YOLO + road model (skip_yolo)
        if frame_id - self.last_yolo_frame >= self.skip_yolo:
            # YOLO detection
            yolo_results = self.yolo(rgb_image, verbose=False)[0]
            detections = []
            for box in yolo_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                detections.append({'box': (x1, y1, x2, y2), 'class_id': class_id})
            # Update tracks
            self.tracks = update_tracks(self.tracks, detections, frame_id, max_age=2)

            # Road mask
            road_mask = self.get_road_mask(rgb_image)
            self.last_road_mask = road_mask
            self.last_yolo_frame = frame_id
        else:
            road_mask = self.last_road_mask
            # Still propagate tracks: increase age, but keep boxes
            for t in self.tracks:
                t.age += 1
            # Remove stale (older than 2 frames without detection)
            self.tracks = [t for t in self.tracks if frame_id - t.last_seen <= 2]

        # 3. Build forbidden mask from all tracked boxes
        h, w = rgb_image.shape[:2]
        forbidden_mask = np.zeros((h, w), dtype=np.uint8)
        for t in self.tracks:
            x1, y1, x2, y2 = t.box
            # Dilate a little to suppress edges near the boundary
            x1 = max(0, x1 - 5)
            y1 = max(0, y1 - 5)
            x2 = min(w, x2 + 5)
            y2 = min(h, y2 + 5)
            cv2.rectangle(forbidden_mask, (x1, y1), (x2, y2), 1, -1)

        # 4. Stair detection
        is_stair, stair_conf, stair_dir = self.stair_detector.detect_stairs(depth)

        # 5. Terrain-specific handling
        if is_stair:
            terrain = "stairs"
            warnings, annotated = self._handle_stairs(rgb_image, depth, stair_conf, stair_dir, self.tracks)
        elif self.stair_detector.consecutive_flat_frames >= 5:
            terrain = "flat"
            warnings, annotated = self._handle_flat(rgb_image, depth, road_mask, forbidden_mask, self.tracks)
        else:
            terrain = "irregular"
            warnings, annotated = self._handle_irregular(rgb_image, road_mask, self.tracks)

        # 6. Edge obstacle detection (only if road_mask exists)
        if road_mask is not None:
            edge_contours = self.edge_obstacle.detect(rgb_image, road_mask, forbidden_mask, scale_factor=0.5)
            cv2.drawContours(annotated, edge_contours, -1, (255, 255, 0), 2)  # cyan

        return warnings, annotated, terrain, depth

    def _handle_stairs(self, rgb, depth, conf, direction, tracks):
        warnings = [f"{direction.upper()} stairs ahead" if conf>0.6 else "Possible stairs"]
        ann = rgb.copy()
        cv2.putText(ann, f"STAIRS ({direction})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        h,w = depth.shape
        step_h = h//8
        for i in range(4):
            y = int(h*0.4) + i*step_h
            cv2.line(ann, (0,y), (w,y), (0,0,255),2)
        # Draw tracked boxes
        for t in tracks:
            x1,y1,x2,y2 = t.box
            cv2.rectangle(ann, (x1,y1), (x2,y2), (0,255,0), 2)
            # Optionally add class name (we don't store name, but can map)
        return warnings, ann

    def _handle_flat(self, rgb, depth, road_mask, forbidden_mask, tracks):
        warnings = []
        ann = rgb.copy()

        if road_mask is None:
            # Fallback: treat whole image as road
            road_mask = np.ones(depth.shape, dtype=np.float32)

        # Ground plane from road region
        ground = self.ground_plane.detect_ground_plane(depth, road_mask, self.frame_counter)
        residual = self.ground_plane.compute_residual(depth, ground)

        # Obstacle mask: positive residual AND road region AND not forbidden (but obstacles can be inside forbidden if they are tracked objects)
        obstacle_raw = (residual > 0.12) & (road_mask > 0.3)
        # We do NOT mask out forbidden here because obstacles inside YOLO boxes are already represented by the box itself.
        # Instead, we will use the obstacle mask to find new obstacles not covered by tracking.
        kernel = np.ones((5,5), np.uint8)
        obstacle_mask = cv2.morphologyEx(obstacle_raw.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        # Find road horizontal extent
        road_cols = np.any(road_mask > 0.3, axis=0)
        if np.any(road_cols):
            left_road = np.where(road_cols)[0][0]
            right_road = np.where(road_cols)[0][-1]
        else:
            left_road, right_road = 0, rgb.shape[1]

        zone_width = (right_road - left_road) // 3
        if zone_width < 1:
            zone_width = 1
        left_zone = (left_road, left_road + zone_width)
        center_zone = (left_road + zone_width, left_road + 2*zone_width)
        right_zone = (left_road + 2*zone_width, right_road)

        def zone_danger(mask, x_start, x_end):
            if x_end <= x_start:
                return 0
            zone = mask[:, x_start:x_end]
            return np.sum(zone) / (zone.shape[0] * (x_end - x_start)) if zone.size > 0 else 0

        left_danger = zone_danger(obstacle_mask, left_zone[0], left_zone[1])
        center_danger = zone_danger(obstacle_mask, center_zone[0], center_zone[1])
        right_danger = zone_danger(obstacle_mask, right_zone[0], right_zone[1])

        # Generate warnings
        if center_danger > 0.25:
            warnings.append("STOP - obstacle ahead!")
        elif center_danger > 0.12:
            if left_danger < 0.08:
                warnings.append("Move LEFT")
            elif right_danger < 0.08:
                warnings.append("Move RIGHT")
            else:
                warnings.append("Obstacle ahead")
        elif left_danger > 0.2:
            warnings.append("Obstacle on LEFT")
        elif right_danger > 0.2:
            warnings.append("Obstacle on RIGHT")

        # Draw obstacles (red)
        obst_contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in obst_contours:
            if cv2.contourArea(cnt) > 200:
                cv2.drawContours(ann, [cnt], -1, (0,0,255), 2)

        # Draw tracked boxes (green)
        for t in tracks:
            x1,y1,x2,y2 = t.box
            cv2.rectangle(ann, (x1,y1), (x2,y2), (0,255,0), 2)

        # Overlay road mask (semi-transparent green)
        green_overlay = np.zeros_like(rgb)
        green_overlay[:,:,1] = (road_mask * 255).astype(np.uint8)
        ann = cv2.addWeighted(ann, 0.7, green_overlay, 0.3, 0)

        # Text info
        cv2.putText(ann, "FLAT", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.putText(ann, f"L:{left_danger:.0%} C:{center_danger:.0%} R:{right_danger:.0%}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        return warnings, ann

    def _handle_irregular(self, rgb, road_mask, tracks):
        warnings = ["Caution: irregular terrain"]
        ann = rgb.copy()
        cv2.putText(ann, "IRREGULAR", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255),2)
        if road_mask is not None:
            green_overlay = np.zeros_like(rgb)
            green_overlay[:,:,1] = (road_mask * 255).astype(np.uint8)
            ann = cv2.addWeighted(ann, 0.7, green_overlay, 0.3, 0)
        for t in tracks:
            x1,y1,x2,y2 = t.box
            cv2.rectangle(ann, (x1,y1), (x2,y2), (0,255,0), 2)
        return warnings, ann

# ============================================================
#  MIDAS DEPTH LOADER
# ============================================================

def load_midas():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading MiDaS on {device}")
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
    model.eval().to(device)
    if device.type == 'cuda':
        model = model.half()
    transform = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True).small_transform
    return model, transform

# ============================================================
#  MAIN
# ============================================================

def main():
    out_dir = "output_videos"
    os.makedirs(out_dir, exist_ok=True)

    midas_model, transform = load_midas()
    nav = OptimizedNavigation(skip_yolo=3, skip_midas=2, use_fp16=True)

    # Get phone IP
    print("\nEnter your phone's IP (from IP Webcam app):")
    ip = input("> ").strip()
    if not ip.startswith('http'):
        ip = f"http://{ip}"
    if ':' not in ip.split('://')[1]:
        ip = f"{ip}:8080"
    video_url = f"{ip}/video"
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print("Failed to connect, trying USB camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No camera available.")
            return

    # Get frame size for video writer
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        out_w = w * 2
        out_h = h
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"navigation_{timestamp}.mp4")
        out_video = cv2.VideoWriter(out_path, fourcc, 15, (out_w, out_h))
        print(f"Saving video to {out_path}")

    print("\n=== Optimized Navigation (Final) ===")
    print("Features: YOLO tracking, forbidden mask, road-mask restricted")
    print("Controls: q quit, s screenshot\n")

    frame_count = 0
    last_warning_time = 0
    current_warnings = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(video_url)
            time.sleep(1)
            continue

        frame_count += 1
        start = time.time()

        warnings, annotated, terrain, depth = nav.process_frame(frame, midas_model, transform)

        # Depth visualization
        depth_viz = (depth * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

        # Combine side-by-side
        combined = np.hstack((annotated, depth_colored))

        # Display warnings (console and on screen)
        now = time.time()
        if warnings and now - last_warning_time > 2.0:
            current_warnings = warnings
            last_warning_time = now
            for w in warnings:
                print(f"⚠️ {w}")
        for i, w in enumerate(current_warnings[:2]):
            cv2.putText(combined, w, (10, annotated.shape[0] - 40 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Terrain indicator
        color = (0,255,0) if terrain=="flat" else ((0,0,255) if terrain=="stairs" else (0,165,255))
        cv2.rectangle(combined, (0,0), (200,30), color, -1)
        cv2.putText(combined, f"TERRAIN: {terrain.upper()}", (5,22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # FPS
        fps = 1/(time.time()-start+1e-6)
        cv2.putText(combined, f"FPS: {fps:.1f}", (combined.shape[1]-100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        out_video.write(combined)
        cv2.imshow("Navigation (Left) + Depth (Right)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshot_{ts}.jpg", combined)
            print("Screenshot saved")

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {out_path}")

if __name__ == '__main__':
    main()