#!/usr/bin/env python3
"""
Optimized Terrain-Aware Navigation with Road Model + Edge/Contour Obstacle Detection
- Uses trained road_model.pth for flat terrain free space
- Canny+contour detection for obstacles missed by model/YOLO
- Upstairs/downstairs classification
- Mobile IP camera input
"""

import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from scipy.signal import find_peaks
import sys
import os

# Import your trained model architecture
from model import LightweightUNet
from config import IMAGE_SIZE, MODEL_SAVE_PATH

# ============================================================
#  ENHANCED STAIR DETECTOR (with direction)
# ============================================================

class StairDetector:
    def __init__(self):
        self.consecutive_stair_frames = 0
        self.consecutive_flat_frames = 0
        self.stair_confidence = 0
        self.stair_direction = "unknown"  # "upstairs", "downstairs"
        self.direction_buffer = []

    def detect_stairs(self, depth_map):
        h, w = depth_map.shape
        roi_start = int(h * 0.3)
        roi_end = int(h * 0.8)
        depth_roi = depth_map[roi_start:roi_end, :]

        scores = []
        direction_scores = {"upstairs": 0, "downstairs": 0}

        # ----- Horizontal Edge Periodicity (same as before) -----
        sobel_y = cv2.Sobel(depth_roi, cv2.CV_64F, 0, 1, ksize=3)
        horizontal_edges = np.abs(sobel_y)
        edge_profile = np.mean(horizontal_edges, axis=1)

        try:
            peaks, _ = find_peaks(edge_profile, height=np.percentile(edge_profile, 60),
                                  distance=max(5, h//30))
            if len(peaks) >= 3:
                spacings = np.diff(peaks)
                if len(spacings) >= 2:
                    spacing_std = np.std(spacings)
                    spacing_mean = np.mean(spacings)
                    regularity = 1 - min(1.0, spacing_std / spacing_mean)
                    if regularity > 0.5:
                        scores.append(0.8 * regularity)
        except:
            pass

        # ----- Depth Step Pattern & Direction -----
        depth_profile = np.median(depth_roi, axis=1)
        depth_diff = np.diff(depth_profile)

        # Significant steps (absolute)
        step_thresh = np.std(depth_diff) * 1.2
        significant_steps = depth_diff[np.abs(depth_diff) > step_thresh]

        if len(significant_steps) >= 3:
            # Determine direction: positive diff = farther (upstairs), negative = closer (downstairs)
            pos_steps = significant_steps[significant_steps > 0]
            neg_steps = significant_steps[significant_steps < 0]
            
            if len(pos_steps) > len(neg_steps):
                direction_scores["upstairs"] += 0.7 * (len(pos_steps) / len(significant_steps))
            else:
                direction_scores["downstairs"] += 0.7 * (len(neg_steps) / len(significant_steps))
            
            step_consistency = 1 - min(1.0, np.std(significant_steps) / np.mean(np.abs(significant_steps)))
            scores.append(0.7 * step_consistency)

        # ----- Vertical Gradient -----
        vertical_gradient = np.mean(np.abs(sobel_y))
        if vertical_gradient > 0.15:
            gradient_score = min(1.0, vertical_gradient / 0.3)
            scores.append(gradient_score * 0.6)

        # ----- Horizontal Line Density -----
        edges = cv2.Canny((depth_map * 255).astype(np.uint8), 50, 150)
        horizontal_kernel = np.ones((1, 30), np.uint8)
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
        line_density = np.sum(horizontal_lines[roi_start:roi_end, :]) / (h * w)
        if line_density > 0.03:
            density_score = min(1.0, line_density / 0.1)
            scores.append(density_score * 0.5)

        # Final confidence
        confidence = sum(scores) / len(scores) if scores else 0
        is_stair = confidence > 0.55

        # Determine direction
        if is_stair:
            if direction_scores["upstairs"] > direction_scores["downstairs"]:
                direction = "upstairs"
            elif direction_scores["downstairs"] > direction_scores["upstairs"]:
                direction = "downstairs"
            else:
                direction = "unknown"
        else:
            direction = "unknown"

        # Temporal smoothing for direction
        if is_stair:
            self.consecutive_stair_frames += 1
            self.consecutive_flat_frames = 0
            self.stair_confidence = confidence
            self.direction_buffer.append(direction)
            if len(self.direction_buffer) > 5:
                self.direction_buffer.pop(0)
            # Majority vote
            if self.direction_buffer:
                self.stair_direction = max(set(self.direction_buffer), key=self.direction_buffer.count)
        else:
            self.consecutive_flat_frames += 1
            self.consecutive_stair_frames = 0
            self.stair_confidence = 0
            self.direction_buffer.clear()

        confirmed_stair = self.consecutive_stair_frames >= 3
        return confirmed_stair, confidence, self.stair_direction


# ============================================================
#  GROUND PLANE DETECTOR (for flat terrain)
# ============================================================

class GroundPlaneDetector:
    def __init__(self):
        self.ground_depth = None
        self.last_update = 0
        self.is_initialized = False

    def detect_ground_plane(self, depth_map, frame_num, road_mask=None, update_rate=5):
        h, w = depth_map.shape
        if frame_num - self.last_update >= update_rate:
            # Sample from bottom 30%, but exclude non-road areas if mask provided
            sample_region = depth_map[int(h * 0.7):, :]
            if road_mask is not None:
                road_region = road_mask[int(h * 0.7):, :]
                sample_region = sample_region[road_region > 0.5]

            valid_samples = sample_region[sample_region > 0.05]
            if len(valid_samples) > 100:
                self.ground_depth = np.percentile(valid_samples, 25)
                self.last_update = frame_num
                self.is_initialized = True
            elif not self.is_initialized:
                self.ground_depth = np.median(valid_samples) if len(valid_samples) > 0 else 0.4
                self.is_initialized = True

        if self.ground_depth is None:
            self.ground_depth = 0.4
        return self.ground_depth

    def compute_residual(self, depth_map, ground_depth):
        return ground_depth - depth_map


# ============================================================
#  EDGE/CONTOUR OBSTACLE DETECTOR
# ============================================================

class EdgeObstacleDetector:
    def __init__(self, canny_low=50, canny_high=150, min_contour_area=300):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area = min_contour_area

    def detect_obstacles(self, rgb_image, road_mask):
        """
        Find contours in regions NOT marked as road.
        Returns list of contours and overlay image.
        """
        # Convert to grayscale and apply Canny
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Dilate edges slightly to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours: outside road mask and large enough
        obstacle_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            # Check if contour is mostly outside road mask
            mask_roi = np.zeros_like(road_mask, dtype=np.uint8)
            cv2.drawContours(mask_roi, [cnt], -1, 1, thickness=cv2.FILLED)
            overlap = np.sum(mask_roi * (road_mask > 0.5))
            total = np.sum(mask_roi)
            if total > 0 and overlap / total < 0.3:  # less than 30% overlap with road
                obstacle_contours.append(cnt)

        return obstacle_contours, edges


# ============================================================
#  TERRAIN-AWARE NAVIGATION WITH ROAD MODEL
# ============================================================

class TerrainAwareNavigation:
    def __init__(self, road_model_path='road_model.pth', yolo_model='yolov8s.pt'):
        # Load YOLO
        self.yolo = YOLO(yolo_model)
        self.class_names = self.yolo.names

        # Load road segmentation model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.road_model = LightweightUNet().to(self.device)
        if os.path.exists(road_model_path):
            self.road_model.load_state_dict(torch.load(road_model_path, map_location=self.device))
            print(f"✅ Loaded road model from {road_model_path}")
        else:
            print(f"⚠️ Road model not found at {road_model_path}, using fallback")
            self.road_model = None
        self.road_model.eval()

        self.stair_detector = StairDetector()
        self.ground_plane = GroundPlaneDetector()
        self.edge_obstacle = EdgeObstacleDetector()

        self.terrain_type = "unknown"
        self.frame_num = 0
        self.obstacle_threshold = 0.12
        self.pothole_threshold = -0.06

    def get_road_mask(self, rgb_image):
        """Run road segmentation model to get binary mask (1=road)"""
        if self.road_model is None:
            return None
        h, w = rgb_image.shape[:2]
        # Preprocess
        rgb_resized = cv2.resize(rgb_image, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        normalized = rgb_resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.road_model(tensor)
            mask = output.cpu().numpy()[0, 0]
            mask = cv2.resize(mask, (w, h))
        return mask

    def process_frame(self, rgb_image, depth_map):
        self.frame_num += 1

        # Get road mask
        road_mask = self.get_road_mask(rgb_image)

        # Stair detection
        is_stair, stair_conf, stair_dir = self.stair_detector.detect_stairs(depth_map)

        if is_stair:
            self.terrain_type = "stairs"
            warnings, annotations = self._handle_stairs(rgb_image, depth_map, stair_conf, stair_dir)
        elif self.stair_detector.consecutive_flat_frames >= 5:
            self.terrain_type = "flat"
            warnings, annotations = self._handle_flat_terrain(rgb_image, depth_map, road_mask)
        else:
            self.terrain_type = "irregular"
            warnings, annotations = self._handle_irregular(rgb_image, depth_map, road_mask)

        # Overlay edge/contour obstacles on annotations
        if road_mask is not None:
            obstacle_contours, edges = self.edge_obstacle.detect_obstacles(rgb_image, road_mask)
            cv2.drawContours(annotations, obstacle_contours, -1, (255, 255, 0), 2)  # Cyan
            # Also overlay edges on depth map for debugging
            # (optional, can be shown in separate window)
        else:
            obstacle_contours = []

        return warnings, annotations, self.terrain_type, obstacle_contours

    def _handle_stairs(self, rgb_image, depth_map, confidence, direction):
        warnings = []
        annotations = rgb_image.copy()

        if confidence > 0.8:
            warn_text = f"STAIRS {direction.upper()}! Watch your step"
        elif confidence > 0.6:
            warn_text = f"{direction.capitalize()} detected, proceed carefully"
        else:
            warn_text = "Possible stairs ahead"
        warnings.append(warn_text)

        cv2.putText(annotations, f"TERRAIN: STAIRS ({direction})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotations, "WARNING: Stair detection active", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw horizontal bands
        h, w = depth_map.shape
        step_height = h // 8
        for i in range(4):
            y = int(h * 0.4) + i * step_height
            cv2.line(annotations, (0, y), (w, y), (0, 0, 255), 2)

        # YOLO
        yolo_results = self.yolo(rgb_image, verbose=False)[0]
        for box in yolo_results.boxes:
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotations, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotations, class_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

        return warnings, annotations

    def _handle_flat_terrain(self, rgb_image, depth_map, road_mask):
        warnings = []
        annotations = rgb_image.copy()

        # Ground plane detection using road mask
        ground_depth = self.ground_plane.detect_ground_plane(depth_map, self.frame_num, road_mask)
        residual = self.ground_plane.compute_residual(depth_map, ground_depth)

        # Obstacle mask: residual > threshold AND not road (if mask available)
        obstacle_mask_raw = residual > self.obstacle_threshold
        if road_mask is not None:
            obstacle_mask_raw = obstacle_mask_raw & (road_mask < 0.5)

        pothole_mask = residual < self.pothole_threshold
        if road_mask is not None:
            pothole_mask = pothole_mask & (road_mask < 0.5)

        # Clean masks
        kernel = np.ones((5,5), np.uint8)
        obstacle_mask = cv2.morphologyEx(obstacle_mask_raw.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        pothole_mask = cv2.morphologyEx(pothole_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        # Danger zone analysis
        h, w = depth_map.shape
        zone_width = w // 3
        def zone_danger(mask, col_start, col_end):
            zone = mask[:, col_start:col_end]
            return np.sum(zone) / (zone.shape[0]*zone.shape[1]) if zone.size>0 else 0

        left_danger = zone_danger(obstacle_mask, 0, zone_width)
        center_danger = zone_danger(obstacle_mask, zone_width, 2*zone_width)
        right_danger = zone_danger(obstacle_mask, 2*zone_width, w)

        if center_danger > 0.3:
            warnings.append("STOP - Obstacle ahead!")
        elif center_danger > 0.15:
            if left_danger < 0.1:
                warnings.append("Obstacle ahead, move LEFT")
            elif right_danger < 0.1:
                warnings.append("Obstacle ahead, move RIGHT")
            else:
                warnings.append("Obstacle ahead, proceed carefully")
        elif left_danger > 0.2:
            warnings.append("Obstacle on LEFT")
        elif right_danger > 0.2:
            warnings.append("Obstacle on RIGHT")

        pothole_contours, _ = cv2.findContours(pothole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pothole_count = sum(1 for c in pothole_contours if cv2.contourArea(c) > 500)
        if pothole_count > 0:
            warnings.append(f"Caution: {pothole_count} pothole(s) detected")

        # Draw overlays
        cv2.putText(annotations, "TERRAIN: FLAT", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.putText(annotations, f"Ground: {ground_depth:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

        # Draw obstacles (red) and potholes (blue)
        obstacle_contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in obstacle_contours:
            if cv2.contourArea(cnt) > 500:
                cv2.drawContours(annotations, [cnt], -1, (0,0,255), 2)
        for cnt in pothole_contours:
            if cv2.contourArea(cnt) > 500:
                cv2.drawContours(annotations, [cnt], -1, (255,0,0), 2)

        # Danger zone percentages
        cv2.putText(annotations, f"L:{left_danger:.0%}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        cv2.putText(annotations, f"C:{center_danger:.0%}", (w//2-30,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        cv2.putText(annotations, f"R:{right_danger:.0%}", (w-80,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

        # Road mask overlay (semi-transparent green)
        if road_mask is not None:
            green_overlay = np.zeros_like(rgb_image)
            green_overlay[:,:,1] = (road_mask * 255).astype(np.uint8)
            annotations = cv2.addWeighted(annotations, 0.7, green_overlay, 0.3, 0)

        # YOLO
        yolo_results = self.yolo(rgb_image, verbose=False)[0]
        for box in yolo_results.boxes:
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotations, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotations, class_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

        return warnings, annotations

    def _handle_irregular(self, rgb_image, depth_map, road_mask):
        warnings = ["Caution: Irregular terrain detected"]
        annotations = rgb_image.copy()
        cv2.putText(annotations, "TERRAIN: IRREGULAR", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255),2)
        cv2.putText(annotations, "Proceed with caution", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255),1)

        if road_mask is not None:
            green_overlay = np.zeros_like(rgb_image)
            green_overlay[:,:,1] = (road_mask * 255).astype(np.uint8)
            annotations = cv2.addWeighted(annotations, 0.7, green_overlay, 0.3, 0)

        yolo_results = self.yolo(rgb_image, verbose=False)[0]
        for box in yolo_results.boxes:
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotations, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotations, class_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

        return warnings, annotations


# ============================================================
#  DEPTH ESTIMATION (MiDaS)
# ============================================================

class MiDaSDepth:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading MiDaS on {self.device}...")
        self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
        self.model.eval().to(self.device)
        self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True).small_transform

    def compute_depth(self, rgb_image):
        img_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb)
        if img_tensor.dim() == 5:
            img_tensor = img_tensor.squeeze(0)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            pred = self.model(img_tensor)
            pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=img_rgb.shape[:2],
                                                   mode="bicubic", align_corners=False).squeeze()
        depth = pred.cpu().numpy()
        dmin, dmax = depth.min(), depth.max()
        if dmax - dmin > 1e-6:
            depth = (depth - dmin) / (dmax - dmin)
        depth = cv2.medianBlur((depth * 255).astype(np.uint8), 5) / 255.0
        return depth


# ============================================================
#  MAIN (IP INPUT DIRECT)
# ============================================================

def main():
    print("Initializing Terrain-Aware Navigation with Road Model...")
    depth_estimator = MiDaSDepth()
    navigation = TerrainAwareNavigation(road_model_path=MODEL_SAVE_PATH)  # uses config path

    # Get IP address directly
    print("\nEnter your phone's IP (from IP Webcam app):")
    ip = input("> ").strip()
    if not ip.startswith('http'):
        ip = f"http://{ip}"
    if ':' not in ip.split('://')[1]:
        ip = f"{ip}:8080"
    video_url = f"{ip}/video"
    print(f"Connecting to {video_url}...")
    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        print("Failed to connect. Trying USB camera as fallback...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No camera available.")
            return

    print("\n=== Optimized Navigation System ===")
    print("Features:")
    print("  - Road segmentation (trained model)")
    print("  - Stair detection with upstairs/downstairs")
    print("  - Edge/contour obstacles (cyan)")
    print("  - Depth-based obstacles (red), potholes (blue)")
    print("Controls: 'q' quit, 's' screenshot\n")

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
        depth = depth_estimator.compute_depth(frame)
        warnings, annotated, terrain_type, edge_contours = navigation.process_frame(frame, depth)

        # Voice-like warning display (console)
        now = time.time()
        if warnings and now - last_warning_time > 2.0:
            current_warnings = warnings
            last_warning_time = now
            for w in warnings:
                print(f"⚠️ {w}")

        for i, w in enumerate(current_warnings[:3]):
            cv2.putText(annotated, w, (10, 120 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Terrain color bar
        terrain_colors = {"flat": (0,255,0), "stairs": (0,0,255), "irregular": (0,165,255)}
        color = terrain_colors.get(terrain_type, (255,255,255))
        cv2.rectangle(annotated, (0,0), (200,30), color, -1)
        cv2.putText(annotated, f"TERRAIN: {terrain_type.upper()}", (5,22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # Show depth map with optional edge overlay
        depth_viz = (depth * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
        # Overlay cyan contours on depth map as well (optional)
        if edge_contours:
            cv2.drawContours(depth_colored, edge_contours, -1, (255,255,0), 1)

        cv2.imshow("Navigation View (Road=Green, Obstacles=Red/Cyan)", annotated)
        cv2.imshow("Depth Map (Cyan=Edge Obstacles)", depth_colored)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"nav_{ts}.jpg", annotated)
            cv2.imwrite(f"depth_{ts}.jpg", depth_colored)
            print(f"Screenshots saved: nav_{ts}.jpg, depth_{ts}.jpg")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()