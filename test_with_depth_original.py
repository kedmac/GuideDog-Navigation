import os
import csv
import time
import datetime
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import NavigationModel
from config import IMAGE_SIZE, NAVIGATION_ACTIONS, NUM_CLASSES, BEST_MODEL_PATH

# --- CONFIGURATION FOR VIDEO FILE ---
VIDEO_INPUT_PATH = "crowd_video1.mp4"  # Change this to your file name

# ── ImageNet normalization ────────────────────────────────────────────────
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# ── Depth zone boundaries ─────────────────────────────────────────────────
ZONE_L = (0.00, 0.33)
ZONE_C = (0.33, 0.67)
ZONE_R = (0.67, 1.00)

ACTION_COLORS = {
    "move_forward":      (0,   200,   0),
    "move_left":         (0,   200, 255),
    "move_right":        (0,   165, 255),
    "stop":              (0,     0, 220),
    "caution_slow_down": (0,   140, 255),
    "unknown":           (120, 120, 120),
}

class DepthEstimator:
    def __init__(self):
        print("Loading MiDaS depth model …")
        self.device = torch.device("cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
        print("MiDaS ready")

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)
        with torch.no_grad():
            raw = self.model(input_batch)
            raw = F.interpolate(raw.unsqueeze(1), size=frame_bgr.shape[:2], mode="bicubic", align_corners=False).squeeze()
        depth = raw.cpu().numpy().astype(np.float32)
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6: depth = (depth - d_min) / (d_max - d_min)
        return depth

    def colorize(self, depth: np.ndarray) -> np.ndarray:
        uint8 = (depth * 255).astype(np.uint8)
        return cv2.applyColorMap(uint8, cv2.COLORMAP_MAGMA)

class Navigator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = NavigationModel(num_classes=NUM_CLASSES, pretrained=False)
        self.model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location="cpu"))
        self.model.eval()
        print(f"Navigation model loaded")

    def predict(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize(IMAGE_SIZE, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        t = (t - MEAN) / STD
        t = t.unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(t), dim=1).squeeze()
        conf, idx = probs.max(dim=0)
        action = NAVIGATION_ACTIONS[idx.item()]
        all_probs = {NAVIGATION_ACTIONS[i]: round(probs[i].item(), 3) for i in range(NUM_CLASSES)}
        return action, conf.item(), all_probs

def zone_stats(depth: np.ndarray):
    w = depth.shape[1]
    l1, l2 = int(w * ZONE_L[0]), int(w * ZONE_L[1])
    c1, c2 = int(w * ZONE_C[0]), int(w * ZONE_C[1])
    r1, r2 = int(w * ZONE_R[0]), int(w * ZONE_R[1])
    stats = {
        "depth_left_mean": float(depth[:, l1:l2].mean()),
        "depth_center_mean": float(depth[:, c1:c2].mean()),
        "depth_right_mean": float(depth[:, r1:r2].mean()),
        "depth_overall_max": float(depth.max()),
    }
    flat_idx = int(depth.argmax())
    stats["closest_col"], stats["closest_row"] = flat_idx % w, flat_idx // w
    return stats

def depth_interpretation(stats: dict) -> str:
    zones = {"LEFT": stats["depth_left_mean"], "CENTER": stats["depth_center_mean"], "RIGHT": stats["depth_right_mean"]}
    closest_zone = max(zones, key=zones.get)
    val = zones[closest_zone]
    severity = "DANGER" if val > 0.70 else "CAUTION" if val > 0.45 else "CLEAR"
    return f"{severity} - {closest_zone} zone"

def build_prob_bar(all_probs: dict, w: int = 320, h: int = 160) -> np.ndarray:
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 30
    n = len(all_probs)
    bar_h = max(1, (h - 20) // n)
    for i, (cls, prob) in enumerate(all_probs.items()):
        color = ACTION_COLORS.get(cls, (150, 150, 150))
        bar_w = int(prob * (w - 110))
        y = 10 + i * bar_h
        cv2.rectangle(canvas, (100, y), (100 + bar_w, y + bar_h - 3), color, -1)
        cv2.putText(canvas, cls[:14], (2, y + bar_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)
        cv2.putText(canvas, f"{prob:.3f}", (100 + bar_w + 3, y + bar_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
    return canvas

def build_display(rgb_frame, depth_colored, all_probs, action, conf, depth_stats, frame_id, fps):
    H, W = rgb_frame.shape[:2]
    target_h = 360
    scale = target_h / H
    tw = int(W * scale)

    # 1. Prepare the base panels
    rgb_resized = cv2.resize(rgb_frame, (tw, target_h))
    depth_resized = cv2.resize(depth_colored, (tw, target_h))
    
    # 2. Overlay Zone Boundaries and Labels on RGB
    # We iterate through the tuples (start, end) and their string labels
    for (zone_range, label) in [(ZONE_L, "LEFT"), (ZONE_C, "CENTER"), (ZONE_R, "RIGHT")]:
        start_frac, end_frac = zone_range
        x1 = int(tw * start_frac)
        x2 = int(tw * end_frac)
        
        # Draw vertical separator lines (gray)
        cv2.rectangle(rgb_resized, (x1, 0), (x2, target_h), (80, 80, 80), 1)
        
        # Add zone text label at the top
        cv2.putText(rgb_resized, label, (x1 + 10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Draw Crosshair at the closest point (obstacle)
    # Rescale coordinates from original frame to resized display
    sx, sy = tw / W, target_h / H
    cx = int(depth_stats["closest_col"] * sx)
    cy = int(depth_stats["closest_row"] * sy)
    cv2.drawMarker(rgb_resized, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    # 4. Create the probability bar chart panel
    prob_bar = build_prob_bar(all_probs, w=tw, h=target_h)

    # 5. Horizontal Stack: [ RGB | Depth | Probabilities ]
    panel = np.hstack([rgb_resized, depth_resized, prob_bar])

    # 6. Create Bottom Info Bar
    info_h = 48
    info_bar = np.zeros((info_h, panel.shape[1], 3), dtype=np.uint8)
    color = ACTION_COLORS.get(action, (150, 150, 150))
    interp = depth_interpretation(depth_stats)

    status_text = (f"Frame {frame_id:05d} | FPS {fps:.1f} | "
                   f"ACTION: {action.upper()} ({conf:.2f}) | "
                   f"DEPTH: {interp}")

    cv2.putText(info_bar, status_text, (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # 7. Vertical Stack: [ Panels ] over [ Info Bar ]
    return np.vstack([panel, info_bar])

class SessionRecorder:
    def __init__(self, base_dir="recordings"):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(base_dir, f"session_{ts}")
        os.makedirs(self.dir, exist_ok=True)
        self.csv_path = os.path.join(self.dir, "session_log.csv")
        self._csv_f = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._csv_f, fieldnames=[
            "frame_id", "timestamp", "action", "confidence", "depth_left_mean", 
            "depth_center_mean", "depth_right_mean", "depth_overall_max", "closest_col", "closest_row"
        ])
        self._writer.writeheader()
        self.video_writer = None

    def init_video_writer(self, frame_size):
        path = os.path.join(self.dir, "processed_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(path, fourcc, 20.0, frame_size)
        print(f"Video recording to: {path}")

    def record(self, frame_id, display_frame, stats, action, conf):
        if self.video_writer is not None:
            self.video_writer.write(display_frame)
        
        self._writer.writerow({
            "frame_id": frame_id, "timestamp": time.time(), "action": action, "confidence": conf,
            "depth_left_mean": stats["depth_left_mean"], "depth_center_mean": stats["depth_center_mean"],
            "depth_right_mean": stats["depth_right_mean"], "depth_overall_max": stats["depth_overall_max"],
            "closest_col": stats["closest_col"], "closest_row": stats["closest_row"]
        })

    def close(self):
        if self.video_writer: self.video_writer.release()
        self._csv_f.close()

def main():
    print("\n--- GuideDog Navigator Initialization ---")
    print("1: Load Video File")
    print("2: IP Webcam Stream")
    choice = input("Select input source (1 or 2): ").strip()

    if choice == "1":
        cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    else:
        ip = input("Enter phone IP (e.g. http://192.168.1.5:8080): ").strip()
        cap = cv2.VideoCapture(f"{ip}/video")

    if not cap.isOpened():
        print("Error: Could not open source.")
        return

    depth_model = DepthEstimator()
    nav_model = Navigator()
    recorder = SessionRecorder()
    
    frame_id = 0
    fps_timer = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break

        depth_raw = depth_model.predict(frame)
        depth_colored = depth_model.colorize(depth_raw)
        action, conf, all_probs = nav_model.predict(frame)
        stats = zone_stats(depth_raw)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_timer, 1e-6))
        fps_timer = now

        display = build_display(frame, depth_colored, all_probs, action, conf, stats, frame_id, fps)
        
        # Initialize video writer once we know the final display size
        if recorder.video_writer is None:
            h, w = display.shape[:2]
            recorder.init_video_writer((w, h))

        cv2.imshow("GuideDog + MiDaS", display)
        recorder.record(frame_id, display, stats, action, conf)

        if (cv2.waitKey(1) & 0xFF) == ord("q"): break
        frame_id += 1

    cap.release()
    recorder.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
