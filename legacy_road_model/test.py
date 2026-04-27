# test.py
import torch
import cv2
import numpy as np
from model import LightweightUNet
from config import IMAGE_SIZE, MODEL_SAVE_PATH

class RoadDetector:
    def __init__(self, model_path=MODEL_SAVE_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = LightweightUNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"✅ Model loaded on {self.device}")
    
    def detect_road(self, frame):
        """Detect road in frame. Returns mask where 1 = road, 0 = obstacle"""
        h, w = frame.shape[:2]
        
        # Preprocess
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(tensor)
            mask = output.cpu().numpy()[0, 0]
            mask = cv2.resize(mask, (w, h))
        
        return mask
    
    def visualize(self, frame, mask):
        """Create visualization: Green = Road, Red = Obstacle"""
        result = frame.copy()
        
        # Green overlay for road
        result[mask > 0.5] = [0, 255, 0]
        
        # Red overlay for obstacles (non-road)
        result[mask <= 0.5] = [0, 0, 255]
        
        # Blend with original
        result = cv2.addWeighted(frame, 0.6, result, 0.4, 0)
        
        # Add info
        road_percent = (mask > 0.5).sum() / mask.size
        cv2.putText(result, f"Road: {road_percent:.1%}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result, mask

def main():
    print("="*50)
    print("Road Detection - Test on Phone Camera")
    print("="*50)
    
    # Get phone IP
    phone_ip = input("\nEnter your phone's IP (from IP Webcam app): ").strip()
    video_url = f"http://{phone_ip}:8080/video"
    
    # Connect to camera
    print(f"\n📱 Connecting to {video_url}...")
    cap = cv2.VideoCapture(video_url)
    
    if not cap.isOpened():
        print("❌ Cannot connect. Trying USB camera...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ No camera available")
        return
    
    # Initialize detector
    detector = RoadDetector()
    
    print("\n✅ Connected!")
    print("\nWhat you'll see:")
    print("  🟢 GREEN = Road (safe to walk)")
    print("  🔴 RED = Obstacle (don't walk here)")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("="*50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reconnecting...")
            cap = cv2.VideoCapture(video_url)
            continue
        
        # Detect road
        mask = detector.detect_road(frame)
        
        # Visualize
        visualization, _ = detector.visualize(frame, mask)
        
        # Show
        cv2.imshow("Road Detection - GREEN=Road, RED=Obstacle", visualization)
        cv2.imshow("Mask (White=Road, Black=Obstacle)", (mask * 255).astype(np.uint8))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("road_detection.jpg", visualization)
            print("📸 Screenshot saved")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()