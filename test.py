# test.py
import torch
import cv2
import numpy as np
from PIL import Image
from model import NavigationModel
from config import IMAGE_SIZE, NAVIGATION_ACTIONS, NUM_CLASSES, BEST_MODEL_PATH


class Navigator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model  = NavigationModel(num_classes=NUM_CLASSES, pretrained=False)
        self.model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location="cpu"))
        self.model.eval()
        print(f"Model loaded from {BEST_MODEL_PATH}")

    def predict(self, frame_bgr):
        rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb).resize(IMAGE_SIZE, Image.BILINEAR)
        arr   = np.array(img, dtype=np.float32) / 255.0
        t     = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(t), dim=1)
        conf, idx = probs.max(dim=1)
        return NAVIGATION_ACTIONS[idx.item()], conf.item()


def main():
    nav = Navigator()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        ip = input("Camera not found. Enter phone IP (or press Enter to skip): ").strip()
        if ip:
            cap = cv2.VideoCapture(f"http://{ip}:8080/video")
    if not cap.isOpened():
        print("No camera available.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        action, conf = nav.predict(frame)
        colour = (0, 200, 0) if conf > 0.7 else (0, 165, 255)
        cv2.putText(frame, f"{action}  ({conf:.2f})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
        cv2.imshow("GuideDog Navigation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
