# test.py
import torch
import cv2
import numpy as np
from PIL import Image
from model import NavigationModel
from config import IMAGE_SIZE, NAVIGATION_ACTIONS, NUM_CLASSES, BEST_MODEL_PATH

# ImageNet normalization — must match what was used during training
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class Navigator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model  = NavigationModel(num_classes=NUM_CLASSES, pretrained=False)
        self.model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location="cpu"))
        self.model.eval()
        print(f"Model loaded from {BEST_MODEL_PATH}")

    def preprocess(self, frame_bgr):
        """Convert a BGR OpenCV frame to a normalised model input tensor."""
        rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb).resize(IMAGE_SIZE, Image.BILINEAR)
        arr   = np.array(img, dtype=np.float32) / 255.0
        t     = torch.from_numpy(arr).permute(2, 0, 1)   # C, H, W
        t     = (t - MEAN) / STD                         # ImageNet normalise
        return t.unsqueeze(0)                             # add batch dim

    def predict(self, frame_bgr):
        """
        Returns (action_name, confidence).
        action_name is one of the NAVIGATION_ACTIONS values.
        confidence is a float 0-1.
        """
        tensor = self.preprocess(frame_bgr)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)
        conf, idx = probs.max(dim=1)
        return NAVIGATION_ACTIONS[idx.item()], conf.item()

    def predict_all(self, frame_bgr):
        """Returns a dict of {action: probability} for all classes — useful for debugging."""
        tensor = self.preprocess(frame_bgr)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1).squeeze()
        return {NAVIGATION_ACTIONS[i]: round(probs[i].item(), 3) for i in range(NUM_CLASSES)}


def main():
    nav = Navigator()

    # Try webcam first, fall back to IP camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        ip = input("Webcam not found. Enter phone IP (e.g. 192.168.1.5): ").strip()
        cap = cv2.VideoCapture(f"http://{ip}:8080/video")
    if not cap.isOpened():
        print("No camera available.")
        return

    print("Press 'q' to quit  |  Press 'd' to toggle debug (show all class probs)")
    debug = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        action, conf = nav.predict(frame)

        # Colour: green if confident, orange if not sure
        colour = (0, 200, 0) if conf > 0.6 else (0, 165, 255)
        cv2.putText(frame, f"{action}  ({conf:.2f})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)

        # Debug overlay — show all class probabilities
        if debug:
            all_probs = nav.predict_all(frame)
            y = 90
            for act, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
                cv2.putText(frame, f"{act}: {prob:.3f}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y += 22

        cv2.imshow("GuideDog Navigation", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            debug = not debug

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
