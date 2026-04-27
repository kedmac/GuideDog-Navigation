# config.py
import os

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "..", "datasets")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

IMAGE_SIZE    = (160, 120)   # (width, height)
BATCH_SIZE    = 32
EPOCHS        = 15
LEARNING_RATE = 1e-3
TRAIN_SPLIT   = 0.9

MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES   = None

CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, "checkpoint.pth")
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")

# 6 classes — step_up / step_down / turn_around merged into caution_slow_down
# (too few samples: 9, 6, 0 — not learnable)
NAVIGATION_ACTIONS = {
    0: "move_forward",
    1: "move_left",
    2: "move_right",
    3: "stop",
    4: "caution_slow_down",
    5: "unknown",
}
NUM_CLASSES = len(NAVIGATION_ACTIONS)
