# config.py
# ============================================
# CONFIGURATION - CHANGE THESE PATHS
# ============================================

# Path to your CamVid dataset
CAMVID_PATH = r"C:\Users\T RAJESH\OneDrive\Desktop\CV Project\Dataset\archive\CamVid"  # <-- CHANGE THIS TO YOUR PATH

# Road color in CamVid labels (from your class_dict.csv)
ROAD_RGB = (128, 64, 128)  # Road is class 19 with this color

# Training settings
IMAGE_SIZE = (320, 240)    # Width, Height (smaller = faster training)
BATCH_SIZE = 8             # How many images per batch
EPOCHS = 30                # Training iterations
LEARNING_RATE = 0.0001     # How fast model learns

# Model settings
MODEL_SAVE_PATH = "road_model.pth"  # Where to save trained model
