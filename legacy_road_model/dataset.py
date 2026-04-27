# dataset.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from config import CAMVID_PATH, ROAD_RGB, IMAGE_SIZE

class CamVidRoadDataset(Dataset):
    """Load CamVid dataset and create road masks"""
    
    def __init__(self, split='train'):
        """
        split: 'train', 'val', or 'test'
        """
        self.split = split
        self.image_size = IMAGE_SIZE
        
        # Set paths based on split
        if split == 'train':
            self.image_dir = os.path.join(CAMVID_PATH, 'train')
            self.label_dir = os.path.join(CAMVID_PATH, 'train_labels')
        elif split == 'val':
            self.image_dir = os.path.join(CAMVID_PATH, 'val')
            self.label_dir = os.path.join(CAMVID_PATH, 'val_labels')
        else:  # test
            self.image_dir = os.path.join(CAMVID_PATH, 'test')
            self.label_dir = os.path.join(CAMVID_PATH, 'test_labels')
        
        # Get all image files
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        
        print(f"Loaded {len(self.images)} {split} images from CamVid")
        print(f"Looking for labels with '_L' suffix")
    
    def __len__(self):
        return len(self.images)
    
    def _create_road_mask(self, label_image):
        """
        Convert color label image to binary road mask
        White (1) = Road, Black (0) = Everything else
        """
        # Road is RGB (128, 64, 128) in CamVid labels
        road_mask = np.all(label_image == ROAD_RGB, axis=2)
        return road_mask.astype(np.float32)
    
    def __getitem__(self, idx):
        # Get image name
        img_name = self.images[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load label - add _L before .png
        # Example: 0001TP_009210.png -> 0001TP_009210_L.png
        base_name = img_name.replace('.png', '')
        label_name = base_name + '_L.png'
        label_path = os.path.join(self.label_dir, label_name)
        
        label = cv2.imread(label_path)
        
        if label is None:
            raise FileNotFoundError(f"Could not read label: {label_path}")
        
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        
        # Create binary road mask
        mask = self._create_road_mask(label)
        
        # Resize
        image = cv2.resize(image, (self.image_size[0], self.image_size[1]))
        mask = cv2.resize(mask, (self.image_size[0], self.image_size[1]), interpolation=cv2.INTER_NEAREST)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        
        return image, mask