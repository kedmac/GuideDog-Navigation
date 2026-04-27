# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from model import LightweightUNet
from dataset import CamVidRoadDataset
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, CAMVID_PATH

class DiceLoss(nn.Module):
    """Dice loss for better segmentation"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union"""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou

def train():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if dataset exists
    if not os.path.exists(CAMVID_PATH):
        print(f"\n❌ ERROR: CamVid dataset not found at '{CAMVID_PATH}'")
        print("Please update the path in config.py")
        return
    
    # Load datasets
    print("\n📂 Loading CamVid dataset...")
    train_dataset = CamVidRoadDataset(split='train')
    val_dataset = CamVidRoadDataset(split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Create model
    print("\n🏗️ Creating model...")
    model = LightweightUNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss functions and optimizer
    bce_loss = nn.BCELoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_iou = 0
    
    print("\n🚀 Starting training...")
    print("="*50)
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
        
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_iou = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]')
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                # Calculate loss
                loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
                val_loss += loss.item()
                
                # Calculate IoU
                iou = calculate_iou(outputs, masks)
                val_iou += iou.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val IoU: {avg_val_iou:.4f}")
        
        # Save best model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✅ Saved best model (IoU: {best_iou:.4f})")
        
        print("-"*50)
    
    print(f"\n🎉 Training complete!")
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Model saved as: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()
