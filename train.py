# train.py
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    MODEL_SAVE_DIR, CHECKPOINT_PATH, BEST_MODEL_PATH,
    MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES, TRAIN_SPLIT,
    NAVIGATION_ACTIONS, NUM_CLASSES,
)
from dataset import GuideDogDataset
from model import NavigationModel


# ── Helpers ───────────────────────────────────────────────────────────────────

def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds.argmax(dim=1) == labels).float().mean().item()


def compute_class_weights(all_labels, num_classes):
    counts = np.bincount(all_labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = np.clip(weights, 0, weights.mean() * 3)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def save_checkpoint(epoch, model, optimizer, scheduler, best_val_acc, path):
    torch.save({
        "epoch":        epoch,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict(),
        "best_val_acc": best_val_acc,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    if not os.path.exists(path):
        return 0, 0.0
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"  ↩  Resumed from epoch {ckpt['epoch']+1}  (best val acc so far: {ckpt['best_val_acc']:.4f})")
    return ckpt["epoch"] + 1, ckpt["best_val_acc"]


# ── Train / eval loops ────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    desc = "Train" if training else "Val  "
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_acc  += accuracy(logits, labels)
            n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  GuideDog Navigation Model — Full Training")
    print("=" * 60)

    device = torch.device("cpu")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nLoading dataset …")
    train_ds, val_ds, all_labels = GuideDogDataset.build_splits(
        train_split=TRAIN_SPLIT,
        max_train=MAX_TRAIN_SAMPLES,
        max_val=MAX_VAL_SAMPLES,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model …")
    model = NavigationModel(num_classes=NUM_CLASSES, pretrained=True).to(device)
    print(f"   Backbone      : MobileNetV3-Small (pretrained)")
    print(f"   Parameters    : {model.param_count:,}")
    print(f"   Model size    : {model.size_mb:.1f} MB")

    # ── Loss with class weights ───────────────────────────────────────────────
    # Only compute weights from the TRAIN subset labels
    train_label_arr = np.array([
        int(train_ds.df.iloc[i]["label_idx"]) for i in train_ds.indices
    ])
    class_weights = compute_class_weights(train_label_arr, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    print(f"\n⚖️  Class weights: { {NAVIGATION_ACTIONS[i]: f'{w:.3f}' for i,w in enumerate(class_weights.tolist())} }")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # ── Resume if checkpoint exists ───────────────────────────────────────────
    start_epoch, best_val_acc = load_checkpoint(
        CHECKPOINT_PATH, model, optimizer, scheduler
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs  (resuming from epoch {start_epoch+1})\n")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}  {'Time':>6}")
    print("-" * 70)

    total_start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, training=False)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            flag = "best"

        # Per-epoch checkpoint (overwritten each epoch → only latest kept)
        save_checkpoint(epoch, model, optimizer, scheduler, best_val_acc, CHECKPOINT_PATH)

        print(
            f"{epoch+1:>6}  {train_loss:>10.4f}  {train_acc:>9.4f}  "
            f"{val_loss:>8.4f}  {val_acc:>7.4f}  {lr:>8.2e}  {elapsed:>5.0f}s{flag}"
        )

    total_time = time.time() - total_start
    print("-" * 70)
    print(f"\nDone! Total time: {total_time/60:.1f} min")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Best model save: {BEST_MODEL_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
