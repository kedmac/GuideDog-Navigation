# dataset.py
import io, os, re, torch, random
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from config import IMAGE_SIZE, NAVIGATION_ACTIONS, DATASET_DIR


def parse_label(text: str) -> str:
    if not text or pd.isna(text):
        return "unknown"

    raw = str(text)
    match = re.search(r"[Tt]o navigate safely[,.]?\s*(.+?)(?:\.|$)", raw, re.IGNORECASE | re.DOTALL)
    action_text = match.group(1).lower().strip() if match else raw.lower().strip()

    # ── turn around ────────────────────────────────────────────────────────────
    if any(k in action_text for k in ("turn around", "go back", "reverse", "180")):
        return "caution_slow_down"   # merged: too few samples to train a class

    # ── steps / stairs / curbs ─────────────────────────────────────────────────
    if any(k in action_text for k in (
        "step up", "step-up", "climb", "curb up", "stairs up",
        "step down", "step-down", "curb down", "stairs down", "descend",
    )):
        return "caution_slow_down"   # merged: too few samples to train a class

    # ── stop ───────────────────────────────────────────────────────────────────
    if any(k in action_text for k in ("stop", "halt", "do not move", "stand still", "wait")):
        return "stop"

    # ── caution ────────────────────────────────────────────────────────────────
    if any(k in action_text for k in ("caution", "slow down", "be careful", "careful")):
        return "caution_slow_down"

    # ── left ───────────────────────────────────────────────────────────────────
    if any(k in action_text for k in (
        "to the left", "toward the left", "towards the left",
        "slightly left", "veer left", "bear left",
        "move left", "turn left", "go left", "shift left",
        "lean left", "angle left", "head left", "navigate left",
        "toward 10 o'clock", "towards 10 o'clock",
        "toward 11 o'clock", "towards 11 o'clock",
        "head to 10", "head to 11", "move to 10", "move to 11",
    )):
        return "move_left"

    # ── right ──────────────────────────────────────────────────────────────────
    if any(k in action_text for k in (
        "to the right", "toward the right", "towards the right",
        "slightly right", "veer right", "bear right",
        "move right", "turn right", "go right", "shift right",
        "lean right", "angle right", "head right", "navigate right",
        "toward 1 o'clock",  "towards 1 o'clock",
        "toward 2 o'clock",  "towards 2 o'clock",
        "head to 1", "head to 2", "move to 1", "move to 2",
    )):
        return "move_right"

    # ── forward / straight ─────────────────────────────────────────────────────
    if any(k in action_text for k in (
        "proceed", "continue", "walk straight", "move straight",
        "go straight", "move forward", "walk forward", "go forward",
        "keep going", "straight ahead", "move ahead", "12 o'clock",
        "keep straight", "head straight", "head forward", "walk ahead",
        "maintain your course", "stay on your current path",
        "stay on the current path", "stay on the path",
        "keep moving", "keep walking", "move directly", "go directly",
        "navigate forward", "keep on",
    )):
        return "move_forward"

    return "unknown"


LABEL_TO_IDX = {v: k for k, v in NAVIGATION_ACTIONS.items()}


# ── Augmentation ──────────────────────────────────────────────────────────────
def augment(image: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    factor = random.uniform(0.85, 1.15)
    arr = np.clip(np.array(image, dtype=np.float32) * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ── Dataset ───────────────────────────────────────────────────────────────────
class GuideDogDataset(Dataset):

    SILVER_FILES = [f"silver-0000{i}-of-00009.parquet" for i in range(9)]

    def __init__(self, indices, df, training=False):
        self.indices  = indices
        self.df       = df
        self.training = training

    @classmethod
    def build_splits(cls, train_split=0.9, max_train=None, max_val=None, seed=42):
        print("Loading parquet files ...")
        dfs = []
        for fname in cls.SILVER_FILES:
            path = os.path.join(DATASET_DIR, fname)
            if not os.path.exists(path):
                print(f"  WARNING  Missing: {path}")
                continue
            shard = pd.read_parquet(path, columns=["image", "silver_label"])
            dfs.append(shard)
            print(f"  OK  {fname}: {len(shard):,} rows")

        if not dfs:
            raise FileNotFoundError(f"No parquet files found in {DATASET_DIR}")

        df = pd.concat(dfs, ignore_index=True)
        df = df[df["silver_label"].notna()].reset_index(drop=True)
        print(f"Total rows with label: {len(df):,}")

        df["label_idx"] = df["silver_label"].apply(
            lambda t: LABEL_TO_IDX[parse_label(t)]
        )

        dist = df["label_idx"].value_counts().sort_index()
        print("\nClass distribution:")
        for idx, count in dist.items():
            pct   = count / len(df) * 100
            bar   = "█" * int(pct / 2)
            print(f"  {NAVIGATION_ACTIONS[idx]:22s}: {count:5,}  ({pct:5.1f}%)  {bar}")

        n_unknown = (df["label_idx"] == LABEL_TO_IDX["unknown"]).sum()
        if n_unknown / len(df) > 0.05:
            print(f"\n  WARNING: {n_unknown:,} rows ({n_unknown/len(df)*100:.1f}%) are 'unknown'.")

        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n_train   = int(len(df) * train_split)
        train_idx = list(range(n_train))
        val_idx   = list(range(n_train, len(df)))

        if max_train: train_idx = train_idx[:max_train]
        if max_val:   val_idx   = val_idx[:max_val]

        print(f"\nTrain: {len(train_idx):,}   Val: {len(val_idx):,}")

        return (
            cls(train_idx, df, training=True),
            cls(val_idx,   df, training=False),
            df["label_idx"].values,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row      = self.df.iloc[self.indices[i]]
        img_data = row["image"]

        if isinstance(img_data, dict):
            raw   = img_data.get("bytes") or img_data.get("path")
            image = Image.open(io.BytesIO(raw)) if isinstance(raw, bytes) else Image.open(raw)
        elif isinstance(img_data, bytes):
            image = Image.open(io.BytesIO(img_data))
        else:
            image = img_data

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize(IMAGE_SIZE, Image.BILINEAR)

        if self.training:
            image = augment(image)

        arr    = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # C, H, W

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor, torch.tensor(int(row["label_idx"]), dtype=torch.long)
