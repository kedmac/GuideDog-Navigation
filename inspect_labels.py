# inspect_labels.py
import os
import pandas as pd
from dataset import parse_label
from config import DATASET_DIR, NAVIGATION_ACTIONS

def main():
    dfs = []
    for i in range(9):
        path = os.path.join(DATASET_DIR, f"silver-0000{i}-of-00009.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path, columns=["silver_label"])
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df[df["silver_label"].notna()]

    print(f"Total labelled rows: {len(df):,}\n")

    # Class distribution
    df["parsed"] = df["silver_label"].apply(parse_label)
    dist = df["parsed"].value_counts()
    print("Class distribution after parsing:")
    for cls, count in dist.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {cls:22s}: {count:5,}  ({pct:5.1f}%)  {bar}")

    # Show 5 raw examples for each 'unknown' to help improve the parser
    unknowns = df[df["parsed"] == "unknown"]["silver_label"]
    if len(unknowns) > 0:
        print(f"\n{len(unknowns):,} rows mapped to 'unknown'. Sample raw labels:")
        for t in unknowns.sample(min(20, len(unknowns)), random_state=42):
            print(f"  → {str(t)[:120]}")
        print("\nIf you see clear patterns above, add them to parse_label() in dataset.py")

if __name__ == "__main__":
    main()
