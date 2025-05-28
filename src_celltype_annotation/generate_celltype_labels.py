#!/usr/bin/env python
"""
Generate label vectors for each split using celltypist output.

Usage:
    python src_celltype_annotation/generate_celltype_labels.py \
        --celltypist_csv data/celltype_labels/pbmc68k_celltypist.csv \
        --train_split    data/splits/train.npz \
        --val_split      data/splits/val.npz \
        --test_split     data/splits/test.npz \
        --output_dir     data/celltype_labels
"""

import argparse
import numpy as np
import pandas as pd
import os
import json

def encode_labels(y_list):
    """Map class names to integer indices. 
    Return (y_int, mapping dict, inverse dict)."""
    classes = sorted(set(y_list))
    str2int = {cls: i for i, cls in enumerate(classes)}
    int2str = {i: cls for i, cls in enumerate(classes)}
    y_int = np.array([str2int[y] for y in y_list], dtype=np.int64)
    return y_int, str2int, int2str

def encode_with_unknown(y, str2int, unknown_value=-1):
    """Map label y to int using str2int, or unknown_value if not present."""
    return np.array([str2int[y_] if y_ in str2int else unknown_value for y_ in y], dtype=np.int64)


def main():
    p = argparse.ArgumentParser(description="Generate split label vectors from celltypist csv.")
    p.add_argument("--celltypist_csv", required=True, help="celltypist predictions (csv with barcodes, cell_type)")
    p.add_argument("--train_split", required=True, help="Path to train split npz")
    p.add_argument("--val_split", required=True, help="Path to val split npz")
    p.add_argument("--test_split", required=True, help="Path to test split npz")
    p.add_argument("--output_dir", required=True, help="Output directory for y_train.npy etc")
    args = p.parse_args()

    # Load celltypist predictions
    df = pd.read_csv(args.celltypist_csv, index_col=0)
    if "cell_type" not in df.columns:
        raise ValueError("celltypist csv must have a 'cell_type' column.")
    barcode2label = df["cell_type"].to_dict()

    # For each split assign labels by barcode order
    os.makedirs(args.output_dir, exist_ok=True)
    for split_name, split_path in [("train", args.train_split), ("val", args.val_split), ("test", args.test_split)]:
        arr = np.load(split_path)
        barcodes = arr["barcodes"]
        y = np.array([barcode2label[bc] for bc in barcodes])
        out_path = os.path.join(args.output_dir, f"y_{split_name}.npy")
        np.save(out_path, y)
        print(f"Saved {out_path}  (shape={y.shape})")

        if split_name == "train":
            # Encode labels to integers based on training set
            y_int, str2int, int2str = encode_labels(y)
            with open(os.path.join(args.output_dir, "label_mapping.json"), "w") as f:
                json.dump({"str2int": str2int, "int2str": int2str}, f, indent=2)
            print(f"Label mapping saved to {args.output_dir}/label_mapping.json")
            np.save(os.path.join(args.output_dir, f"y_{split_name}_int.npy"), y_int)
        else:
            y_int = encode_with_unknown(y, str2int)
            if (y_int == -1).any():
                missing = set(y[y_int == -1])
                print(f"WARNING: {len(missing)} {split_name} labels were never seen in training: {missing}")
            np.save(os.path.join(args.output_dir, f"y_{split_name}_int.npy"), y_int)

    print("All label files saved and aligned to splits.")

if __name__ == "__main__":
    main()
