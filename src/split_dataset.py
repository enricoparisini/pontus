#!/usr/bin/env python
"""Split a normalised .h5ad dataset into numpy-based train/val/test splits.
Example:
  python src/split_dataset.py --data data/Zheng68K_5k.h5ad --out_dir data/splits
"""
import argparse, os, pathlib, sys
from typing import Tuple

import anndata as ad
import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))
from data_handling import load_tensor_dataset, split_dataset 


def _subset_to_numpy(ds, subset, barcodes):
    """Return X and barcodes np arrays for a dataset subset."""
    idx = subset.indices
    # (cells, genes)
    X_np = ds.tensors[0][idx].detach().cpu().numpy()
    bc_np = barcodes[idx]
    return X_np, bc_np

def _save_npz(path, X, barcodes):
    # TODO: improve
    np.savez_compressed(path, X=X, barcodes=barcodes)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--train_frac', type=float, default=0.8)
    p.add_argument('--val_frac',   type=float, default=0.1)
    p.add_argument('--out_dir', default='data/splits')
    p.add_argument('--seed',       type=int,   default=42)
    args = p.parse_args()

    ds = load_tensor_dataset(args.data)
    adata = ad.read_h5ad(args.data)
    barcodes = np.asarray(adata.obs_names, dtype='U')  # keep as unicode str

    train_ds, val_ds, test_ds = split_dataset(ds, args.train_frac, args.val_frac, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    for split_name, subset in zip(('train', 'val', 'test'), (train_ds, val_ds, test_ds)):
        X_np, bc_np = _subset_to_numpy(ds, subset, barcodes)
        np_path = os.path.join(args.out_dir, f'{split_name}.npz')
        _save_npz(np_path, X_np, bc_np)
        print(f'Saved {split_name} split to {np_path}')

if __name__ == '__main__':
    main()

    
 