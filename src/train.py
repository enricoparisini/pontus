#!/usr/bin/env python
"""Training script for Pontus models.

Usage:

python src/train.py \
  --train_data data/splits/train.npz \
  --val_data   data/splits/val.npz \
  --test_data  data/splits/test.npz \
  --epochs 50
"""

from engine import train_one_epoch, eval_one_epoch

import argparse
import os
import sys
import math
import torch
import mlflow
import bitsandbytes as bnb
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))
from model import Pontus
from data_handling import load_tensor_dataset, split_dataset, load_dataset


def _gene_dim(ds):
    if hasattr(ds, 'tensors'):
        return ds.tensors[0].shape[1]
    elif hasattr(ds, 'dataset'):
        return _gene_dim(ds.dataset)
    else:
        raise AttributeError("Could not locate tensors in dataset")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',)
    p.add_argument('--train_data')
    p.add_argument('--val_data') 
    p.add_argument('--test_data')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--device', default='cuda')
    p.add_argument('--mlflow_uri', default='mlruns')
    p.add_argument('--subset_frac', type=float, default=1.0)
    p.add_argument('--subset_seed', type=int, default=42)
    p.add_argument('--ckpt_dir', default='checkpoints', help='folder to save checkpoints')
    
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--torch_compile', action='store_true', default=False)
    p.add_argument('--xformers', action='store_true', default=False)
    p.add_argument('--clip_grad', type=float, default=1.0, help='0 = disable')
    return p.parse_args()


def get_datasets(args):
    if args.train_data and args.val_data:
        train_ds = load_dataset(args.train_data)
        val_ds   = load_dataset(args.val_data)
        test_ds  = load_dataset(args.test_data) if args.test_data else None
        n_genes  = _gene_dim(train_ds)
    else:
        if args.data is None:
            raise SystemExit("Provide either --train_data/--val_data or --data")
        full = load_tensor_dataset(args.data)
        train_ds, val_ds, test_ds = split_dataset(full)
        n_genes = _gene_dim(full)
    return train_ds, val_ds, test_ds, n_genes


def _build_ckpt_tag(args):
    """Return a compact string with hyperparameters for file names."""
    fields = ['lr', 'batch_size', 'subset_frac', 'emb_dim', 'n_head', 'dropout', "xformers"]
    parts = []
    for f in fields:
        if hasattr(args, f):
            val = getattr(args, f)
            if isinstance(val, float):
                val = f"{val:g}"
            parts.append(f"{f}={val}")
    return "_".join(parts)


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = _build_ckpt_tag(args)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Data loading:
    train_ds, val_ds, test_ds, n_genes = get_datasets(args)
    g = torch.Generator().manual_seed(args.subset_seed)
    if args.subset_frac < 1.0:
        idx = torch.randperm(len(train_ds), generator=g)[: int(len(train_ds)*args.subset_frac)]
        train_ds = torch.utils.data.Subset(train_ds, idx.tolist())
        idx = torch.randperm(len(val_ds), generator=g)[: int(len(val_ds)*args.subset_frac)]
        val_ds   = torch.utils.data.Subset(val_ds, idx.tolist())
        if test_ds:
            idx = torch.randperm(len(test_ds), generator=g)[: int(len(test_ds)*args.subset_frac)]
            test_ds  = torch.utils.data.Subset(test_ds, idx.tolist())

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size) if test_ds else None

    # Model and optimiser:
    device = torch.device(args.device)
    model = Pontus(n_genes=n_genes, xformers = args.xformers).to(device)
    if args.torch_compile:
        model = torch.compile(model, mode="max-autotune")
    optim = bnb.optim.AdamW8bit(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # Training:
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment('Pontus')
    with mlflow.start_run():
        mlflow.log_params({**vars(args), 'n_genes': n_genes})

        for ep in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_dl, optim, scaler,
                                         device, args.amp, args.clip_grad, ep)
            mlflow.log_metric('train_loss', train_loss, step=ep)

            val_loss = eval_one_epoch(model, val_dl, device, args.amp)
            mlflow.log_metric('val_loss', val_loss, step=ep)

            mlflow.pytorch.log_state_dict(model.state_dict(),
                                          artifact_path=f'state_dicts/epoch{ep}')

            if ep % 10 == 0 or ep == args.epochs:
                ckpt_path = ckpt_dir / f"{tag}_epoch{ep}.pt"
                torch.save({
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if args.amp else None,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
            
        # Test:
        if test_dl:
            test_loss = eval_one_epoch(model, test_dl, device, args.amp)
            mlflow.log_metric('test_loss', test_loss)
            print(f'Test MSE: {test_loss:.5f}')
            

if __name__ == '__main__':
    main()