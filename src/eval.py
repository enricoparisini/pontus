
#!/usr/bin/env python
"""Evaluate a trained Pontus model on a test split.

Usage:
    python src/eval.py   \
	--ckpt  checkpoints/ckpt_name.pt  \
	--test_data data/splits/test.npz 
"""

import argparse
from engine import eval_one_epoch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_handling import load_dataset 
from model import Pontus

def _num_genes(ds):
    """Follow .dataset links until we get to the base TensorDataset"""
    while hasattr(ds, 'dataset'):
        ds = ds.dataset  
    if hasattr(ds, 'tensors'):
        return ds.tensors[0].shape[1]
    # fallback: look at the first sample
    sample = ds[0][0] if isinstance(ds[0], (tuple, list)) else ds[0]
    return sample.shape[-1]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True,
                        help='checkpoint .pt file to load')
    parser.add_argument('--test_data', type=str, required=True,
                        help='path to test .pt tensor dataset')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)

    test_ds = load_dataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, pin_memory=True)
    n_genes = _num_genes(test_ds)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model_kwargs = ckpt.get('model_kwargs', {})
    model_kwargs['n_genes'] = n_genes
    model = Pontus(**model_kwargs).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss = eval_one_epoch(model, test_loader, device, args.amp)
    print(f"Test loss: {test_loss:.4f}")

if __name__ == '__main__':
    main()