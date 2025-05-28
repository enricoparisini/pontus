#!/usr/bin/env python
"""
Finetune a Pontus model checkpoint on cell type classification task.

Usage:
    python src_celltype_annotation/fine_tune_celltype.py \
	--train_inputs    data/splits/val.npz \
	--val_inputs      data/splits/val.npz \
	--test_inputs     data/splits/test.npz \
	--train_labels    data/celltype_labels/y_val_int.npy \
	--val_labels    data/celltype_labels/y_val_int.npy \
	--test_labels    data/celltype_labels/y_test_int.npy \
	--ckpt_base     checkpoints/ckpt_name.pt \
	--batch_size 32 \
	--epochs 20 \
	--freeze_encoder 5 \
	--subset_frac 1.0
"""

import os
import sys
import json
from pathlib import Path
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import accuracy_score, f1_score

repo_root = Path(__file__).resolve().parent.parent / 'src'
sys.path.append(str(repo_root))
from model import Pontus



def load_inputs(path):
    arr = np.load(path)
    X = arr["X"]
    return torch.tensor(X, dtype=torch.float32)

def load_labels(path):
    y = np.load(path)
    return torch.from_numpy(y).long()


class Classifier(torch.nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder
        d = encoder.ln_f.normalized_shape[0]
        self.head = torch.nn.Linear(d, n_classes)

    def forward(self, x):
        h = self.encoder.embed_count(x) + self.encoder.embed_pos(x.shape[1])
        for blk in self.encoder.blocks: h = blk(h)
        h = self.encoder.ln_f(h).mean(1)
        return self.head(h)

def run_epoch(loader, model, crit, opt, device, amp, epoch):
    train = True if opt is not None else False
    tag = 'train' if train else 'val'
    model.train(train)
    tot, preds, targs = 0.0, [], []
    autocast = torch.amp.autocast if amp else torch.no_grad 
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    for X, y in tqdm(loader, desc=f"Epoch {epoch} [{tag}]"):
        X, y = X.to(device), y.to(device)
        with autocast('cuda', enabled=amp):
            out = model(X)
            loss = crit(out, y)
        if train:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        tot += loss.item()*len(X)
        preds.append(out.argmax(1).cpu()); targs.append(y.cpu())
    preds = torch.cat(preds); targs = torch.cat(targs)
    return (tot/len(loader.dataset), 
            accuracy_score(targs, preds), 
            f1_score(targs, preds, average='weighted') )



def main():
    p = argparse.ArgumentParser(description="Fine-tune cell type classifier.")
    p.add_argument("--train_inputs", required=True)
    p.add_argument("--val_inputs", required=True)
    p.add_argument("--test_inputs", required=True)
    p.add_argument('--train_labels', required=True)
    p.add_argument('--val_labels', required=True)
    p.add_argument('--test_labels', required=True)
    p.add_argument("--ckpt_base", required=True)
    p.add_argument('--ckpt_dir', default='checkpoints')

    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--freeze_encoder', type=int, default=1) # 5
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--xformers', action='store_true', default=False)
    p.add_argument('--device', default='cuda')

    p.add_argument('--subset_frac', type=float, default=1.0)
    p.add_argument('--subset_seed', type=int, default=42)

    args = p.parse_args()
    Path(args.ckpt_dir).mkdir(exist_ok=True, parents=True)

    X = load_inputs(args.train_inputs)
    y = load_labels(args.train_labels)
    train_ds = TensorDataset(X, y)
    n_classes = int(y.max()) + 1
    n_genes = X.shape[1]

    X = load_inputs(args.val_inputs)
    y = load_labels(args.val_labels)
    val_ds = TensorDataset(X, y)

    X = load_inputs(args.test_inputs)
    y = load_labels(args.test_labels)
    test_ds = TensorDataset(X, y)
    
    g = torch.Generator().manual_seed(args.subset_seed)
    if args.subset_frac < 1.0:
        idx = torch.randperm(len(train_ds), generator=g)[: int(len(train_ds)*args.subset_frac)]
        train_ds = torch.utils.data.Subset(train_ds, idx.tolist())
        idx = torch.randperm(len(val_ds), generator=g)[: int(len(val_ds)*args.subset_frac)]
        val_ds   = torch.utils.data.Subset(val_ds, idx.tolist())
        idx = torch.randperm(len(test_ds), generator=g)[: int(len(test_ds)*args.subset_frac)]
        test_ds  = torch.utils.data.Subset(test_ds, idx.tolist())

    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
    print("Dataloading for fine-tuning complete.")

    # Here we assume the pretrained model was initialised with the defaul prms:
    encoder = Pontus(n_genes=n_genes, xformers=args.xformers) 
    encoder.load_state_dict(torch.load(args.ckpt_base, map_location='cpu')['model_state_dict'])

    model = Classifier(encoder, n_classes).to(args.device)
    crit = torch.nn.CrossEntropyLoss()

    # Freeze encoder first
    for p in model.encoder.parameters(): 
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr)
    best = -1.0

    for ep in range(1, args.epochs+1):
        t_loss, t_acc, _ = run_epoch(loader = tr_loader, model = model, 
                                    crit = crit, opt = opt, 
                                    device = args.device, amp = args.amp, 
                                    epoch = ep)
        print(f'{ep:03d} | train acc {t_acc:.3f} | loss {t_loss:.3f}')
        if args.train_inputs != args.val_inputs:
            v_loss, v_acc, _ = run_epoch(val_loader, model, crit, None, args.device, args.amp, ep)
            metric = v_acc
            print(f'{ep:03d} | val acc {v_acc:.3f} | loss {v_loss:.3f}')
        else:
            metric = t_acc
        metric = t_acc 
        if metric > best:
            best = metric
            torch.save({'model_state_dict': model.state_dict(), 'acc': best},
                       Path(args.ckpt_dir)/'finetuned.pt')

        if ep == args.freeze_encoder:
            print(f'Unfreezing encoder at epoch {ep}, reducing LR by 10x')
            for p in model.encoder.parameters(): 
                p.requires_grad_(True)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr*0.1)

    # final test
    _, te_acc, te_f1 = run_epoch(test_loader, model, crit, None, args.device, args.amp, ep)
    print(f'Test acc {te_acc:.3f} | F1 {te_f1:.3f}')



if __name__ == "__main__":
    main()
