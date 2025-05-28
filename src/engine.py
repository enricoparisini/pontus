
"""Training / evaluation loop helpers reused by train.py and eval.py."""
from contextlib import nullcontext

import torch
from tqdm import tqdm

def _sanitise(t):
    """Replace NaN/Inf with zeros so they don't propagate."""
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

def _to_tensor(batch, device):
    """Move batch to device and sanitise it."""
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    return _sanitise(x).to(device, non_blocking=True)


@torch.no_grad()
def eval_one_epoch(model, loader, device, amp):
    model.eval()
    running, n_elem = 0.0, 0
    ctx = (torch.autocast(device_type=str(device).split(':')[0], dtype=torch.bfloat16)
           if amp else nullcontext())
    with ctx:
        for xb in tqdm(loader, desc="[eval]", leave=False):
            xb = _to_tensor(xb, device)
            _, loss = model(xb, target=xb)
            if torch.isfinite(loss):
                running += loss.item() * xb.size(0)
                n_elem  += xb.size(0)
    return float('nan') if n_elem == 0 else running / n_elem


def train_one_epoch(model, loader, optimizer, scaler, device, amp, clip_grad, epoch):
    model.train()
    running, n_elem = 0.0, 0
    for xb, in tqdm(loader, desc=f"Epoch {epoch} [train]"):
        xb = _to_tensor(xb, device)
        optimizer.zero_grad(set_to_none=True)

        ctx = (torch.autocast(device_type=str(device).split(':')[0], dtype=torch.bfloat16)
               if amp else nullcontext())
        with ctx:
            _, loss = model(xb, target=xb)

        # backward pass
        if amp:
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        running += loss.item() * xb.size(0)
        n_elem  += xb.size(0)

    return float('nan') if n_elem == 0 else running / n_elem
