
import torch, anndata as ad
import numpy as np
import pathlib
from torch.utils.data import TensorDataset, random_split

def load_tensor_dataset(path):
    adata = ad.read_h5ad(path)
    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else X
    X = torch.tensor(X, dtype=torch.float32)

    libsize = X.sum(1, keepdims=True)
    libsize[libsize == 0] = 1            # avoid divide byzero
    X = X / libsize
    X = torch.log1p(X * 1e4 + 1)

    X[torch.isinf(X)] = 0                # replace inf with 0
    X = (X / X.max()).unsqueeze(-1)      # (cells, genes, 1)
    return TensorDataset(X)

def split_dataset(ds, train_frac=0.8, val_frac=0.1, seed=42):
    n = len(ds)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    return random_split(ds, [n_train, n_val, n_test],
                        generator=torch.Generator().manual_seed(seed))


def load_dataset(path):
    """Load a saved .pt TensorDataset **or** a .h5ad file."""
    if path.endswith('.pt'):
        return torch.load(path)
    elif path.endswith('.npz'):
        arr = np.load(path)
        X = torch.from_numpy(arr['X']).float()
        ds = TensorDataset(X)
        ds.barcodes = arr['barcodes']
        return ds
    elif path.endswith('.h5ad'):
        return load_tensor_dataset(path)
    else:
        raise ValueError(f'Cannot infer dataset type from extension: {path}')
