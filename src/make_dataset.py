
#!/usr/bin/env python
"""Download the full Zheng68k pbmc dataset from 10x Genomics 
and keep the top N highly-variable genes.
Steps:
1. Download tarball into raw_dir (default `data/raw/`)
2. Extract
3. Read the 10x matrix with scanpy
4. Copy the adata, `log1p` normalise to 10K counts, and
   run `highly_variable_genes()` on the copy
5. Subset the original raw matrix to those higly variable genes and save as .h5ad

Example:
    python src/make_dataset.py --out data/Zheng68K_5k.h5ad --n_top_genes 5000
"""

import argparse
import os
import sys
import tarfile
import glob
import requests
import scanpy as sc
import numpy as np
from pathlib import Path

URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
    "fresh_68k_pbmc_donor_a/"
    "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
)

def download(url: str, dest: Path, chunk_size: int = 1 << 20):
    """Download with progress bar if file not present."""
    if dest.exists():
        print(f"Using cached {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    done = 0
    with open(dest, "wb") as f:
        for block in r.iter_content(chunk_size):
            f.write(block)
            done += len(block)
            pct = 100 * done / total if total else 0
            bar = "█" * int(pct / 2)
            sys.stdout.write(f"\rDownloading {dest.name}: {pct:6.2f}% {bar:<50}")
            sys.stdout.flush()
    print("Done.")

def find_matrix_dir(root: Path) -> Path:
    """Return path that contains `matrix.mtx` after extraction."""
    patterns = [
        "**/filtered_feature_bc_matrix",
        "**/filtered_gene_bc_matrices",
        "**/filtered_matrices_mex",
    ]
    for pat in patterns:
        for cand in root.glob(pat):
            mtx = list(cand.rglob("matrix.mtx*"))
            if mtx:
                return mtx[0].parent
    raise FileNotFoundError("matrix.mtx not found in extracted archive")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_top_genes", type=int, default=5000)
    p.add_argument("--out", default="data/Zheng68K_5k.h5ad")
    p.add_argument("--raw_dir", default="data/raw")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    tar_path = raw_dir / "pbmc68k.tar.gz"
    extract_dir = raw_dir / "pbmc68k_extracted"

    # Download
    print("Checking dataset tarball...")
    download(URL, tar_path)

    # Extract
    if not extract_dir.exists():
        print("Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
    else:
        print("Archive already extracted")

    # Read matrix
    matrix_dir = find_matrix_dir(extract_dir)
    print(f"Reading 10x matrix from {matrix_dir}")
    adata = sc.read_10x_mtx(str(matrix_dir), var_names="gene_symbols", cache=False)
    print(f"Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # HVG selection on normalised log counts
    print(f"Selecting top {args.n_top_genes} HVGs (normalised & log1p)")
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp,
                                n_top_genes=args.n_top_genes,
                                flavor="seurat")
    hvgs = tmp.var.highly_variable
    adata = adata[:, hvgs]

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {adata.n_obs:,} cells x {adata.n_vars:,} genes to {out_path}")
    adata.write(str(out_path))

if __name__ == "__main__":
    main()
