
#!/usr/bin/env python
"""Run celltypist on an adata file and export barcode to cell_type csv.
Automatically log normalises counts to 1e4 if raw.
Works with celltypist >= 1.5.

Usage:
python src_celltype_annotation/run_celltypist.py \
    --input data/Zheng68K_5k.h5ad \
    --model Immune_All_Low.pkl \
    --output data/celltype_labels/pbmc68k_celltypist.csv \
	--n_jobs -1
"""
import argparse
import os
import inspect
import scanpy as sc
import pandas as pd
import numpy as np
import celltypist

def ensure_normalised(adata):
    if adata.X.max() > 100:       
        print('Normalising counts to 1e4 and log1p ...')
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print(f' Input looks normalised (max={adata.X.max():.1f})')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_jobs', type=int, default=-1, help='Number of cpu cores for celltypist (if supported)')
    p.add_argument('--input', required=True)
    p.add_argument('--model', default='Immune_All_Low.pkl')
    p.add_argument('--output', default='celltypist_labels.csv')
    args = p.parse_args()

    adata = sc.read_h5ad(args.input)
    ensure_normalised(adata)

    # Build kwargs dynamically in case celltypist version does not support n_jobs
    annotate_kwargs = dict(model=args.model, majority_voting=True)
    if args.n_jobs is not None:
        if 'n_jobs' in inspect.signature(celltypist.annotate).parameters:
            annotate_kwargs['n_jobs'] = args.n_jobs
    res = celltypist.annotate(adata, **annotate_kwargs)

    labels = res.predicted_labels
    if not isinstance(labels, pd.Series):
        labels = labels.iloc[:, 0]

    df = pd.DataFrame({'barcode': labels.index, 'cell_type': labels.values})
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f'Saved to {args.output}')

if __name__ == '__main__':
    main()
