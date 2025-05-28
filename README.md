
# PONTUS

**P**ermutation-invariant **O**rder-agnostic **N**oise-aware **T**ransformer for **U**nstructured **S**ingle-cell data

---

PONTUS is a foundation model prototype  for single-cell transcriptomic data that handles the continuity and permutation invariance of gene expression data natively.

1. **Permutation-invariant attention**. Self-attention is masked with a randomized lower-triangular pattern each batch, ensuring the model treats every gene token equally, with no fixed left-to-right bias, and encouraging generalisation across datasets.

2. **Continuous auto-discretization and regression**  
   - Raw gene expression values  are noisy and inherently continuous (especially when normalized). Rather than using hard bins, a learned quantisation scheme is applied, softly binning the data and allowing the model to learn how to denoise expression values directly.  
   - Instead of predicting categories over gene counts, we use a reconstruction loss on the embedding space, aligning predicted and input gene token embeddings. This enhances denoising and improves embedding learning for downstream tasks.


The model is optimized for large-scale training via
- Mixed precision training (AMP)
- Checkpointing and `torch.compile`
- 8-bit quantized optimizers
- [xFormers](https://facebookresearch.github.io/xformers/) optimised attention layers

Note that most of the existing single-cell foundation models:
- Either use hard bins (scBERT and scGPT) or drop expression information entirely (e.g. by ranking in Geneformer).
- Impose a fixed gene order and either an autoregressive masking (scGPT) or bidirectional full attention (scBERT and
Geneformer).


## Installation

```bash
git clone https://github.com/enricoparisini/pontus.git
cd pontus
pip install -r requirements.txt
```
A Docker image that mirrors the latest version of this repository is available:
```bash
$ docker pull eparisini/pontus
```


## Test on cell type annotation

As an example, you can download and split the [Zheng68K](https://www.nature.com/articles/ncomms14049) PBMC dataset into train, val and test sets keeping only the 5K most variable genes, 
```
python src/make_dataset.py --out data/Zheng68K_5k.h5ad --n_top_genes 5000

python src/split_dataset.py --data data/Zheng68K_5k.h5ad --out_dir data/splits
```
then pretrain the model on the training set:
```
python src/train.py \
  --train_data data/splits/train.npz \
  --val_data   data/splits/val.npz \
  --test_data  data/splits/test.npz \
  --epochs 50

python src/eval.py   \
  --ckpt  checkpoints/{ckpt_pretrained}.pt  \
	--test_data data/splits/test.npz 
```

We evaluate the performance of the pretrained model in cell annotation by comparing its predictions to CellTypist ground-truth labels, by adding a linear head to PONTUS and fine-tuning it over the val set.

Generate the "ground-truth" cell type labels with CellTypist:
```
python src_celltype_annotation/run_celltypist.py \
    --input data/Zheng68K_5k.h5ad \
    --model Immune_All_Low.pkl \
    --output data/celltype_labels/pbmc68k_celltypist.csv \
	  --n_jobs -1

python src_celltype_annotation/generate_celltype_labels.py \
    --celltypist_csv data/celltype_labels/pbmc68k_celltypist.csv \
    --train_split    data/splits/train.npz \
    --val_split      data/splits/val.npz \
    --test_split     data/splits/test.npz \
    --output_dir     data/celltype_labels
```

Fine-tune the pretrained model on the val set, and test the accuracy in cell annotation on the test set: 
```
python src_celltype_annotation/fine_tune_celltype.py \
	--train_inputs    data/splits/val.npz \
	--val_inputs      data/splits/val.npz \
	--test_inputs     data/splits/test.npz \
	--train_labels    data/celltype_labels/y_val_int.npy \
	--val_labels      data/celltype_labels/y_val_int.npy \
	--test_labels     data/celltype_labels/y_test_int.npy \
	--ckpt_base       checkpoints/{ckpt_pretrained}.pt \
	--batch_size 32 \
	--epochs 10 \
	--freeze_encoder 5 
```




## Results

This is the resulting performance for cell annotation over 5-fold cross-validation compared to scBERT and scGPT: 

| Model     | Parameters |       Accuracy      |           F1    |
|:-----------:|:------------:|:---------------------:|:-----------------:|
| PONTUS    | 4.4 M      | 0.691 ± 0.014       | 0.664 ± 0.016   |
| scBERT    | 8.9 M      | 0.703 ± 0.011       | 0.670 ± 0.008   |
| scGPT     | 53 M       | 0.741 ± 0.026       | 0.728 ± 0.023   |

Despite being significantly smaller and pretrained on a much smaller corpus, PONTUS achieves performance comparable to scBERT and shows promise for further improvement upon large-scale pretraining.

The following ablation study cofirms the important role played by the strategies adopted in PONTUS:

| Model      |       Accuracy        |           F1      |
|:----------:|:---------------------:|:-----------------:|
| PONTUS  with hard bins and causal attn   | 0.639 ± 0.008         | 0.624 ± 0.009     |
| PONTUS with hard bins    | 0.663 ± 0.006         | 0.635 ± 0.005     |
| PONTUS with causal attn     | 0.658 ± 0.012         | 0.641 ± 0.010     |
| **PONTUS**     | **0.691 ± 0.014**         | **0.664 ± 0.016**     |








## Large-scale training 

Useful tags for large-scale training when running `src/train.py`:
- `--amp` for mixed precision and faster training.
- `--torch_compile` to enable `torch.compile()` for graph-level optimizations.
- `--xformers` to enable `xformers` optimised attention blocks.

When running `src_celltype_annotation/fine_tune_celltype.py`, tune `--freeze_encoder` and `--lr` schedules for optimal finetuning.

In both scripts, for debug purposes use `--subset_frac` to train or finetune the model on a fraction (between 0. and 1.) of the training set. 



