import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp 
from xformers.ops import memory_efficient_attention

#################################################################################
###  Embeddings: 
#################################################################################

class SoftBinnedGeneEmbedding(nn.Module):
    """Soft bin embedding for continuous gene-expression counts."""
    def __init__(self, n_bins, emb_dim):
        super().__init__()
        self.weights = nn.Sequential(
            nn.Linear(1, n_bins),
            nn.ReLU(),
            nn.Linear(n_bins, n_bins),
            nn.Softmax(dim=-1),
        )
        self.bin_emb = nn.Embedding(n_bins, emb_dim)

    def forward(self, x: torch.Tensor):          # (B,T,1)
        w = self.weights(x)                      # (B,T,n_bins)
        return w @ self.bin_emb.weight           # (B,T,d)


class GenePositionalEmbedding(nn.Module):
    def __init__(self, n_genes, emb_dim):
        super().__init__()
        self.table = nn.Embedding(n_genes, emb_dim)

    def forward(self, T):                   
        return self.table(torch.arange(T, device=self.table.weight.device)) # (T,d)


################################################################################# 
### Randomised multi-head attention: 
#################################################################################

class RandomCausalMHA(nn.Module):
    """Multi-head selfattention with a different random mask each forward pass.
     Input: (B,T,C) tensors where C = n_head * head_dim."""
    def __init__(self, emb_dim, n_head, dropout):
        super().__init__()
        assert emb_dim % n_head == 0, "emb_dim must divide n_head"
        self.n_head   = n_head
        self.head_dim = emb_dim // n_head

        self.qkv  = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.drop = nn.Dropout(dropout)

    def _mask(self, T, device, dtype):
        """Additive attention mask. We start with a causal mask, then apply
        a random permutation to the keys so each query attends to a
        different random causal prefix."""
        tril  = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        perm  = torch.randperm(T, device=device)
        masked = ~tril[:, perm]                  # True -> hide
        return torch.where(
            masked,
            torch.tensor(float('-inf'), dtype=dtype, device=device),
            torch.tensor(0.0,          dtype=dtype, device=device),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                       # (B,T,h,d)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B,h,T,d)
        
        att = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=self._mask(T, x.device, q.dtype),
            dropout_p=self.drop.p,
            is_causal=False,
        )
        y = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)



class RandomCausalMHA_xformers(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # QKV projection and output projection
        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        device = x.device

        # Project QKV and split
        qkv = self.to_qkv(x)                     # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multihead: (B, heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Random mask 
        perm = torch.randperm(T, device=device)
        tril = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        bool_mask = ~tril[:, perm]  # shape (T, T)

        # Convert boolean mask to additive bias
        # xformers expects a float tensor where masked positions are a large negative value
        attn_bias = bool_mask.to(torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
        attn_bias = attn_bias.expand(B, self.num_heads, T, T) * -1e9

        # Apply xformers attention
        out = memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=self.dropout.p)

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(out)
        return self.out_proj(out)



class Block(nn.Module):
    def __init__(self, emb_dim, n_head, dropout, xformers, ff_mult=2):
        super().__init__()
        self.ln1  = nn.LayerNorm(emb_dim)
        if xformers:
            self.attn = RandomCausalMHA_xformers(emb_dim, n_head, dropout)
        else:
            self.attn = RandomCausalMHA(emb_dim, n_head, dropout)
        self.ln2  = nn.LayerNorm(emb_dim)
        self.ff   = nn.Sequential(
            nn.Linear(emb_dim, ff_mult * emb_dim),
            nn.GELU(),
            nn.Linear(ff_mult * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


#################################################################################
###  Full model:
#################################################################################

class Pontus(nn.Module):
    def __init__(
        self, n_genes, n_bins = 32, emb_dim = 256, n_head = 8, 
        n_block = 6, dropout = 0.1, xformers = False, ff_mult = 2,
    ):
        super().__init__()
        self.n_genes = n_genes

        self.embed_count = SoftBinnedGeneEmbedding(n_bins, emb_dim)
        self.embed_pos   = GenePositionalEmbedding(n_genes, emb_dim)

        self.blocks = nn.ModuleList(
            [Block(emb_dim, n_head, dropout, xformers, ff_mult=ff_mult) for _ in range(n_block)]
        )

        self.ln_f = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x, target = None,
                 grad_ckpt = True):
        """
        x:      (B, T, 1) raw or pre-processed counts
        target: (B, T, 1) reconstruction target; 
                if None, loss is not returned
        """
        B, T, _ = x.shape
        assert T == self.n_genes, "Input length must be equal to n_genes"

        h = self.embed_count(x) + self.embed_pos(T)              # (B,T,d)

        for blk in self.blocks:
            if grad_ckpt:
                h = cp.checkpoint(blk, h, use_reentrant=False)
            else:
                h = blk(h)

        out = self.head(self.ln_f(h))                            # (B,T,1)
        loss = self.loss_fn(out, target) if target is not None else None
        return out, loss