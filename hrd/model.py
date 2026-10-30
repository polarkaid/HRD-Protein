"""Turn residue features into protein vectors with attention pooling."""
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight,
                            self.bias, self.eps).to(x.dtype)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj_weight = nn.Parameter(torch.empty(256, 256))
        self.k_proj_weight = nn.Parameter(torch.empty(256, 1024))
        self.v_proj_weight = nn.Parameter(torch.empty(256, 1024))
        self.in_proj_bias = nn.Parameter(torch.zeros(768))
        self.out_proj = nn.Linear(256, 256)
        for w in (self.q_proj_weight, self.k_proj_weight, self.v_proj_weight):
            nn.init.xavier_uniform_(w)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, q, x, valid):
        bq, bk, bv = self.in_proj_bias.chunk(3)
        q = F.linear(q, self.q_proj_weight, bq)
        k = F.linear(x, self.k_proj_weight, bk)
        v = F.linear(x, self.v_proj_weight, bv)
        q, k, v = [t.reshape(len(x), -1, 4, 64).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :], scale=0.125)
        return self.out_proj(y.transpose(1, 2).reshape(len(x), 8, 256))


class Pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.randn(8, 256))
        self.attn = Attention()
        self.ln_q = LayerNorm(256)

    def forward(self, x, valid):
        return self.attn(self.ln_q(self.query)[None].expand(len(x), -1, -1), x, valid)


class PostPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = Pool()
        self.ln_post = LayerNorm(256)

    def forward(self, x, valid):
        return self.ln_post(self.pool(x, valid))


class HRD(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        self.model = nn.Module()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model.pool = PostPool()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            first = nn.Linear(2048, 4096)
            nn.init.trunc_normal_(first.weight, std=0.02)
            nn.init.zeros_(first.bias)
            last = nn.Linear(4096, 512, bias=False)
            nn.init.trunc_normal_(last.weight, std=0.02)
        self.model.project = nn.Sequential(first, nn.GELU(), last)

    def forward(self, x, valid):
        pooled = self.model.pool(x, valid)
        return F.normalize(self.model.project(pooled.flatten(1)).float(), dim=-1)
