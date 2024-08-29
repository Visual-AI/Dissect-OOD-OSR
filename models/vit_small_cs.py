# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_for_small_dataset.py

from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn
from methods.ARPL.arpl_models.ABN import MultiLayerNorm

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from collections import OrderedDict
import operator
from itertools import islice
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = MultiLayerNorm(dim, 2)
        self.fn = fn
    def forward(self, x, domain_label=None, **kwargs):
        ans, _ = self.norm(x, domain_label)
        return self.fn(ans, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, **kwargs):
        return self.net(x)

class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, domain_label=None, return_feat=False):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if return_feat:
            return attn, self.to_out(out)
        else:
            return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, domain_label=None, return_feat=False):
        attn_weights = []
        for attn, ff in self.layers:
            if return_feat:
                at, out = attn(x, domain_label, return_feat=return_feat)
                attn_weights.append(at)
            else:
                out = attn(x, domain_label, return_feat=return_feat)
            x = out + x
            x = ff(x, domain_label) + x
        if return_feat:
            return x, attn_weights
        else:
            return x

class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.ln = MultiLayerNorm(patch_dim, 2)
        self.fc = nn.Linear(patch_dim, dim)

    def forward(self, x, domain_label):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        ans = self.to_patch_tokens(x_with_shifts)
        ans, _ = self.ln(ans, domain_label)
        ans = self.fc(ans)
        return ans

class ViT_cs(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.ln = MultiLayerNorm(dim, 2)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img, return_feat=False, domain_label=None):
        if domain_label is None:
            domain_label = 0 * torch.ones(img.shape[0], dtype=torch.long).cuda()

        x = self.to_patch_embedding(img, domain_label)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        if return_feat:
            x, attn_weights = self.transformer(x, domain_label, return_feat)
        else:
            x = self.transformer(x, domain_label, return_feat)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        ln_x, _ = self.ln(x, domain_label)

        if return_feat:
            return x, self.mlp_head(ln_x)
        else:
            return self.mlp_head(ln_x)

    def forward_threshold(self, img, threshold=1e10, domain_label=None, return_feat=False):
        if domain_label is None:
            domain_label = 0 * torch.ones(img.shape[0], dtype=torch.long).cuda()

        x = self.to_patch_embedding(img, domain_label)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        if return_feat:
            x, attn_weights = self.transformer(x, return_feat)
        else:
            x = self.transformer(x, domain_label, return_feat)

        x = x.clip(max=threshold)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        ln_x, _ = self.ln(x, domain_label)

        if return_feat:
            return x, self.mlp_head(ln_x)
        else:
            return self.mlp_head(ln_x)