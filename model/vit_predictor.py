from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.to(dtype=torch.float32)
        y32 = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + self.eps)
        return (y32 * self.weight).to(dtype=x.dtype)


def channel_norm_2d(x: Tensor, norm: RMSNorm) -> Tensor:
    x32 = x.to(dtype=torch.float32)
    y32 = x32 * torch.rsqrt(x32.square().mean(dim=1, keepdim=True) + norm.eps)
    return (y32 * norm.weight.view(1, -1, 1, 1)).to(dtype=x.dtype)


class RoPE2D(nn.Module):
    def __init__(self, grid_size: int, d_head: int, theta: float = 32.0) -> None:
        super().__init__()
        if d_head % 4 != 0:
            raise ValueError(f"d_head={d_head} must be divisible by 4 for 2D RoPE")

        self.grid_size = int(grid_size)
        self.d_head = int(d_head)
        self.theta = float(theta)

        n_tokens = self.grid_size * self.grid_size
        d_axis = self.d_head // 2
        idx = torch.arange(0, d_axis, 2, dtype=torch.float32)
        freqs = 1.0 / (self.theta ** (idx / float(d_axis)))
        pos = torch.arange(n_tokens, dtype=torch.int64)
        rows = (pos // self.grid_size).to(dtype=torch.float32)
        cols = (pos % self.grid_size).to(dtype=torch.float32)

        row_angles = rows.unsqueeze(1) * freqs.unsqueeze(0)
        col_angles = cols.unsqueeze(1) * freqs.unsqueeze(0)
        self.register_buffer("row_cos", torch.cos(row_angles).unsqueeze(0).unsqueeze(2), persistent=False)
        self.register_buffer("row_sin", torch.sin(row_angles).unsqueeze(0).unsqueeze(2), persistent=False)
        self.register_buffer("col_cos", torch.cos(col_angles).unsqueeze(0).unsqueeze(2), persistent=False)
        self.register_buffer("col_sin", torch.sin(col_angles).unsqueeze(0).unsqueeze(2), persistent=False)

    @staticmethod
    def _rotate_into(y: Tensor, x: Tensor, cos: Tensor, sin: Tensor) -> None:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        y[..., 0::2] = x_even * cos - x_odd * sin
        y[..., 1::2] = x_even * sin + x_odd * cos

    def _apply_rope(self, x: Tensor) -> Tensor:
        x32 = x.to(dtype=torch.float32)
        y32 = torch.empty_like(x32)
        d_axis = self.d_head // 2
        self._rotate_into(y32[..., :d_axis], x32[..., :d_axis], self.row_cos, self.row_sin)
        self._rotate_into(y32[..., d_axis:], x32[..., d_axis:], self.col_cos, self.col_sin)
        return y32.to(dtype=x.dtype)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        if q.shape[1] != self.grid_size * self.grid_size:
            raise ValueError(f"Expected {self.grid_size * self.grid_size} tokens, got {q.shape[1]}")
        return self._apply_rope(q), self._apply_rope(k)

class ViTBlock(nn.Module):
    def __init__(self, width: int, n_heads: int, mlp_dim: int) -> None:
        super().__init__()
        if width % n_heads != 0:
            raise ValueError(f"width={width} must be divisible by n_heads={n_heads}")

        d_head = width // n_heads
        if d_head % 4 != 0:
            raise ValueError(
                f"Per-head dimension must be divisible by 4 for 2D RoPE. "
                f"Got width={width}, n_heads={n_heads}, d_head={d_head}"
            )

        self.width = int(width)
        self.n_heads = int(n_heads)
        self.d_head = int(d_head)

        self.attn_norm = RMSNorm(width)
        self.qkv_proj = nn.Linear(width, 3 * width, bias=False)
        self.qk_norm = RMSNorm(d_head)
        self.out_proj = nn.Linear(width, width, bias=False)

        self.ffn_norm = RMSNorm(width)
        self.gate_proj = nn.Linear(width, mlp_dim, bias=False)
        self.up_proj = nn.Linear(width, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, width, bias=False)

    def attention(self, x: Tensor, rope: RoPE2D) -> Tensor:
        qkv = self.qkv_proj(x).view(x.shape[0], x.shape[1], 3, self.n_heads, self.d_head)
        q = self.qk_norm(qkv[:, :, 0])
        k = self.qk_norm(qkv[:, :, 1])
        v = qkv[:, :, 2]

        q, k = rope(q, k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return y.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.width)

    def forward(self, x: Tensor, rope: RoPE2D) -> Tensor:
        x = x + self.out_proj(self.attention(self.attn_norm(x), rope=rope))
        y = self.ffn_norm(x)
        y = F.silu(self.gate_proj(y)) * self.up_proj(y)
        return x + self.down_proj(y)


class Transformer2D(nn.Module):
    def __init__(self, *, width: int, depth: int, n_heads: int, mlp_dim: int, grid_size: int, rope_theta: float = 32.0) -> None:
        super().__init__()
        if width % n_heads != 0:
            raise ValueError(f"width={width} must be divisible by n_heads={n_heads}")
        self.grid_size = int(grid_size)
        self.rope = RoPE2D(grid_size=grid_size, d_head=width // n_heads, theta=rope_theta)
        self.blocks = nn.ModuleList([ViTBlock(width=width, n_heads=n_heads, mlp_dim=mlp_dim) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        if height != self.grid_size or width != self.grid_size:
            raise ValueError(f"Expected patch grid {(self.grid_size, self.grid_size)}, got {(height, width)}")
        y = x.flatten(2).transpose(1, 2).contiguous()
        for block in self.blocks:
            y = block(y, rope=self.rope)
        return y.transpose(1, 2).reshape(batch, channels, height, width)


class SingleStepViTPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        image_size: int = 256,
        patch_size: int = 8,
        width: int = 128,
        depth: int = 12,
        n_heads: int = 8,
        mlp_dim: int = 512,
        cleanup_hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.embed = nn.Conv2d(
            in_channels,
            width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.embed_norm = RMSNorm(width)
        self.transformer = Transformer2D(
            width=width,
            depth=depth,
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            grid_size=self.grid_size,
            rope_theta=32.0,
        )
        self.unpatchify = nn.Sequential(
            nn.Conv2d(width, out_channels * self.patch_size * self.patch_size, kernel_size=1, bias=False),
            nn.PixelShuffle(self.patch_size),
        )
        self.cleanup = nn.Sequential(
            nn.Conv2d(out_channels, cleanup_hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cleanup_hidden_channels, cleanup_hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cleanup_hidden_channels, cleanup_hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cleanup_hidden_channels, out_channels, kernel_size=3, padding=1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, x: Tensor) -> Tensor:
        if tuple(x.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(f"Expected input spatial shape {(self.image_size, self.image_size)}, got {tuple(x.shape[-2:])}")
        x = self.embed(x)
        x = channel_norm_2d(x, self.embed_norm)
        x = self.transformer(x)
        x = self.unpatchify(x)
        return x + self.cleanup(x)
