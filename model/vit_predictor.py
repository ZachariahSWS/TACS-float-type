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
        y32 = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y32 * self.weight).to(dtype=x.dtype)


def channel_last_norm(x: Tensor, norm: RMSNorm) -> Tensor:
    batch, channels, height, width = x.shape
    y = x.reshape(batch, channels, height * width).transpose(1, 2)
    y = norm(y)
    return y.transpose(1, 2).reshape(batch, channels, height, width)


def rope_freqs(dim: int, theta: float, device: torch.device) -> Tensor:
    idx = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    return 1.0 / (theta ** (idx / float(dim)))


def apply_rope_1d(x: Tensor, freqs: Tensor, pos: Tensor) -> Tensor:
    x_even = x[..., 0::2].to(dtype=torch.float32)
    x_odd = x[..., 1::2].to(dtype=torch.float32)

    angles = pos.to(dtype=torch.float32).unsqueeze(-1) * freqs.unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(2)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(2)

    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos
    y = torch.stack([y_even, y_odd], dim=-1).reshape(*x.shape)
    return y.to(dtype=x.dtype)


def apply_2d_rope(q: Tensor, k: Tensor, width: int, theta: float) -> tuple[Tensor, Tensor]:
    n_tokens = q.shape[1]
    d_half = q.shape[-1] // 2

    rows = torch.arange(n_tokens, device=q.device, dtype=torch.int64) // width
    cols = torch.arange(n_tokens, device=q.device, dtype=torch.int64) % width
    row_freqs = rope_freqs(d_half, theta=theta, device=q.device)
    col_freqs = rope_freqs(d_half, theta=theta, device=q.device)

    q = torch.cat([apply_rope_1d(q[..., :d_half], row_freqs, rows), apply_rope_1d(q[..., d_half:], col_freqs, cols)], dim=-1)
    k = torch.cat([apply_rope_1d(k[..., :d_half], row_freqs, rows), apply_rope_1d(k[..., d_half:], col_freqs, cols)], dim=-1)
    return q, k


class ViTBlock(nn.Module):
    def __init__(self, width: int, n_heads: int, mlp_dim: int, rope_theta: float = 32.0) -> None:
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
        self.rope_theta = float(rope_theta)

        self.attn_norm = RMSNorm(width)
        self.qkv_proj = nn.Linear(width, 3 * width, bias=False)
        self.qk_norm = RMSNorm(d_head)
        self.out_proj = nn.Linear(width, width, bias=False)

        self.ffn_norm = RMSNorm(width)
        self.gate_proj = nn.Linear(width, mlp_dim, bias=False)
        self.up_proj = nn.Linear(width, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, width, bias=False)

    def attention(self, x: Tensor, width: int) -> Tensor:
        qkv = self.qkv_proj(x).view(x.shape[0], x.shape[1], 3, self.n_heads, self.d_head)
        q = self.qk_norm(qkv[:, :, 0])
        k = self.qk_norm(qkv[:, :, 1])
        v = qkv[:, :, 2]

        q, k = apply_2d_rope(q, k, width=width, theta=self.rope_theta)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return y.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.width)

    def forward(self, x: Tensor, width: int) -> Tensor:
        x = x + self.out_proj(self.attention(self.attn_norm(x), width=width))
        y = self.ffn_norm(x)
        y = F.silu(self.gate_proj(y)) * self.up_proj(y)
        return x + self.down_proj(y)


class Transformer2D(nn.Module):
    def __init__(self, width: int, depth: int, n_heads: int, mlp_dim: int, rope_theta: float = 32.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ViTBlock(width=width, n_heads=n_heads, mlp_dim=mlp_dim, rope_theta=rope_theta) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        y = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            y = block(y, width=width)
        return y.transpose(1, 2).reshape(batch, channels, height, width)


class SingleStepViTPredictor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, image_size: int = 256, patch_size: int = 8, width: int = 128, depth: int = 12, n_heads: int = 8, mlp_dim: int = 512, cleanup_hidden_channels: int = 32) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")

        self.image_size = int(image_size)
        self.embed = nn.Sequential(
            nn.PixelUnshuffle(int(patch_size)),
            nn.Conv2d(in_channels * patch_size * patch_size, width, kernel_size=1, bias=False),
        )
        self.embed_norm = RMSNorm(width)
        self.transformer = Transformer2D(
            width=width,
            depth=depth,
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            rope_theta=32.0
        )
        self.unpatchify = nn.Sequential(
            nn.Conv2d(width, out_channels * patch_size * patch_size, kernel_size=1, bias=False),
            nn.PixelShuffle(int(patch_size)),
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
        x = channel_last_norm(x, self.embed_norm)
        x = self.transformer(x)
        x = self.unpatchify(x)
        return x + self.cleanup(x)
