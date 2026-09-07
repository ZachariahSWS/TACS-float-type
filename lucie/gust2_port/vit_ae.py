"""ViT autoencoder — port of gust2/vit_ae.py (JAX/Equinox) to PyTorch.

Kept identical: strided-conv stem (kernel 3, stride 2, pad 1, SiLU + channel
RMSNorm after every conv), transposed-conv decoder stem (kernel 4, stride 2,
pad 1, SiLU + RMSNorm after all but the last), pre-norm ViT blocks with
RMSNorm, fused bias-free QKV, a SHARED QK RMSNorm over d_head, 2D RoPE (first
half of head dim <- row position, second half <- column), SwiGLU FFN
(down(silu(gate(x)) * up(x))), bias-free projections, encoder RMSNorm +
Linear(d, codebook_dim), decoder Linear(codebook_dim, d). Norms run in f32
and cast back, as gust2 does.

Equinox defaults reproduced: RMSNorm(eps=1e-5) has BOTH weight and bias;
Linear/Conv init is U(-1/sqrt(fan_in), 1/sqrt(fan_in)) (PyTorch's default).

One generalisation: ``n_downsample`` (gust2 hard-codes 3 = 8x, 256 -> 32).
On the 48x96 ERA5 grid two strided convs (4x) give the 12x24 latent that
matches the LUCIE default ladder; channel widths follow gust2's progression
(d/4, d/2, d for 3 convs; d/2, d for 2).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Equinox RMSNorm: weight * x * rsqrt(mean(x^2) + eps) + bias, in f32."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        inv = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * (x32 * inv) + self.bias).to(dtype)


def _channel_norm(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
    """RMSNorm over the channel dim of (B, C, H, W)."""
    return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _stem_widths(d: int, n_downsample: int) -> list[int]:
    # gust2: C -> d//4 -> d//2 -> d for three convs
    return [d // (2 ** (n_downsample - 1 - i)) for i in range(n_downsample)]


class EncoderStem(nn.Module):
    def __init__(self, d: int, in_channels: int = 1, n_downsample: int = 3) -> None:
        super().__init__()
        widths = _stem_widths(d, n_downsample)
        convs, norms = [], []
        previous = in_channels
        for width in widths:
            convs.append(nn.Conv2d(previous, width, 3, stride=2, padding=1))
            norms.append(RMSNorm(width))
            previous = width
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = _channel_norm(norm, F.silu(conv(x)))
        return x


class DecoderStem(nn.Module):
    def __init__(self, d: int, out_channels: int = 1, n_downsample: int = 3) -> None:
        super().__init__()
        widths = list(reversed(_stem_widths(d, n_downsample)))  # d, d//2, d//4
        convs, norms = [], []
        for i in range(n_downsample):
            inp = widths[i]
            out = widths[i + 1] if i + 1 < n_downsample else out_channels
            convs.append(nn.ConvTranspose2d(inp, out, 4, stride=2, padding=1))
            if i + 1 < n_downsample:
                norms.append(RMSNorm(out))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = len(self.convs)
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i + 1 < n:
                x = _channel_norm(self.norms[i], F.silu(x))
        return x


# ---------- 2D Rotary Position Embedding ----------


def _rope_freqs(dim: int, theta: float = 10000.0, device=None) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """x: (..., N, heads, dim) f32; pos: (N,) f32 -> rotated x (interleaved pairs)."""
    angles = pos[:, None] * freqs[None, :]                     # (N, dim//2)
    cos, sin = torch.cos(angles), torch.sin(angles)
    # broadcast over leading batch dims and the heads dim: (N, 1, dim//2)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).reshape(x.shape)


def apply_2d_rope_coords(q: torch.Tensor, k: torch.Tensor, rows: torch.Tensor,
                         cols: torch.Tensor, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """2D axial RoPE on (B, N, heads, d_head): first half <- rows, second <- cols.
    Computed in f32 (gust2 promotes to f32 internally), cast back to input dtype."""
    dtype = q.dtype
    d_half = q.shape[-1] // 2
    freqs = _rope_freqs(d_half, theta, device=q.device)
    q32, k32 = q.float(), k.float()
    rows = rows.float()
    cols = cols.float()
    q_out = torch.cat([_apply_rope(q32[..., :d_half], freqs, rows),
                       _apply_rope(q32[..., d_half:], freqs, cols)], dim=-1)
    k_out = torch.cat([_apply_rope(k32[..., :d_half], freqs, rows),
                       _apply_rope(k32[..., d_half:], freqs, cols)], dim=-1)
    return q_out.to(dtype), k_out.to(dtype)


def _apply_2d_rope(q, k, H: int, W: int, theta: float):
    n = torch.arange(H * W, device=q.device)
    return apply_2d_rope_coords(q, k, n // W, n % W, theta)


# ---------- ViT Block ----------


class ViTBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_dim: int, rope_theta: float = 32.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.rope_theta = rope_theta
        self.attn_norm = RMSNorm(d)
        self.qkv_proj = nn.Linear(d, 3 * d, bias=False)
        self.qk_norm = RMSNorm(self.d_head)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.ffn_norm = RMSNorm(d)
        self.gate_proj = nn.Linear(d, mlp_dim, bias=False)
        self.up_proj = nn.Linear(d, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, d, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, N, d), N = H*W
        B, N, d = x.shape
        r = x
        x = self.attn_norm(x)
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]      # (B, N, nh, dh)
        q = self.qk_norm(q)
        k = self.qk_norm(k)
        q, k = _apply_2d_rope(q, k, H, W, self.rope_theta)
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2).reshape(B, N, d)
        x = r + self.out_proj(x)

        r = x
        x = self.ffn_norm(x)
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return r + x


class Transformer(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_dim: int, depth: int,
                 rope_theta: float = 32.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ViTBlock(d, n_heads, mlp_dim, rope_theta) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, d, H, W) -> (B, d, H, W)
        B, d, H, W = x.shape
        x = x.reshape(B, d, H * W).transpose(1, 2)               # (B, N, d)
        for block in self.blocks:
            x = block(x, H, W)
        return x.transpose(1, 2).reshape(B, d, H, W)


class Encoder(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_dim: int, depth: int, codebook_dim: int,
                 in_channels: int = 1, rope_theta: float = 32.0, n_downsample: int = 3) -> None:
        super().__init__()
        self.stem = EncoderStem(d, in_channels, n_downsample)
        self.transformer = Transformer(d, n_heads, mlp_dim, depth, rope_theta)
        self.norm = RMSNorm(d)
        self.proj = nn.Linear(d, codebook_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, codebook_dim, H', W')
        x = self.transformer(self.stem(x))
        B, d, H, W = x.shape
        x = self.proj(self.norm(x.reshape(B, d, H * W).transpose(1, 2)))
        return x.transpose(1, 2).reshape(B, -1, H, W)


class Decoder(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_dim: int, depth: int, codebook_dim: int,
                 out_channels: int = 1, rope_theta: float = 32.0, n_downsample: int = 3) -> None:
        super().__init__()
        self.proj = nn.Linear(codebook_dim, d, bias=False)
        self.transformer = Transformer(d, n_heads, mlp_dim, depth, rope_theta)
        self.stem = DecoderStem(d, out_channels, n_downsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, codebook_dim, H', W') -> (B, C, H, W)
        B, V, H, W = x.shape
        x = self.proj(x.reshape(B, V, H * W).transpose(1, 2))
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        return self.stem(self.transformer(x))


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
