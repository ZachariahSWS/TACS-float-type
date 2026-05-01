from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


Tensor = torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        out_dtype = x.dtype
        x32 = x.to(dtype=torch.float32)
        denom = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y32 = x32 * denom
        return (y32 * self.weight).to(dtype=out_dtype)


def channel_rms_norm(norm: RMSNorm, x: Tensor) -> Tensor:
    batch, channels, height, width = x.shape
    y = x.reshape(batch, channels, height * width).transpose(1, 2)
    y = norm(y)
    return y.transpose(1, 2).reshape(batch, channels, height, width)


def rope_freqs(dim: int, theta: float, device: torch.device) -> Tensor:
    idx = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    return 1.0 / (theta ** (idx / float(dim)))


def apply_rope_1d(x: Tensor, freqs: Tensor, pos: Tensor) -> Tensor:
    batch, n_tokens, n_heads, d_head = x.shape
    x_even = x[..., 0::2].to(dtype=torch.float32)
    x_odd = x[..., 1::2].to(dtype=torch.float32)

    angles = pos.to(dtype=torch.float32).unsqueeze(-1) * freqs.unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(2)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(2)

    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos
    y = torch.stack([y_even, y_odd], dim=-1).reshape(batch, n_tokens, n_heads, d_head)
    return y.to(dtype=x.dtype)


def apply_2d_rope(q: Tensor, k: Tensor, height: int, width: int, theta: float) -> tuple[Tensor, Tensor]:
    n_tokens = q.shape[1]
    d_head = q.shape[-1]
    d_half = d_head // 2

    rows = torch.arange(n_tokens, device=q.device, dtype=torch.int64) // width
    cols = torch.arange(n_tokens, device=q.device, dtype=torch.int64) % width
    row_freqs = rope_freqs(d_half, theta=theta, device=q.device)
    col_freqs = rope_freqs(d_half, theta=theta, device=q.device)

    q_out = torch.cat(
        [
            apply_rope_1d(q[..., :d_half], row_freqs, rows),
            apply_rope_1d(q[..., d_half:], col_freqs, cols),
        ],
        dim=-1,
    )
    k_out = torch.cat(
        [
            apply_rope_1d(k[..., :d_half], row_freqs, rows),
            apply_rope_1d(k[..., d_half:], col_freqs, cols),
        ],
        dim=-1,
    )
    return q_out, k_out


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

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        residual = x
        x = self.attn_norm(x)

        qkv = self.qkv_proj(x)
        qkv = qkv.view(x.shape[0], x.shape[1], 3, self.n_heads, self.d_head)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        q = self.qk_norm(q)
        k = self.qk_norm(k)
        q, k = apply_2d_rope(q, k, height=height, width=width, theta=self.rope_theta)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(residual.shape[0], residual.shape[1], self.width)
        x = residual + self.out_proj(attn_out)

        residual = x
        x = self.ffn_norm(x)
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return residual + self.down_proj(x)


class Transformer2D(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        n_heads: int,
        mlp_dim: int,
        rope_theta: float = 32.0,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.blocks = nn.ModuleList(
            [ViTBlock(width=width, n_heads=n_heads, mlp_dim=mlp_dim, rope_theta=rope_theta) for _ in range(depth)]
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        y = x.flatten(2).transpose(1, 2)

        for block in self.blocks:
            if self.training and self.use_gradient_checkpointing:
                y = checkpoint(lambda inp, blk=block: blk(inp, height=height, width=width), y, use_reentrant=False)
            else:
                y = block(y, height=height, width=width)

        return y.transpose(1, 2).reshape(batch, channels, height, width)


class PatchifyEmbed(nn.Module):
    def __init__(self, in_channels: int, width: int, patch_size: int = 8) -> None:
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(int(patch_size))
        self.proj = nn.Conv2d(in_channels * patch_size * patch_size, width, kernel_size=1, bias=False)
        self.norm = RMSNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pixel_unshuffle(x)
        x = self.proj(x)
        return channel_rms_norm(self.norm, x)


class UnpatchifyProject(nn.Module):
    def __init__(self, width: int, out_channels: int, patch_size: int = 8) -> None:
        super().__init__()
        self.proj = nn.Conv2d(width, out_channels * patch_size * patch_size, kernel_size=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(int(patch_size))

    def forward(self, x: Tensor) -> Tensor:
        return self.pixel_shuffle(self.proj(x))


class ConvCleanupHead(nn.Module):
    def __init__(self, channels: int, hidden_channels: int = 32, depth: int = 4) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1), nn.GELU()]
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.GELU())
        layers.append(nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


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
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")

        self.image_size = int(image_size)
        self.embed = PatchifyEmbed(in_channels=in_channels, width=width, patch_size=patch_size)
        self.transformer = Transformer2D(
            width=width,
            depth=depth,
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            rope_theta=32.0,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.unpatchify = UnpatchifyProject(width=width, out_channels=out_channels, patch_size=patch_size)
        self.cleanup = ConvCleanupHead(channels=out_channels, hidden_channels=cleanup_hidden_channels, depth=4)
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
        height = int(x.shape[-2])
        width = int(x.shape[-1])
        if height != self.image_size or width != self.image_size:
            raise ValueError(f"Expected input spatial shape {(self.image_size, self.image_size)}, got {(height, width)}")

        x = self.embed(x)
        x = self.transformer(x)
        x = self.unpatchify(x)
        return self.cleanup(x)
