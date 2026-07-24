from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class TransformerBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("Transformer width must be divisible by head count")
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = width // heads
        self.norm1 = nn.RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.proj = nn.Linear(width, width, bias=False)
        hidden = int(width * mlp_ratio)
        self.norm2 = nn.RMSNorm(width)
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        fused_key = prefix + "gate_up.weight"
        gate_key = prefix + "gate.weight"
        up_key = prefix + "up.weight"
        if fused_key not in state_dict and gate_key in state_dict:
            state_dict[fused_key] = torch.cat(
                [state_dict.pop(gate_key), state_dict.pop(up_key)], dim=0
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(
            batch, length, 3, self.heads, self.head_dim
        )
        q = self.q_norm(qkv[:, :, 0]).transpose(1, 2)
        k = self.k_norm(qkv[:, :, 1]).transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0
        )
        attended = attended.transpose(1, 2).reshape(batch, length, self.width)
        x = x + self.proj(attended)
        y = self.norm2(x)
        gate, up = self.gate_up(y).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


class TransformerStack(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        heads: int,
        mlp_ratio: float = 4.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(width, heads, mlp_ratio) for _ in range(depth)]
        )
        self.activation_checkpointing = bool(activation_checkpointing)
        self.final_norm = nn.RMSNorm(width)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                x = checkpoint(
                    block, x, attention_mask, use_reentrant=False
                )
            else:
                x = block(x, attention_mask)
        return self.final_norm(x)
