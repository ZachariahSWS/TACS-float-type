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
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("Transformer width must be divisible by head count")
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = width // heads
        # nn.Dropout holds no parameters, so enabling this keeps existing
        # checkpoints loadable
        self.dropout = float(dropout)
        self.residual_dropout = nn.Dropout(self.dropout)
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

    def _qkv(self, x: torch.Tensor):
        batch, length, _ = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(
            batch, length, 3, self.heads, self.head_dim
        )
        q = self.q_norm(qkv[:, :, 0]).transpose(1, 2)
        k = self.k_norm(qkv[:, :, 1]).transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        return q, k, v

    def _finish(self, x: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        attended = attended.transpose(1, 2).reshape(batch, length, self.width)
        x = x + self.residual_dropout(self.proj(attended))
        y = self.norm2(x)
        gate, up = self.gate_up(y).chunk(2, dim=-1)
        return x + self.residual_dropout(self.down(F.silu(gate) * up))

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        q, k, v = self._qkv(x)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self._finish(x, attended)

    def forward_cached(
        self, x: torch.Tensor, past: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Inference step for new tokens ``x`` that attend, unmasked, to every
        cached key/value and to each other; returns the output and the grown
        cache. Exact for block-causal generation, where earlier tokens never
        attend to later ones."""
        q, k, v = self._qkv(x)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        return self._finish(x, F.scaled_dot_product_attention(q, k, v)), (k, v)


class TransformerStack(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        heads: int,
        mlp_ratio: float = 4.0,
        activation_checkpointing: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(width, heads, mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
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

    def forward_cached(self, x: torch.Tensor, caches: list | None = None):
        """``TransformerBlock.forward_cached`` through every block: returns the
        normalised output and the per-block key/value caches."""
        grown = []
        for index, block in enumerate(self.blocks):
            x, cache = block.forward_cached(x, None if caches is None else caches[index])
            grown.append(cache)
        return self.final_norm(x), grown
