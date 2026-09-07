"""Next-Scale Prediction model — port of gust2/nsp_model.py to PyTorch.

Everything structural is kept: NSPConfig with data-derived fields, sequences
padded to multiples of 128 as [full t0 | t1 truncated to all-but-last
scale], scale/frame embeddings on top of a FROZEN codebook projection,
pre-norm blocks (RMSNorm, fused QKV, shared QK RMSNorm, 2D axial RoPE on
cell-centre coordinates of the finest grid, SwiGLU with hidden
round64(4d*2/3)), the asymmetric attention mask (t0->t0 full, t0->t1 blocked,
t1->t0 full, t1->t1 source_scale <= target_scale, padding attends only to
itself), ExpansionHeads (per trainable scale a bias-free Linear(n_embd,
effective_vocab); shared pos_proj Linear(2, n_embd) WITH bias on normalised
target coordinates; a shared stack of n_refine_layers NSPBlocks run over the
target grid with local cell-centre RoPE coords), teacher-forced forward,
greedy / temperature / top-k / top-p generation with per-scale (and optional
per-position) support masks, and rope_theta auto = max grid side.

Batched natively (gust2 vmaps a single-sample function).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_ae import RMSNorm, apply_2d_rope_coords

NEG = -1e9


@dataclass
class NSPConfig:
    scales: tuple = ()
    effective_vocab_size: int = 0
    codebook_dim: int = 0
    first_trainable_scale: int = 0
    tokens_per_frame: int = 0
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    rope_theta: float | None = None
    n_refine_layers: int = 2

    @property
    def n_scales(self) -> int:
        return len(self.scales)

    @property
    def scale_boundaries(self) -> list:
        boundaries = [0]
        for h, w in self.scales:
            boundaries.append(boundaries[-1] + h * w)
        return boundaries

    @property
    def padded_seq_len(self) -> int:
        return ((self.tokens_per_frame + 127) // 128) * 128

    @property
    def trainable_scale_indices(self) -> list:
        return list(range(self.first_trainable_scale, self.n_scales))


def padded_len(n_tokens: int) -> int:
    return ((n_tokens + 127) // 128) * 128


def build_rope_coords(scales: tuple, padded: int) -> torch.Tensor:
    """(padded, 2) cell-centre coordinates on the finest grid; padding -> (0, 0)."""
    if len(scales) == 0:
        return torch.zeros(padded, 2)
    max_h = max(h for h, w in scales)
    max_w = max(w for h, w in scales)
    coords = []
    for h, w in scales:
        step_h, step_w = max_h / h, max_w / w
        for i in range(h):
            for j in range(w):
                coords.append((i * step_h + step_h / 2, j * step_w + step_w / 2))
    while len(coords) < padded:
        coords.append((0.0, 0.0))
    return torch.tensor(coords, dtype=torch.float32)


def get_scale_ids(scales: tuple, padded: int) -> torch.Tensor:
    ids = []
    for k, (h, w) in enumerate(scales):
        ids.extend([k] * (h * w))
    ids.extend([len(scales)] * (padded - len(ids)))
    return torch.tensor(ids, dtype=torch.long)


def build_teacher_forced_mask(scales_t0, padded_t0, scales_t1, padded_t1) -> torch.Tensor:
    """Boolean (L, L) mask, True = may attend (gust2's additive -1e9 mask, as bool)."""
    L0, L1 = padded_t0, padded_t1
    total = L0 + L1
    allowed = torch.zeros(total, total, dtype=torch.bool)
    allowed[:L0, :L0] = True                       # t0 -> t0 full
    allowed[L0:, :L0] = True                       # t1 -> t0 full
    ids_t1 = get_scale_ids(scales_t1, L1)          # t1 -> t1 : source_scale <= target_scale
    allowed[L0:, L0:] = ids_t1[None, :] <= ids_t1[:, None]
    n0 = sum(h * w for h, w in scales_t0)
    n1 = sum(h * w for h, w in scales_t1)
    is_padding = torch.cat([torch.arange(L0) >= n0, torch.arange(L1) >= n1])
    eye = torch.eye(total, dtype=torch.bool)
    allowed[is_padding] = False                    # padding rows: identity only
    allowed[:, is_padding] = False                 # nobody attends to padding
    allowed = allowed | (is_padding[:, None] & eye)
    return allowed


class NSPEmbedding(nn.Module):
    def __init__(self, config: NSPConfig, codebook: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("codebook", codebook.detach().float().clone())   # frozen
        self.codebook_proj = nn.Linear(config.codebook_dim, config.n_embd, bias=False)
        self.scale_embed = nn.Embedding(config.n_scales, config.n_embd)
        self.frame_embed = nn.Embedding(2, config.n_embd)

    def forward(self, tokens, scale_ids, frame_ids):
        vectors = self.codebook[tokens.long()].detach()
        tok_emb = self.codebook_proj(vectors.to(self.codebook_proj.weight.dtype))
        scale_ids = scale_ids.clamp(max=self.scale_embed.num_embeddings - 1)
        return tok_emb + self.scale_embed(scale_ids) + self.frame_embed(frame_ids)


class NSPBlock(nn.Module):
    def __init__(self, config: NSPConfig) -> None:
        super().__init__()
        d = config.n_embd
        self.n_heads = config.n_head
        self.d_head = d // config.n_head
        self.rope_theta = float(config.rope_theta)
        self.attn_norm = RMSNorm(d)
        self.qkv_proj = nn.Linear(d, 3 * d, bias=False)
        self.qk_norm = RMSNorm(self.d_head)
        self.out_proj = nn.Linear(d, d, bias=False)
        hidden = (4 * d * 2) // 3
        hidden = ((hidden + 63) // 64) * 64
        self.ffn_norm = RMSNorm(d)
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)

    def forward(self, x, attn_mask, rope_coords):
        """x (B, L, d); attn_mask bool (L, L) or None; rope_coords (L, 2)."""
        B, L, d = x.shape
        r = x
        x = self.attn_norm(x)
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = self.qk_norm(q)
        k = self.qk_norm(k)
        q, k = apply_2d_rope_coords(q, k, rope_coords[:, 0], rope_coords[:, 1], self.rope_theta)
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=None if attn_mask is None else attn_mask[None, None],
        ).transpose(1, 2).reshape(B, L, d)
        x = r + self.out_proj(x)
        r = x
        x = self.ffn_norm(x)
        return r + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class NSPModel(nn.Module):
    def __init__(self, config: NSPConfig, codebook: torch.Tensor) -> None:
        super().__init__()
        self.config = config
        self.embedding = NSPEmbedding(config, codebook)
        self.blocks = nn.ModuleList([NSPBlock(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.remat = True

    def forward(self, tokens, scale_ids, frame_ids, attn_mask, rope_coords):
        x = self.embedding(tokens, scale_ids, frame_ids)
        for block in self.blocks:
            if self.remat and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, attn_mask, rope_coords,
                                                      use_reentrant=False)
            else:
                x = block(x, attn_mask, rope_coords)
        return self.ln_f(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ExpansionHeads(nn.Module):
    def __init__(self, config: NSPConfig) -> None:
        super().__init__()
        n = len(config.trainable_scale_indices)
        self.heads = nn.ModuleList(
            [nn.Linear(config.n_embd, config.effective_vocab_size, bias=False) for _ in range(n)])
        self.pos_proj = nn.Linear(2, config.n_embd, bias=True)
        self.refinement = nn.ModuleList(
            [NSPBlock(config) for _ in range(max(int(config.n_refine_layers), 0))])

    def expand(self, h, head_idx, local_coords):
        """h (B, K, n_embd) positioned hidden states -> (B, K, effective_vocab) logits."""
        for blk in self.refinement:
            h = blk(h, None, local_coords)
        return self.heads[head_idx](h)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _local_cell_coords(h_k: int, w_k: int, device) -> torch.Tensor:
    rows = torch.arange(h_k, dtype=torch.float32, device=device) + 0.5
    cols = torch.arange(w_k, dtype=torch.float32, device=device) + 0.5
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([grid_r, grid_c], dim=-1).reshape(h_k * w_k, 2)


def _target_pos_coords(h_k: int, w_k: int, device) -> torch.Tensor:
    rows = torch.arange(h_k, dtype=torch.float32, device=device) / max(h_k - 1, 1)
    cols = torch.arange(w_k, dtype=torch.float32, device=device) / max(w_k - 1, 1)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([grid_r, grid_c], dim=-1)          # (h_k, w_k, 2)


class SequenceLayout:
    """Static per-config tensors shared by training, generation and eval."""

    def __init__(self, config: NSPConfig, device) -> None:
        self.config = config
        self.scales_t0 = tuple(config.scales)
        self.scales_t1 = tuple(config.scales[:-1])
        self.tokens_t0 = sum(h * w for h, w in self.scales_t0)
        self.tokens_t1 = sum(h * w for h, w in self.scales_t1)
        self.padded_t0 = padded_len(self.tokens_t0)
        self.padded_t1 = padded_len(self.tokens_t1)
        self.attn_mask = build_teacher_forced_mask(
            self.scales_t0, self.padded_t0, self.scales_t1, self.padded_t1).to(device)
        self.scale_ids = torch.cat([get_scale_ids(self.scales_t0, self.padded_t0),
                                    get_scale_ids(self.scales_t1, self.padded_t1)]).to(device)
        self.frame_ids = torch.cat([torch.zeros(self.padded_t0, dtype=torch.long),
                                    torch.ones(self.padded_t1, dtype=torch.long)]).to(device)
        self.rope_coords = torch.cat([build_rope_coords(self.scales_t0, self.padded_t0),
                                      build_rope_coords(self.scales_t1, self.padded_t1)]).to(device)
        self.boundaries_t0 = [0]
        for h, w in self.scales_t0:
            self.boundaries_t0.append(self.boundaries_t0[-1] + h * w)
        self.boundaries_t1 = [0]
        for h, w in self.scales_t1:
            self.boundaries_t1.append(self.boundaries_t1[-1] + h * w)
        self.boundaries = config.scale_boundaries
        self.device = device

    def pack(self, t0_full: torch.Tensor, t1_trunc: torch.Tensor) -> torch.Tensor:
        """(B, tokens) + (B, tokens_t1) -> padded (B, L0 + L1) input."""
        B = t0_full.shape[0]
        t0 = F.pad(t0_full, (0, self.padded_t0 - self.tokens_t0))
        t1 = F.pad(t1_trunc, (0, self.padded_t1 - self.tokens_t1))
        return torch.cat([t0, t1], dim=1)

    def source_hidden(self, hidden: torch.Tensor, scale_idx: int):
        """Hidden states of scale k-1 (t0's scale 0 for k = 0) -> (B, h_src, w_src, d)."""
        h_t0 = hidden[:, :self.padded_t0]
        h_t1 = hidden[:, self.padded_t0:]
        if scale_idx == 0:
            h_src, w_src = self.config.scales[0]
            src = h_t0[:, self.boundaries_t0[0]:self.boundaries_t0[1]]
        else:
            j = scale_idx - 1
            h_src, w_src = self.scales_t1[j]
            src = h_t1[:, self.boundaries_t1[j]:self.boundaries_t1[j + 1]]
        return src.reshape(src.shape[0], h_src, w_src, -1)


def forward_teacher_forced(model: NSPModel, tokens_in: torch.Tensor, layout: SequenceLayout):
    return model(tokens_in, layout.scale_ids, layout.frame_ids, layout.attn_mask, layout.rope_coords)


def scale_logits(exp_heads: ExpansionHeads, hidden: torch.Tensor, layout: SequenceLayout,
                 scale_idx: int, head_idx: int) -> torch.Tensor:
    """Bilinear-upsample scale k-1 hidden states, add target position encoding,
    refine, project -> (B, h_k*w_k, effective_vocab) logits (unmasked)."""
    config = layout.config
    h_k, w_k = config.scales[scale_idx]
    src = layout.source_hidden(hidden, scale_idx)                    # (B, hs, ws, d)
    up = F.interpolate(src.permute(0, 3, 1, 2), size=(h_k, w_k), mode="bilinear",
                       align_corners=False).permute(0, 2, 3, 1)      # (B, h_k, w_k, d)
    pos = exp_heads.pos_proj(_target_pos_coords(h_k, w_k, hidden.device).to(up.dtype))
    h_flat = (up + pos[None]).reshape(up.shape[0], h_k * w_k, config.n_embd)
    return exp_heads.expand(h_flat, head_idx, _local_cell_coords(h_k, w_k, hidden.device))


def apply_support_mask(logits, scale_masks, scale_idx, position_mask, layout):
    logits = logits.float()
    if position_mask is not None:
        s, e = layout.boundaries[scale_idx], layout.boundaries[scale_idx + 1]
        return logits.masked_fill(~position_mask[s:e][None], NEG)
    return logits.masked_fill(~scale_masks[scale_idx][None, None], NEG)


def sample_ancestral(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
                     generator=None) -> torch.Tensor:
    """gust2's legacy sampler: greedy (T == 0) else temperature + top-k / top-p."""
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    if top_k > 0:
        kth = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, NEG)
    if top_p < 1.0:
        sorted_logits, _ = logits.sort(dim=-1, descending=True)
        probs = (sorted_logits / temperature).softmax(dim=-1)
        cum = probs.cumsum(dim=-1)
        shifted = torch.cat([torch.zeros_like(cum[..., :1]), cum[..., :-1]], dim=-1)
        keep = shifted < top_p
        kept = torch.where(keep, sorted_logits, torch.full_like(sorted_logits, float("inf")))
        threshold = kept.min(dim=-1, keepdim=True).values
        logits = logits.masked_fill(logits < threshold, NEG)
    probs = (logits / temperature).softmax(dim=-1)
    flat = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=generator)
    return flat.reshape(probs.shape[:-1])


@torch.no_grad()
def generate_t1_frame(model: NSPModel, exp_heads: ExpansionHeads, layout: SequenceLayout,
                      t0_tokens: torch.Tensor, scale_masks: torch.Tensor,
                      temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0,
                      position_mask: torch.Tensor | None = None, generator=None,
                      autocast=None) -> torch.Tensor:
    """(B, tokens_per_frame) t0 compact tokens -> (B, tokens_per_frame) t1 tokens.

    One forward pass per trainable scale; deterministic scales are copied
    from t0 (single valid code across the dataset), exactly as gust2.
    """
    from contextlib import nullcontext
    autocast = autocast or nullcontext
    config = layout.config
    trainable = config.trainable_scale_indices
    B = t0_tokens.shape[0]
    t1 = torch.zeros(B, config.tokens_per_frame, dtype=torch.long, device=t0_tokens.device)
    first = layout.boundaries[trainable[0]]
    t1[:, :first] = t0_tokens[:, :first]
    for i, scale_idx in enumerate(trainable):
        tokens_in = layout.pack(t0_tokens, t1[:, :layout.tokens_t1])
        with autocast():
            hidden = forward_teacher_forced(model, tokens_in, layout)
            logits = scale_logits(exp_heads, hidden, layout, scale_idx, i)
        logits = apply_support_mask(logits, scale_masks, scale_idx, position_mask, layout)
        predicted = sample_ancestral(logits, temperature, top_k, top_p, generator)
        s, e = layout.boundaries[scale_idx], layout.boundaries[scale_idx + 1]
        t1[:, s:e] = predicted
    return t1


def create_nsp_model(token_data: dict, config: NSPConfig):
    codebook = torch.as_tensor(token_data["codebook"]).float()
    config.scales = tuple(tuple(s) for s in token_data["scales"])
    config.effective_vocab_size = int(token_data["effective_vocab_size"])
    config.codebook_dim = int(token_data["codebook_dim"])
    config.tokens_per_frame = sum(h * w for h, w in config.scales)
    config.first_trainable_scale = int(token_data.get("first_trainable_scale", 0))
    if config.rope_theta is None:
        config.rope_theta = float(max(max(h, w) for h, w in config.scales))
    model = NSPModel(config, codebook)
    exp_heads = ExpansionHeads(config)
    print(f"NSPModel: {model.get_num_params()/1e6:.2f}M parameters")
    print(f"ExpansionHeads: {exp_heads.get_num_params()/1e6:.2f}M parameters")
    print(f"Scales: {config.scales}")
    print(f"Tokens/frame: {config.tokens_per_frame}, padded: {config.padded_seq_len}")
    print(f"Effective vocab: {config.effective_vocab_size}")
    print(f"First trainable scale: {config.first_trainable_scale}")
    print(f"RoPE theta: {config.rope_theta}, refine layers: {config.n_refine_layers}")
    return model, exp_heads
