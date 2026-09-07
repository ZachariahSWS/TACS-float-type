"""Multi-scale residual VQ — port of gust2/vq.py to PyTorch.

Identical algorithm: VAR-style residual quantisation over a scale ladder
against a SHARED codebook that lives outside the module (EMAState), nearest
neighbour by ||z||^2-free distance, straight-through estimator, commitment
loss, per-scale phi Conv2d(D, D, 3, pad 1) + shared post_quant_conv, bicubic
resizing between scales, EMA codebook update with Laplace smoothing and
dead-code reinitialisation (threshold 0.25 x uniform usage, random unit
normals), and the two losses (compound / simple).

Deviations: scales are (h, w) pairs (gust2: ints s -> s x s) so the 48x96
grid's 12x24 latent can use LUCIE's 1x2 ... 12x24 ladder; bicubic resize is
torch's (cubic a=-0.75, antialiased when downsampling) versus jax.image's
Keys kernel (a=-0.5) — same operation, marginally different kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

Scale = tuple[int, int]


def bicubic_resize(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(B, D, H_in, W_in) -> (B, D, h, w)."""
    if x.shape[-2:] == (h, w):
        return x
    down = h < x.shape[-2] or w < x.shape[-1]
    return F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False,
                         antialias=down)


def quantize(z_flat: torch.Tensor, codebook: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest-neighbour quantise with STE.

    z_flat: (..., D); codebook: (K, D).
    Returns z_q (..., D) with straight-through gradient, indices (...,), commit loss.
    """
    dists = -2.0 * (z_flat @ codebook.T) + (codebook ** 2).sum(dim=1)
    indices = dists.argmin(dim=-1)
    z_q_raw = codebook[indices]
    z_q = z_flat + (z_q_raw - z_flat).detach()
    commit_loss = ((z_flat - z_q_raw.detach()) ** 2).mean()
    return z_q, indices, commit_loss


class MultiScaleVQ(nn.Module):
    """Residual multi-scale quantiser; the codebook is an ARGUMENT, not state."""

    def __init__(self, codebook_dim: int, scales: tuple[Scale, ...],
                 full_size: Scale) -> None:
        super().__init__()
        self.scales = tuple((int(h), int(w)) for h, w in scales)
        self.full_size = (int(full_size[0]), int(full_size[1]))
        self.codebook_dim = int(codebook_dim)
        self.phi_convs = nn.ModuleList(
            [nn.Conv2d(codebook_dim, codebook_dim, 3, padding=1) for _ in self.scales])
        self.post_quant_conv = nn.Conv2d(codebook_dim, codebook_dim, 3, padding=1)

    def forward(self, z_e: torch.Tensor, codebook: torch.Tensor):
        """z_e: (B, D, H, W) f32; codebook: (K, D).

        Returns (z_q_final, all_indices [list of (B, h*w)], partials [list of
        (B, D, H, W)], commit_loss (scalar, summed over scales), all_z_flat
        [list of detached (B, h*w, D) residuals for the EMA update]).
        """
        B, D = z_e.shape[:2]
        f_hat = z_e.new_zeros((B, D, 1, 1))
        partials, all_indices, all_z_flat = [], [], []
        commit_loss = z_e.new_zeros(())
        for k, (h, w) in enumerate(self.scales):
            f_hat = bicubic_resize(f_hat, h, w)
            res = bicubic_resize(z_e, h, w) - f_hat
            res_flat = res.reshape(B, D, h * w).transpose(1, 2)        # (B, hw, D)
            z_q_flat, indices, commit = quantize(res_flat, codebook)
            z_q = z_q_flat.transpose(1, 2).reshape(B, D, h, w)
            z_q = self.phi_convs[k](z_q)
            f_hat = f_hat + z_q
            partial = self.post_quant_conv(bicubic_resize(f_hat, *self.full_size))
            partials.append(partial)
            all_indices.append(indices)
            all_z_flat.append(res_flat.detach())
            commit_loss = commit_loss + commit
        return partials[-1], all_indices, partials, commit_loss, all_z_flat


# ---------- EMA State ----------


@dataclass
class EMAState:
    cluster_sizes: torch.Tensor   # (K,)
    codebook_sums: torch.Tensor   # (K, D)
    codebook: torch.Tensor        # (K, D)

    def to(self, device):
        return EMAState(self.cluster_sizes.to(device), self.codebook_sums.to(device),
                        self.codebook.to(device))

    def state_dict(self) -> dict:
        return {"cluster_sizes": self.cluster_sizes.cpu(),
                "codebook_sums": self.codebook_sums.cpu(), "codebook": self.codebook.cpu()}

    @classmethod
    def from_state_dict(cls, state: dict, device="cpu") -> "EMAState":
        return cls(torch.as_tensor(state["cluster_sizes"]).float().to(device),
                   torch.as_tensor(state["codebook_sums"]).float().to(device),
                   torch.as_tensor(state["codebook"]).float().to(device))


def _unit_normal_rows(shape, generator: torch.Generator, device) -> torch.Tensor:
    x = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    return x / x.norm(dim=1, keepdim=True)


def init_ema_state(codebook_dim: int, codebook_size: int, generator: torch.Generator,
                   device="cpu") -> EMAState:
    codebook = _unit_normal_rows((codebook_size, codebook_dim), generator, device)
    return EMAState(cluster_sizes=torch.ones(codebook_size, device=device),
                    codebook_sums=codebook.clone(), codebook=codebook)


@torch.no_grad()
def ema_update(state: EMAState, all_indices: list[torch.Tensor],
               all_z_flat: list[torch.Tensor], decay: float = 0.99,
               epsilon: float = 1e-5, dead_threshold: float = 0.25, *,
               generator: torch.Generator) -> EMAState:
    """EMA codebook update + Laplace smoothing + dead-code reinit (gust2 semantics).

    all_indices: list of (B, N_k); all_z_flat: list of (B, N_k, D). Called
    OUTSIDE the gradient boundary.
    """
    K, D = state.codebook.shape
    device = state.codebook.device
    new_counts = torch.zeros(K, device=device)
    new_sums = torch.zeros(K, D, device=device)
    total_vectors = 0
    for indices, z_flat in zip(all_indices, all_z_flat):
        flat_idx = indices.reshape(-1).long()
        new_counts += torch.bincount(flat_idx, minlength=K).float()
        new_sums.index_add_(0, flat_idx, z_flat.reshape(-1, D).float())
        total_vectors += flat_idx.numel()

    cluster_sizes = decay * state.cluster_sizes + (1.0 - decay) * new_counts
    codebook_sums = decay * state.codebook_sums + (1.0 - decay) * new_sums

    n = cluster_sizes.sum()
    sizes_smooth = (cluster_sizes + epsilon) / (n + K * epsilon) * n
    codebook = codebook_sums / sizes_smooth[:, None]

    uniform_usage = total_vectors / K
    dead_mask = cluster_sizes < dead_threshold * uniform_usage
    replacements = _unit_normal_rows((K, D), generator, device)
    codebook = torch.where(dead_mask[:, None], replacements, codebook)
    cluster_sizes = torch.where(dead_mask, torch.full_like(cluster_sizes, uniform_usage),
                                cluster_sizes)
    codebook_sums = torch.where(dead_mask[:, None], replacements * uniform_usage, codebook_sums)
    return EMAState(cluster_sizes, codebook_sums, codebook)


# ---------- Losses ----------


def _encode_quantize(encoder, vq, codebook, x_batch, autocast):
    """Encoder in the autocast dtype, VQ always in f32 (gust2's split)."""
    with autocast():
        z_e = encoder(x_batch)
    with torch.autocast(device_type=x_batch.device.type, enabled=False):
        return vq(z_e.float(), codebook)


def vqvae_loss(model: tuple, codebook: torch.Tensor, x_batch: torch.Tensor,
               lambdas, beta: float = 0.25, autocast=None):
    """Compound loss: decoder called once PER SCALE on the partial latents."""
    from contextlib import nullcontext
    autocast = autocast or nullcontext
    encoder, decoder, vq = model
    z_q, all_indices, partials, commit, all_z_flat = _encode_quantize(
        encoder, vq, codebook, x_batch, autocast)
    recon_loss = x_batch.new_zeros((), dtype=torch.float32)
    for k, partial in enumerate(partials):
        with autocast():
            y_k = decoder(partial.to(x_batch.dtype))
        recon_loss = recon_loss + float(lambdas[k]) * F.mse_loss(y_k.float(), x_batch.float())
    commit_loss = commit.mean()
    total = recon_loss + beta * commit_loss
    return total, {"recon_loss": recon_loss.detach(), "commit_loss": commit_loss.detach(),
                   "all_indices": all_indices, "all_z_flat": all_z_flat}


def vqvae_loss_simple(model: tuple, codebook: torch.Tensor, x_batch: torch.Tensor,
                      beta: float = 0.25, autocast=None):
    """Single reconstruction from the final quantised latent (what train.py uses)."""
    from contextlib import nullcontext
    autocast = autocast or nullcontext
    encoder, decoder, vq = model
    z_q, all_indices, _partials, commit, all_z_flat = _encode_quantize(
        encoder, vq, codebook, x_batch, autocast)
    with autocast():
        y = decoder(z_q.to(x_batch.dtype))
    recon_loss = F.mse_loss(y.float(), x_batch.float())
    commit_loss = commit.mean()
    total = recon_loss + beta * commit_loss
    return total, {"recon_loss": recon_loss.detach(), "commit_loss": commit_loss.detach(),
                   "all_indices": all_indices, "all_z_flat": all_z_flat,
                   "reconstruction": y.detach()}
