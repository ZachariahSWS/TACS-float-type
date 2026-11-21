"""Tokenizer reconstruction of the whole val year scored like a rollout
(pooled pixel histograms + KE spectrum + token entropy) -> recon.score.json."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "lucie/lucie_tokens/scripts")
from lucie_figures import RolloutScore
from lucie_tokens import diagnostics as D
from lucie_tokens.era5_npz import legendre_gauss_area_weights
from lucie_tokens.tokens import load_tokenizer_checkpoint
I = Path("lucie/local/lucie_figs/inputs")
dev = "cuda"
tok, _ = load_tokenizer_checkpoint(str(I / "tokenizer_y1_12x24_best.pt"), dev)
tokens, _, _ = D.load_token_file(str(I / "val_source.h5"))
channels = "temperature,humidity,u_wind,v_wind,surface_pressure".split(",")
wind = tuple(slice(channels.index(n), channels.index(n) + 1) for n in ("u_wind", "v_wind"))
SCALES = [(1, 2), (2, 4), (3, 6), (6, 12), (12, 24)]  # 388 tokens
n = len(tokens)
score = None
with torch.inference_mode():
    for a in range(0, n, 64):
        grids = [D.initial_pyramid(tokens[i], SCALES, dev) for i in range(a, min(a + 64, n))]
        batch = [torch.cat([g[s] for g in grids]) for s in range(len(SCALES))]
        frames = tok.decode_indices(tuple(batch), final_only=True)[0].float()
        if score is None:
            score = RolloutScore(1, frames.shape[1], legendre_gauss_area_weights(48), wind,
                                 SCALES, 4096, dev)
        for f in range(len(frames)):  # members=1 histogram layout
            score.update(frames[f:f + 1], tuple(b[f:f + 1] for b in batch))
res = score.result(I / "refs", channels)
res.pop("member_mean_pixel_emd")
Path("lucie/local/lucie_figs/supp/recon.score.json").write_text(json.dumps(res))
print(n, "frames; recon EMD", res["ensemble_mean_pixel_emd"], "KE relL2", res["ke_rel_l2_vs_data"])
