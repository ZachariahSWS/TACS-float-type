"""10-year references for supp_best_cells, scored exactly like the rollouts
(lucie_figures.RolloutScore: pooled pixel histograms, lat-weighted zonal KE):
  refs10/truth_{hist,spectrum}.npz  ERA5, Zenodo frames [start, start+frames)
  recon10.score.json                tokenizer reconstruction of the same frames
                                    (train tokens; train = Zenodo frames 0..14599)
`--check` instead scores the val year (Zenodo 14600..16059) and compares it
with the existing one-year refs, to validate the pipeline."""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "lucie/lucie_tokens/scripts")
from lucie_figures import RolloutScore
from lucie_tokens import diagnostics as D
from lucie_tokens.era5_npz import legendre_gauss_area_weights
from lucie_tokens.tokens import load_tokenizer_checkpoint

p = argparse.ArgumentParser(); p.add_argument("--check", action="store_true"); a = p.parse_args()
HERE = Path("lucie/local/lucie_figs/supp"); I = Path("lucie/local/lucie_figs/inputs"); Z = Path("lucie/data/zenodo_15164648/preprocessed")
CH = ["temperature", "humidity", "u_wind", "v_wind", "surface_pressure"]  # Zenodo vars 0-4
SCALES = [(1, 2), (2, 4), (3, 6), (6, 12), (12, 24)]
dev = "cuda"
with np.load(I / "refs/normalize_mean.npz") as m, np.load(I / "refs/normalize_std.npz") as s:
    mu = np.array([float(m[c]) for c in CH])[:, None, None]; sd = np.array([float(s[c]) for c in CH])[:, None, None]
zen = np.load(Z / "data_inp.npy", mmap_mode="r")
raw_mu = np.load(Z / "raw_means.npy").ravel()[:5, None, None]; raw_sd = np.load(Z / "raw_stds.npy").ravel()[:5, None, None]
start, frames = (14600, 1460) if a.check else (0, 14600)
wind = (slice(2, 3), slice(3, 4))

def scorer():
    return RolloutScore(1, len(CH), legendre_gauss_area_weights(48), wind, SCALES, 4096, dev)

truth = scorer()
dummy = tuple(torch.zeros((1, h, w), dtype=torch.long, device=dev) for h, w in SCALES)
for i in range(start, start + frames, 365):
    block = np.asarray(zen[i:min(i + 365, start + frames), :5]) * raw_sd + raw_mu  # physical
    x = torch.as_tensor((block - mu) / sd, dtype=torch.float32, device=dev)  # y1 normalization
    for f in range(len(x)):
        truth.update(x[f:f + 1], dummy)
counts = truth.counts.sum(0).cpu().numpy(); ke = (truth.ke_sum / truth.frames).cpu().numpy()[:-1]
if a.check:
    with np.load(I / "refs/truth_hist.npz") as b: ref_counts = b["counts"]
    with np.load(I / "refs/truth_spectrum.npz") as b: ref_ke = b["kinetic_energy"]
    print("hist: total", counts.sum(), ref_counts.sum(), " EMD per channel vs refs",
          [round(D.histogram_emd(counts[c], ref_counts[c]), 5) for c in range(5)])
    print("KE max rel diff", float(np.max(np.abs(ke - ref_ke) / ref_ke)))
    sys.exit()
out = HERE / "refs10"; out.mkdir(exist_ok=True)
np.savez(out / "truth_hist.npz", edges=D.HIST_EDGES, counts=counts)
np.savez(out / "truth_spectrum.npz", kinetic_energy=ke)
for f in ("normalize_mean.npz", "normalize_std.npz"):
    shutil.copy(I / "refs" / f, out / f)

tok, _ = load_tokenizer_checkpoint(str(I / "tokenizer_y1_12x24_best.pt"), dev)
tokens, _, _ = D.load_token_file(str(I / "train_source.h5"))
assert len(tokens) == frames, len(tokens)
recon = scorer()
with torch.inference_mode():
    for i in range(0, frames, 128):
        grids = [D.initial_pyramid(tokens[j], SCALES, dev) for j in range(i, min(i + 128, frames))]
        batch = [torch.cat([g[s] for g in grids]) for s in range(len(SCALES))]
        x = tok.decode_indices(tuple(batch), final_only=True)[0].float()
        for f in range(len(x)):
            recon.update(x[f:f + 1], tuple(b[f:f + 1] for b in batch))
res = recon.result(out, CH); res.pop("member_mean_pixel_emd")
(HERE / "recon10.score.json").write_text(json.dumps(res))
print("10-yr recon EMD", round(res["ensemble_mean_pixel_emd"], 4), "KE relL2", round(res["ke_rel_l2_vs_data"], 4))
