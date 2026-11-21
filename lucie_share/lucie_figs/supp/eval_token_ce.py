"""Teacher-forced CE per token (token-weighted mean over all 388 positions, the
GUST paper's definition) and per scale, over the whole train split, for every
model-only ladder checkpoint -> token_ce.json. Same pairs and forward pass as
scripts/eval_split_ce.py; only the across-scale average differs."""
import glob, json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lucie_tokens.dynamics import dynamics_loss
from lucie_tokens.runtime import autocast_context, resolve_device, resolve_precision
from lucie_tokens.tokens import TokenPairDataset, collate_token_pairs
from lucie_tokens.training import _unpack_dynamics_batch
from lucie_tokens.workflows import load_dynamics_checkpoint

HERE = Path(__file__).parent
out_path = HERE / "token_ce.json"
results = json.loads(out_path.read_text()) if out_path.exists() else {}
device = resolve_device("auto"); precision = resolve_precision("auto", device)
dataset = TokenPairDataset(HERE / "tokens", "train")
loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_token_pairs, num_workers=2)
for ck in sorted(glob.glob(str(HERE.parent / "inputs/ckpts/sc_r*_s*_epoch50_model.pt"))):
    run = Path(ck).name.removesuffix("_epoch50_model.pt")
    if run in results:
        continue
    model, _ = load_dynamics_checkpoint(ck, device); model.eval()
    sums = None; pairs = 0; weighted = 0.0
    with torch.inference_mode():
        for batch in loader:
            current, target, cond = _unpack_dynamics_batch(batch, device)
            with autocast_context(device, precision):
                logits = model(current, target, cond)
                _, m = dynamics_loss(logits, target)
            per = torch.stack([m[f"scale_{h}x{w}"] for h, w in model.scales]).double() * len(target[0])
            sums = per if sums is None else sums + per
            weighted += float(m['loss']) * len(target[0])
            pairs += len(target[0])
    per_scale = (sums / pairs).tolist()
    tokens = [h * w for h, w in model.scales]
    results[run] = {"ce_per_token": sum(c * n for c, n in zip(per_scale, tokens)) / sum(tokens),
                    "ce_per_scale": dict(zip([f"{h}x{w}" for h, w in model.scales], per_scale)), "pairs": pairs,
                    "ce_inv_sqrt": weighted / pairs}
    out_path.write_text(json.dumps(results, indent=1))
    print(run, round(results[run]["ce_per_token"], 4), flush=True)
    del model; torch.cuda.empty_cache()
