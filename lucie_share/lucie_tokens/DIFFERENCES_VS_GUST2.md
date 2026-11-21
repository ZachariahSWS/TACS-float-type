# LUCIE vs gust2 — why val gets worse after 5–20 epochs

Diagnosis of the "loss improves, then degrades after ~5–20 epochs" symptom, by
comparing LUCIE's training recipe against the gust2 reference
(`gust2/train.py` for the VQ-VAE, `gust2/train_nsp.py` for next-scale
prediction).

**Short version:** the model architectures are essentially the same. The
difference is the *optimizer schedule*. gust2 warms up and then cosine-decays
the learning rate to ~1% of peak over the whole run. LUCIE holds the learning
rate **fixed** for every step of every epoch. A fixed, relatively high LR
(2e-4) is exactly what produces "improves for a while, then gets worse" — the
model converges into a basin, then the too-large step size bounces it back out
and val loss climbs. Your mentor is right that it shouldn't happen; the fix is a
scheduler.

---

## 1. No learning-rate schedule — PRIMARY CAUSE

**gust2** (`train_nsp.py:687-702`, and identically in `train.py:250-265`):

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=args.lr,          # e.g. 1e-4
    warmup_steps=warmup_steps,   # min(1000, total_steps // 10)
    decay_steps=total_steps,     # epochs * steps_per_epoch
    end_value=args.lr * 0.01,    # decays to 1% of peak
)
optimizer = optax.chain(
    optax.clip_by_global_norm(args.grad_clip),
    optax.adamw(schedule, weight_decay=args.weight_decay),
)
```

- LR warms up **0 → peak** over the first ~1000 steps, then **cosine-decays to
  1% of peak** across the whole run.

**LUCIE** (`runtime.py:61-72`, used in `training.py:259` and `training.py:637`):

```python
def make_adamw(parameters, lr, weight_decay, device):
    return torch.optim.AdamW(parameters, lr=float(lr),
                             weight_decay=float(weight_decay),
                             fused=device.type == "cuda")
```

- No warmup. No decay. The LR is **constant at `lr` (2e-4 in every config) for
  the entire run**. The only `warmup` token in the whole `src/` tree is the
  Torch profiler schedule — there is no LR scheduler object anywhere, and
  nothing ever touches `optimizer.param_groups[...]['lr']`.

**Why this produces the exact symptom.** With a fixed LR, AdamW keeps taking
peak-sized steps after the model has already reached a good basin. Early on the
gradient signal dominates and loss drops; once near a minimum, the steps are too
large to settle, so the parameters oscillate/drift and **val loss turns back
up** — typically after a handful to a few dozen epochs, precisely the "5–20"
you're seeing. gust2 never sees this because by the time the model is near
convergence its LR is a small fraction of peak.

> Note: LUCIE keeps `best.pt` (lowest val), so the *artifact* isn't lost — but
> the *training curve* degrading is real and means you're wasting the second
> half of every run and never reaching the lower minimum a decayed LR would
> find.

### Fix

Add a per-optimizer-step warmup+cosine scheduler to both `train_tokenizer` and
`train_dynamics` in `training.py`. Sketch:

```python
from torch.optim.lr_scheduler import LambdaLR

total_steps = int(optimization["epochs"]) * steps_per_epoch // accumulation
warmup_steps = min(1000, max(1, total_steps // 10))

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return 0.01 + 0.99 * cosine          # floor at 1% of peak, like gust2

scheduler = LambdaLR(optimizer, lr_lambda)
```

Then call `scheduler.step()` immediately after each `_optimizer_step(...)`
(the two call sites per training function), and log the current LR into
`metrics` so the schedule is visible. On resume, restore
`scheduler.load_state_dict(checkpoint["scheduler"])` and save it into `payload`
alongside optimizer/scaler.

`steps_per_epoch` is `len(train_loader)` (or `pipeline.batches_per_epoch` in the
noised path) — both are already computed in `train_dynamics`.

---

## 2. Per-scale loss weighting defaults to the mode the code itself warns against

`dynamics_loss` in `dynamics.py:274-328` supports three weightings and defaults
to `inv_sqrt`. Its own docstring flags the problem (`dynamics.py:280-288`):

> "on the 12x24 ladder it gives the 2-token coarsest scale 48% of the loss and
> ~1700x the per-token weight of the finest… that concentrates gradient on the
> most memorizable part of the problem; inv_log (VAR/gust2's choice) is the
> flatter option."

gust2 uses `1/log(token_count + 1)` (`train_nsp.py:655-663`) — the flatter
option.

Several dynamics configs **do not set `scale_loss_weighting`**, so they silently
fall back to `inv_sqrt`:

```
configs/bridges_dyn_1h_cn.toml
configs/bridges_dyn_1h.toml
configs/bridges_dyn_cond.toml
configs/bridges_dyn_noise.toml
configs/bridges_dyn_res16_cn.toml
configs/bridges_dyn_res24_cn.toml
configs/era5_t30_res24x48_cond.toml
```

Concentrating almost half the gradient on a 2-token, highly-memorizable coarse
scale is a textbook overfitting setup: the model nails those tokens on train,
and val CE on them rises once memorization kicks in — a **secondary contributor
to the same U-shaped curve**.

**However (per Zach, 2026-08-31): `inv_log` has tested WORSE than `inv_sqrt`
in this project's runs, so do NOT blindly copy gust2 here.** Note the
arithmetic: on the 12x24 ladder `inv_log` still gives the 2-token coarsest
scale ~45% of the loss vs `inv_sqrt`'s ~48-52% — it is barely flatter, so it
buys little de-concentration while changing the objective. The option that
actually de-emphasizes the coarse 2-token scale is `uniform` (equal weight per
scale → ~20% each on a 5-scale ladder), already supported in `dynamics_loss`.
If coarse-scale memorization needs addressing after the LR fix, `uniform` is
the arm to try — not `inv_log`.

---

## 3. The actual gust2 training recipe (from their run scripts, not defaults)

Every gust2 production script (`scripts/bridges/sweep_nsp.sh`,
`sweep_nsp_chain.sh`, `sweep_nsp_sc917.sh`, `sweep_robust_scaling.sh`,
`scripts/derecho/train_robust.sh`) uses the SAME recipe, and it differs from
LUCIE's far more than the argparse defaults suggested:

| Knob | gust2 (actual) | LUCIE era5 dynamics | Delta |
|------|-------|-------|-------|
| epochs | **400** | 25 | 16x longer |
| peak LR | **1e-4** | 2e-4 | half |
| weight decay | **1e-4** | 0.01 | 1/100th |
| cosine span | 400 ep (at ep25 LR ≈ 99% peak) | 25 ep (at ep25 LR = 1%) | decay stretched past the overfit window |
| input noise | **always on** — substitution 0.1 (newer) or context-drop 0.25 (older) | none | the standing regularizer |
| grad clip | 1.0 | 1.0 | = |
| batch | 16–64 | 16 | ≈ |
| dropout | 0.0 | 0.0 | = |

Their recipe is tuned for a LONG, GENTLE run: low LR, negligible WD, noise as
the regularizer, and a schedule that spends hundreds of epochs in the
high-LR regime. LUCIE's 25-epoch cosine compressed the entire decay into the
overfitting window — better than constant LR (section: Verification
experiment) but structurally still a short-run recipe. Their derecho "micro"
run (3L/384 — nearly LUCIE's size) uses the identical 400/1e-4/1e-4/sub-0.1
recipe, so this is recipe, not model scale.

### gust2-recipe arms: outcome (2026-08-31)

**v1** (`local_era5t30_dyn_gust2hp.toml`: lr 1e-4, wd 1e-4, sub 0.1 on t0
only, cosine/100): degraded from ep10 (4.63 → 6.08 by ep25), cancelled.
Audit found the replication gap: gust2 noises t0 AND the teacher-forced t1
context independently (`train_nsp.py:267-311`); LUCIE's pipeline noised only
t0, leaving every scale ≥2x4 free to over-trust a pristine context.

**v2** (`local_era5t30_dyn_gust2hp_ctx.toml`: same + independent t1-context
noise via `token_noise.substituted_context` → `dynamics.forward
target_context`, labels clean; 25 epochs): the healthiest CE curve of any
arm — min 4.606 at ep8, final 5.024, rise from min only +0.42 (vs sched
+0.94, v1 +1.45), best coarse accuracy (peak 0.545). The v1-vs-v2 pair
isolates context noising as the difference between degrading and holding.

**BUT rollout diagnostics rank the arms differently** (2000-step, T=1.0):
sched ep24 (cosine 2e-4, no noise) remains decisively best on the metrics
the lab prioritizes — KE relL2 vs data 0.299 vs ctx's 0.380-0.406, hi-m
retention 0.189 vs 0.130, mean EMD 0.082 vs 0.122 (ctx is worse than even
the constant-LR baseline on EMD). ctx ep8 does have the most data-like
coarse-chain temporal stats measured to date (decorr lag 4 = data, change
0.407, dwell 2.46) — context noise buys planetary-scale dynamical honesty
at the price of defensively smooth predictions. Consistent with the prior
finding that uniform substitution is a GLOBAL corruption while rollout
errors are local. Also gust2's `train_nsp.py` computes NO validation CE at
all — their 400-epoch recipe was only ever validated by rollouts.

**Operating point: sched ep24** (`dynamics_era5t30_seed0_sched/latest.pt`).
Open arms: extend the sched recipe to 50-100 epochs (rollout-selected);
neighbour-restricted or float-noise context corruption (reuses the
`target_context` hook); lower ctx rates (0.02-0.05).

Tokenizer-side (also from their scripts): VQ-VAE grad-clip 0.5 (LUCIE 1.0)
and best-sweep codebook `ema_decay 0.90` (LUCIE 0.99) — worth a look only
after the dynamics recipe is settled.

## 4. Things that are NOT the problem (verified same)

- **Architecture.** LUCIE's `nn.py` uses RMSNorm, QK-norm (`q_norm`/`k_norm`),
  bias-free projections, and bilinear-upsampled expansion heads with learned
  target-position encodings — matching gust2's `nsp_model.py` design.
- **Mixed precision.** Both keep f32 master weights and run forward/backward in
  bf16.
- **Teacher-forced next-scale objective + attention mask** (t0↔t0 full, t1→t0
  full, t1→t1 causal-in-scale) are equivalent.
- **No model-weight EMA in either.** gust2's NSP saves raw weights too, so
  that's not a difference.

---

## Verification experiment (2026-08-31, local 5070)

Controlled A/B on the era5 T30 12x24-ladder dynamics model
(`dynamics_era5t30_seed0` replica: 384/8, batch 16, 25 epochs, lr 2e-4,
wd 0.01, seed 0, inv_sqrt) — the ONLY change is the new cosine schedule
(`configs/local_era5t30_dyn_sched.toml`). Result
(`lucie/local/out/sched_compare/val_curves.png`):

| | constant LR (baseline) | cosine schedule |
|---|---|---|
| best val CE | 4.891 (ep7) | **4.381 (ep5), -10.4%** |
| final val CE (ep24) | 6.461 | **5.317, -17.7%** |
| rise from min | +1.570 | +0.936 |
| val acc 1x2 @ best | 0.339 | **0.465** |
| val acc 1x2 @ ep24 | 0.261 | **0.397** |

The schedule arm is better at **24/25 matched epochs** and at every scale's
accuracy, and its late curve flattens (last 5 epochs: +0.07 total) while the
baseline is still degrading. The residual post-minimum rise is genuine
overfitting (coarse-scale sample starvation — few independent planetary
states), consistent with the project's prior findings; the constant LR was
amplifying it roughly 2x, and the schedule also leaves `latest.pt` (the
checkpoint that historically rolls out best) far healthier. Milestone
checkpoints epoch{5,10,15,20,25}.pt were kept for rollout comparison.

### Rollout diagnostics (2000-step val rollouts, T=1.0, same truth refs)

Same 4-way comparison, scored the way this project trusts (spectra + entropy
+ coarse-chain, not CE):

| | base ep7 | base ep24 | sched ep5 | sched ep24 |
|---|---|---|---|---|
| KE relL2 vs data | 0.373 | 0.367 | 0.370 | **0.299** |
| KE relL2 vs recon (dynamics-only) | 0.360 | 0.354 | 0.358 | **0.285** |
| hi-m retention vs data | 0.131 | 0.142 | 0.152 | **0.189** |
| mean pixel EMD (σ) | 0.133 | 0.113 | 0.112 | **0.082** |
| coarse-chain change rate (data: 0.304) | — | 0.611 | 0.460 | **0.512** |
| coarse-chain dwell (data: 3.35) | — | 1.65 | 2.19 | **1.95** |
| late/early std (stability) | 1.004 | 0.984 | 1.010 | 0.973 |

**sched ep24 wins every distributional metric**: dynamics-only KE error −20%,
mean pixel EMD −28% (temperature EMD halved: 0.063 vs 0.122; every channel
better), hi-m retention +33%, and low-m KE damping improved from ~0.5× to
~0.6× of truth. Collapse ruled out: per-scale entropy deltas are positive
(inflated, the known T=1/sample-size regime) — not the drained-entropy
frozen-pattern signature — and the coarse chain moves *toward* the data's
temporal statistics (baseline ep24 flickers at 2× the data's change rate;
sched ep24 cuts that to 1.7×). ep24 ≻ ep5 again confirms train-long,
select-late under the schedule. Outputs:
`lucie/local/out/era5_sched_{best,latest}/`.

## Recommended order of action

1. **Add the warmup+cosine LR schedule (section 1).** This alone should remove
   the degradation. Do the tokenizer and the dynamics trainer.
2. Re-run one short config (e.g. `era5_t30_res24x48.toml`) and confirm the val
   curve is now monotonic-ish past epoch 20.
3. If coarse-scale memorization persists after the LR fix, try a `uniform`
   weighting arm (section 2) — NOT `inv_log`, which tested worse than
   `inv_sqrt` here.
4. Leave the minor knobs (section 3) until after you've confirmed the schedule
   fixed the curve, so you're changing one thing at a time.
