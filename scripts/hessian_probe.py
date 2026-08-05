#!/usr/bin/env python3
"""Hessian spectrum probe for a LevT checkpoint.

Estimates the Hessian extremes (lambda_max / lambda_min) and curvature of the
loss at a given checkpoint using Hessian-vector products via double
backpropagation:

  - power iteration on H        -> largest eigenvalue  lambda_max
  - power iteration on -H       -> most negative eig.  lambda_min
  - Lanczos (full reorth.)      -> mini spectrum (top/bottom k)
  - Hutchinson trace estimator  -> tr(H), tr(H^2) -> mean curvature, participation ratio,
                                   plus a per-module curvature breakdown

Loss is assembled exactly as the validation path does (deterministic oracles:
PolicyConfig alpha=0.0 [model-filled deletion roll-in], beta=1.0 [true
insertion roll-in]; same label smoothing), on a small fixed subset of the
validation data so the forward graph fits on GPU for double-backprop.

Usage (run from the repo root so validation_data resolves):
  PYTHONPATH=. python scripts/hessian_probe.py --checkpoint checkpoints/step_00006480.pt \
      --batch-size 8 --max-len 512 --power-iters 50 --trace-samples 60
"""

import argparse
import time

import torch

# ---------------------------------------------------------------------------
# HVP machinery (operate on "flat" vectors so power iteration / Lanczos are
# simple vector arithmetic over the full parameter space).
# ---------------------------------------------------------------------------


def flatten(vecs):
    return torch.cat([v.reshape(-1) for v in vecs])


def unflatten(flat, refs):
    out = []
    i = 0
    for r in refs:
        n = r.numel()
        out.append(flat[i : i + n].reshape(r.shape))
        i += n
    return out


def make_hvp(loss, params):
    """Return op(v_flat) = flatten(H @ v). The loss graph is retained between
    calls (create_graph + retain_graph on the first-order grads)."""
    grads = torch.autograd.grad(
        loss, params, create_graph=True, retain_graph=True, materialize_grads=True,
    )

    def hvp(v_flat):
        vecs = unflatten(v_flat, params)
        dot = sum((g * v).sum() for g, v in zip(grads, vecs))
        g2 = torch.autograd.grad(
            dot, params, retain_graph=True, materialize_grads=True,
        )
        return flatten(g2)

    return hvp


def rayleigh_quotient(v, hv):
    return (v * hv).sum()  # v unit-norm => v^T H v / v^T v


def power_iteration(op, n, iters, seed=None):
    """Power iteration for the largest-eigenvalue eigenvector of operator `op`.
    Returns (rayleigh, trajectory, final_unit_vector)."""
    g = torch.Generator(device="cuda").manual_seed(seed if seed is not None else 1234)
    v = torch.randn(n, device="cuda", dtype=torch.float32, generator=g)
    v.div_(v.norm())
    lam = 0.0
    traj = []
    for _ in range(iters):
        ov = op(v)
        lam = rayleigh_quotient(v, ov)
        traj.append(lam.item())
        v = ov.div_(ov.norm())
    return lam, traj, v


def lanczos(hvp, n, k, seed=None):
    """Lanczos with full reorthogonalization. Returns tridiagonal T (k x k)."""
    g = torch.Generator(device="cuda").manual_seed(seed if seed is not None else 4321)
    V = torch.zeros(k, n, device="cuda", dtype=torch.float32)
    alpha = torch.zeros(k, device="cuda", dtype=torch.float32)
    beta = torch.zeros(k - 1, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32, generator=g)
    w.div_(w.norm())
    V[0] = w
    for j in range(k):
        q = hvp(w)
        alpha[j] = (w * q).sum()
        q.add_(V[j], alpha=-alpha[j])
        if j > 0:
            q.add_(V[j - 1], alpha=-beta[j - 1])
        for i in range(j + 1):  # full reorthogonalization
            q.add_(V[i], alpha=-(V[i] * q).sum())
        if j < k - 1:
            b = q.norm()
            beta[j] = b
            if b < 1e-12:
                break
            w = q.div_(b)
            V[j + 1] = w
    T = torch.zeros(k, k, device="cuda", dtype=torch.float32)
    T.diagonal().copy_(alpha)
    T.diagonal(1).copy_(beta[: k - 1])
    T.diagonal(-1).copy_(beta[: k - 1])
    return T.cpu()


def hutchinson(hvp, n, samples, refs, seed=None):
    """Hutchinson trace estimators: tr(H), tr(H^2) and per-parameter tr(z_i Hz_i)."""
    g = torch.Generator(device="cuda").manual_seed(seed if seed is not None else 777)
    tr = 0.0
    tr_h2 = 0.0
    per_param = torch.zeros(n, device="cuda", dtype=torch.float32)
    for _ in range(samples):
        z = torch.randn(n, device="cuda", dtype=torch.float32, generator=g)
        hz = hvp(z)
        tr += (z * hz).sum().item()
        tr_h2 += (hz * hz).sum().item()
        per_param.add_(z * hz)
    return tr / samples, tr_h2 / samples, per_param / samples


def gpu_mem_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e6


# ---------------------------------------------------------------------------
# Model + loss assembly (repo-specific).
# ---------------------------------------------------------------------------


def build_model(ckpt, device):
    from levt.config import LevTConfig
    from levt.model import LevTModel

    cfg = LevTConfig.from_dict(ckpt["model_config"])
    model = LevTModel(cfg)
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt["model"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"load_state_dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device).float().eval()
    return model, cfg


def truncate_row(row, max_len):
    """Cap row lengths for the probe while keeping BOS-start / EOS-end."""
    out = dict(row)
    out["src"] = row["src"][:max_len]
    for name in ("target", "initial"):
        seq = row[name]
        out[name] = seq if len(seq) <= max_len else seq[: max_len - 1] + [seq[-1]]
    return out


def build_loss_callable(device, args, model, model_cfg):
    from levt.config import PolicyConfig
    from levt.data import JsonlDataset, LevTCollator
    from levt.trainer import DualPolicyTrainer

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tcfg = ckpt["train_config"]
    valid_path = tcfg.get("validation_data") or args.validation_data
    full_src = tcfg.get("max_source_length", 1024)
    full_tgt = tcfg.get("max_target_length", 1024)

    dataset = JsonlDataset(
        valid_path, model_cfg,
        max_source_length=full_src,
        max_target_length=full_tgt,
        allow_interior_boundaries=tcfg.get("packed", False),
    )
    collator = LevTCollator(
        model_cfg,
        max_source_length=args.max_len,
        max_target_length=args.max_len,
        allow_interior_boundaries=dataset.packed,
    )
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(dataset), generator=g)[: args.batch_size].tolist()
    rows = [truncate_row(dataset[i], args.max_len) for i in idx]
    batch = collator(rows)
    lengths = [(len(r["src"]), len(r["target"])) for r in rows]
    print(f"[probe] {valid_path}: {len(dataset)} rows; sampled {len(rows)} "
          f"(src,tgt) lengths {lengths}")

    smoothing = tcfg.get("label_smoothing", 0.1)
    # alpha=0.0 -> deletion roll-in is always the (deterministic, no_grad)
    # model-filled sequence, so the deletion head contributes to the loss.
    # beta=1.0 -> insertion roll-in is always the true oracle (no random del).
    trainer = DualPolicyTrainer(
        model, model_cfg,
        policy_config=PolicyConfig(
            alpha=0.0, beta=1.0, random_delete_prob=0.0, label_smoothing=smoothing,
        ),
        oracle_batch_size=0,
    )

    def loss_fn():
        prepared = trainer.prepare_batch(batch)
        sums, counts = trainer.loss_sums_and_counts(prepared)
        return trainer.normalized_loss(sums, counts)

    return loss_fn


def component_name(name):
    """Coarse grouping for per-module curvature reporting."""
    s = name.lower()
    if "shared_embedding" in s:
        return "embedding"
    if "encoder_input_projection" in s:
        return "enc_input_proj"
    if "decoder_input_projection" in s:
        return "dec_input_proj"
    if "deletion_head" in s:
        return "del_head"
    if "placeholder_head" in s:
        return "plh_head"
    if "self_attn" in s:
        return "attention_self"
    if "cross_attn" in s:
        return "attention_cross"
    if "q_norm" in s or "k_norm" in s:
        return "qk_norm"
    if "norm" in s:
        return "norm"
    if "ffn" in s:
        return "ffn"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--validation-data", default=None,
                    help="override the checkpoint's validation_data path")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--power-iters", type=int, default=50)
    ap.add_argument("--lanczos-k", type=int, default=20)
    ap.add_argument("--trace-samples", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={dev}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"[probe] checkpoint global_step = {ckpt.get('global_step')}")
    model, model_cfg = build_model(ckpt, dev)
    loss_fn = build_loss_callable(dev, args, model, model_cfg)

    params = list(model.parameters())
    names = [nm for nm, _ in model.named_parameters()]
    n = sum(p.numel() for p in params)
    print(f"[probe] params = {n/1e6:.1f}M  (GPU mem {gpu_mem_mb():.0f} MB)")

    loss = loss_fn()
    print(f"[probe] probe loss = {loss.item():.4f}  (GPU mem {gpu_mem_mb():.0f} MB)")

    hvp = make_hvp(loss, params)
    t0 = time.time()

    # --- lambda_max ---
    lam_max, traj, vmax = power_iteration(hvp, n, args.power_iters, seed=args.seed)
    step = max(1, len(traj) // 8)
    print(f"[probe] power(max) traj: " + " ".join(f"{x:.4f}" for x in traj[::step]))
    print(f"[probe] lambda_max = {lam_max:.4f}  (t={time.time()-t0:.1f}s, mem {gpu_mem_mb():.0f} MB)")

    # --- lambda_min via shifted power iteration on (sigma*I - H), sigma = lambda_max.
    # (Plain power iteration on -H is dominated by the lambda_max eigenvector
    # whenever |lambda_min| < lambda_max, so it cannot isolate the negative tail.)
    t1 = time.time()
    sigma = lam_max
    lam_shift, traj_s, vmin = power_iteration(
        lambda v: sigma * v - hvp(v), n, args.power_iters, seed=args.seed + 7,
    )
    lam_min = sigma - lam_shift
    step = max(1, len(traj_s) // 8)
    print(f"[probe] shift(lambda_min) traj: " + " ".join(f"{x:.4f}" for x in traj_s[::step]))
    cos = float((vmax * vmin).sum().abs())
    print(f"[probe] lambda_min = {lam_min:.4f}  |<vmax,vmin>| = {cos:.4f}  (t={time.time()-t1:.1f}s)")

    # --- Lanczos mini-spectrum ---
    t2 = time.time()
    T = lanczos(hvp, n, args.lanczos_k, seed=args.seed + 13)
    evals = torch.linalg.eigvalsh(T)
    print(f"[probe] Lanczos top5 : {['%.4f' % x for x in evals[-5:].flip(0).tolist()]}")
    print(f"[probe] Lanczos bot5 : {['%.4f' % x for x in evals[:5].tolist()]}")
    print(f"[probe] Lanczos lambda_max = {evals[-1].item():.4f}  lambda_min = {evals[0].item():.4f}  "
          f"(t={time.time()-t2:.1f}s)")

    # --- Hutchinson trace ---
    t3 = time.time()
    tr, tr_h2, per_param = hutchinson(hvp, n, args.trace_samples, params, seed=args.seed + 29)
    mean_lam = tr / n
    particip = (tr * tr) / tr_h2 if tr_h2 > 0 else float("nan")
    print(f"[probe] tr(H) = {tr:.3e}  tr(H^2) = {tr_h2:.3e}  (t={time.time()-t3:.1f}s)")
    print(f"[probe] mean curvature tr/N = {mean_lam:.3e}")
    print(f"[probe] participation ratio tr^2/tr(H^2) = {particip:.1f}")

    # per-module curvature breakdown (per_param is flat, aligned with `params`)
    groups: dict[str, float] = {}
    group_params: dict[str, int] = {}
    i = 0
    for nm, p in zip(names, params):
        k = component_name(nm)
        groups[k] = groups.get(k, 0.0) + float(per_param[i : i + p.numel()].sum())
        group_params[k] = group_params.get(k, 0) + p.numel()
        i += p.numel()
    print("[probe] per-module tr(H) (raw contribution; per-param mean curvature):")
    for k in sorted(groups, key=lambda kv: -abs(groups[kv])):
        print(f"  {k:20s} raw {groups[k]:+.3e}   per-param {groups[k]/group_params[k]:+.3e}")

    # --- summary ---
    # lambda_max: power iteration and Lanczos agree.
    # lambda_min: the negative tail is an extremely low-mass direction in a
    # dense-near-zero spectrum, so random-start power iteration (plain or
    # shifted) cannot reach it in a few dozen iterations; the Lanczos bottom
    # Ritz value is the reliable estimate (stable across k).
    lam_pos = lam_max
    lam_neg = evals[0].item()
    cond = lam_pos / abs(lam_neg) if abs(lam_neg) > 1e-12 else float("inf")
    print("\n===== SUMMARY =====")
    print(f"lambda_max  = {lam_pos:+.4f}  (power iter == Lanczos: {evals[-1].item():.4f})")
    print(f"lambda_min  = {lam_neg:+.4f}  (Lanczos bottom; shifted-power estimate {lam_min:+.4f} not converged)")
    print(f"condition # = {cond:.3f}  (lambda_max / |lambda_min|)")
    print(f"trace       = {tr:.3e}")
    print(f"mean curv   = {mean_lam:.3e}")
    print(f"participation ratio = {particip:.1f} of N={n/1e6:.1f}M")
    print(f"peak GPU mem = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    # Fused SDPA backends (flash / mem-efficient / cudnn) do not implement
    # higher-order autograd, which HVP needs. Force the math backend for the
    # whole run (module-level torch.backends.cuda.*_sdp_enabled flags are
    # no-ops on torch 2.12; the context manager is authoritative).
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        main()
