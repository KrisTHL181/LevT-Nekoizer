"""Tests for the fused FastMuon optimizer."""

from __future__ import annotations

import torch

from levt.fast_muon import FastMuon


def _make_params(shapes):
    params = [torch.randn(*s) for s in shapes]
    for p in params:
        p.requires_grad_(True)
        p.grad = torch.randn_like(p)
    return params


_SHAPES = [(4, 4), (4, 4), (8, 4), (4, 8), (2, 6), (3, 3)]


def test_matches_torch_muon_within_bf16_tolerance():
    shapes = _SHAPES
    a = _make_params(shapes)
    b = _make_params(shapes)
    # identical initial weights and gradients
    for pa, pb in zip(a, b):
        pb.data.copy_(pa.data)
        pb.grad.copy_(pa.grad)

    torch_opt = torch.optim.Muon(
        a, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5
    )
    fast = FastMuon(
        b, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5
    )

    for _ in range(10):
        torch_opt.step()
        fast.step()
        # refresh identical gradients each step
        for pa, pb in zip(a, b):
            g = torch.randn_like(pa)
            pa.grad = g.clone()
            pb.grad = g.clone()

    for pa, pb in zip(a, b):
        # bf16 Newton-Schulz introduces ~1e-3 relative noise; assert far tighter
        # than any meaningful weight change.
        rel = (pa - pb).abs().max() / pa.abs().max()
        assert rel < 1e-2, f"FastMuon diverged from torch Muon (rel={rel:.2e})"


def test_momentum_buffer_matches_torch_muon():
    shapes = [(5, 5), (5, 5)]
    a = _make_params(shapes)
    b = _make_params(shapes)
    for pa, pb in zip(a, b):
        pb.data.copy_(pa.data)
        pb.grad.copy_(pa.grad)

    torch_opt = torch.optim.Muon(a, lr=0.01, weight_decay=0.0, momentum=0.9, ns_steps=2)
    fast = FastMuon(b, lr=0.01, weight_decay=0.0, momentum=0.9, ns_steps=2)
    torch_opt.step()
    fast.step()

    ta = torch_opt.state[a[0]]["momentum_buffer"]
    fb = fast.state[b[0]]["momentum_buffer"]
    assert torch.allclose(ta, fb, atol=1e-5), "momentum buffers diverge after 1 step"


def test_state_dict_round_trip_and_param_group_keys():
    params = _make_params([(4, 4), (4, 4)])
    opt = FastMuon(params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True)
    opt.step()
    sd = opt.state_dict()

    # param group keys mirror torch.optim.Muon's layout.
    assert set(sd["param_groups"][0].keys()) == {
        "params", "lr", "weight_decay", "momentum", "nesterov",
        "ns_coefficients", "eps", "ns_steps", "adjust_lr_fn",
    }
    assert len(sd["state"]) == len(params)
    for pstate in sd["state"].values():
        assert "momentum_buffer" in pstate

    # load into a fresh optimizer bound to the same tensors.
    fresh = FastMuon(
        params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5
    )
    fresh.load_state_dict(sd)
    assert torch.equal(fresh.state[params[0]]["momentum_buffer"], sd["state"][0]["momentum_buffer"])


def test_rejects_non_2d_params():
    p = torch.randn(8, requires_grad=True)
    try:
        FastMuon([p])
    except ValueError:
        return
    raise AssertionError("FastMuon should reject 1-D parameters")
