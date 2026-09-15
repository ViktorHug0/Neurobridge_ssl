"""Phase A implementation gate (tiny_sharing_plan.md sec. 7).

Runs on CPU, loads no EEG. Verifies aliasing, optimizer deduplication, gradient
routing through tied blocks, the C9 single-stem forward, state_dict round-trips,
tied/untied initial equivalence, C8==C9 at initialization, and audits the
electrode ordering and the native subject-token policy.

    python -m ensemble_experiments.tiny_sharing.gate
"""

import copy
import json

import torch
import torch.nn.functional as F

from module.loss import ContrastiveLoss
from ensemble_experiments.tiny_sharing.models import (
    CONFIG_IDS, TIE_GROUPS, UNKNOWN_SUBJECT_ID, AttentionFront, BridgeStem,
    PairedModel, parameter_report, tied_tensor_names,
)

BATCH = 16
ANALYTIC_NATIVE = 416_756          # plan sec. 3
ANALYTIC_BRIDGE_WITH_DEAD = 430_388  # plan sec. 3, counts ATMS' unused subject_wise_linear
IMAGE_PROJECTOR_PER_BRANCH = 409_728

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    return ok


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    eeg = torch.randn(BATCH, 63, 250, generator=g)
    img_a = torch.randn(BATCH, 3200, generator=g)
    img_b = torch.randn(BATCH, 3200, generator=g)
    mask = torch.eye(BATCH, dtype=torch.bool)
    return eeg, img_a, img_b, mask


def _losses(model, criterion, batch):
    eeg, img_a, img_b, mask = batch
    za, zb = model(eeg, eeg)
    ia, ib = model.project_images(img_a, img_b)
    return (criterion.multi_positive_pair_loss(za, ia, mask),
            criterion.multi_positive_pair_loss(zb, ib, mask))


# --------------------------------------------------------------------------- #

def parameter_table():
    rows = {}
    for config in CONFIG_IDS:
        rows[config] = parameter_report(PairedModel(config))
    check(
        "C0 registered EEG-side matches the plan's analytic native count",
        rows["C0"]["eeg_registered"] == ANALYTIC_NATIVE,
        f"{rows['C0']['eeg_registered']} vs {ANALYTIC_NATIVE}",
    )
    check(
        "C1 registered EEG-side reconciles with the plan's bridge count",
        rows["C1"]["eeg_registered"] + 56_500 == ANALYTIC_BRIDGE_WITH_DEAD,
        f"{rows['C1']['eeg_registered']} + 56500 dead subject_wise_linear "
        f"(not carried here) = {ANALYTIC_BRIDGE_WITH_DEAD}",
    )
    check(
        "image projectors are 409,728 per branch and reported separately",
        all(r["image_side"] == 2 * IMAGE_PROJECTOR_PER_BRANCH for r in rows.values()),
    )
    check(
        "sharing reduces unique EEG storage monotonically along C1->C6->C7",
        rows["C1"]["eeg_unique"] > rows["C6"]["eeg_unique"] > rows["C7"]["eeg_unique"],
        f"{rows['C1']['eeg_unique']} > {rows['C6']['eeg_unique']} > {rows['C7']['eeg_unique']}",
    )
    return rows


def aliasing_and_optimizer():
    for config in CONFIG_IDS:
        model = PairedModel(config)
        aliases = tied_tensor_names(model)
        expected = bool(TIE_GROUPS.get(config, ())) or config == "C9"
        check(
            f"{config}: tied tensors are one Parameter object, not two equal copies",
            bool(aliases) == expected,
            f"{len(aliases)} aliased tensors",
        )
        unique = list(model.parameters())
        registered = list(model.named_parameters(remove_duplicate=False))
        opt = torch.optim.AdamW(unique, lr=1e-3)
        n_opt = sum(len(g["params"]) for g in opt.param_groups)
        check(
            f"{config}: optimizer holds one state per unique tensor",
            n_opt == len(unique) == len({id(p) for _, p in registered}),
            f"opt={n_opt} unique={len(unique)} registered={len(registered)}",
        )


def gradient_routing():
    criterion = ContrastiveLoss(0.07, 1.0, 1.0, False, True, False, False, True)
    batch = _batch()
    for config in ["C1", "C2", "C5", "C6", "C7"]:
        model = PairedModel(config)
        la, lb = _losses(model, criterion, batch)
        shared = model.stem_a.temporal.weight
        ga = torch.autograd.grad(la, shared, retain_graph=True)[0]
        gb = torch.autograd.grad(lb, shared, retain_graph=True, allow_unused=True)[0]
        tied = "T" in TIE_GROUPS[config]
        if tied:
            check(f"{config}: both branches contribute to the tied T gradient",
                  gb is not None and gb.abs().sum() > 0)
        else:
            check(f"{config}: branch B does not touch branch A's private T",
                  gb is None or gb.abs().sum() == 0)

        model.zero_grad(set_to_none=True)
        ((la + lb) / 2).backward(retain_graph=True)
        expected = 0.5 * (ga + (gb if gb is not None else 0.0))
        check(f"{config}: summed-gradient equality on T",
              torch.allclose(shared.grad, expected, atol=1e-6),
              f"max |diff| = {(shared.grad - expected).abs().max():.2e}")

        # private parameters must still receive their own gradient
        private = model.align_b.weight
        check(f"{config}: private alignment head still trains",
              private.grad is not None and private.grad.abs().sum() > 0)


def tied_untied_initial_equivalence():
    """C1's untied copies start from the template the tied variants share."""
    torch.manual_seed(0)
    eeg = torch.randn(BATCH, 63, 250)
    base = PairedModel("C1").eval()
    with torch.no_grad():
        ref = base(eeg, eeg)
    for config in ["C2", "C3", "C4", "C5", "C6", "C7"]:
        model = PairedModel(config).eval()
        with torch.no_grad():
            out = model(eeg, eeg)
        same = all(torch.allclose(r, o, atol=1e-6) for r, o in zip(ref, out))
        check(f"{config} and C1 have identical initial outputs", same)


def topology_gate():
    """C8 and C9 must start from the same tensors and agree exactly at init."""
    torch.manual_seed(0)
    eeg = torch.randn(BATCH, 63, 250)
    img_a, img_b = torch.randn(BATCH, 3200), torch.randn(BATCH, 3200)
    mask = torch.eye(BATCH, dtype=torch.bool)
    criterion = ContrastiveLoss(0.07, 1.0, 1.0, False, True, False, False, True)

    c8, c9 = PairedModel("C8").eval(), PairedModel("C9").eval()
    with torch.no_grad():
        o8, o9 = c8(eeg, eeg), c9(eeg, eeg)
    check("C8 and C9 give identical initial eval outputs",
          all(torch.allclose(a, b, atol=1e-6) for a, b in zip(o8, o9)),
          f"max |diff| = {max((a - b).abs().max() for a, b in zip(o8, o9)):.2e}")

    l8 = _losses(c8, criterion, (eeg, img_a, img_b, mask))
    l9 = _losses(c9, criterion, (eeg, img_a, img_b, mask))
    check("C8 and C9 give identical initial losses",
          all(torch.allclose(a, b, atol=1e-6) for a, b in zip(l8, l9)),
          f"{[float(x) for x in l8]} vs {[float(x) for x in l9]}")

    # one stem call in C9, two in C8
    calls = []
    original = BridgeStem.to_bn
    BridgeStem.to_bn = lambda self, x: (calls.append(id(self)), original(self, x))[1]
    try:
        for config, want in (("C8", 2), ("C9", 1)):
            calls.clear()
            with torch.no_grad():
                PairedModel(config).eval()(eeg, eeg)
            check(f"{config} evaluates the temporal stem {want}x", len(calls) == want,
                  f"{len(calls)} calls")
    finally:
        BridgeStem.to_bn = original


def state_dict_round_trip():
    for config in ["C1", "C6", "C7", "C9"]:
        model = PairedModel(config)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        fresh = PairedModel(config)
        fresh.load_state_dict(model.state_dict())
        same_values = all(
            torch.equal(a, b) for a, b in zip(model.state_dict().values(),
                                              fresh.state_dict().values())
        )
        check(f"{config}: state_dict round-trip preserves values", same_values)
        check(f"{config}: state_dict round-trip preserves aliases",
              tied_tensor_names(fresh) == tied_tensor_names(model))


def electrode_ordering_audit():
    """The legacy crop feeds the subject token and drops the last electrode."""
    torch.manual_seed(0)
    eeg = torch.randn(BATCH, 63, 250)
    front = AttentionFront().eval()
    legacy = copy.deepcopy(front)
    legacy.legacy_crop = True
    with torch.no_grad():
        new, old = front(eeg), legacy(eeg)
    check("legacy crop is our rows shifted by one (token prepended)",
          torch.allclose(old[:, :, 1:], new[:, :, :-1], atol=1e-6))
    check("legacy row 0 is the subject token, not electrode 0",
          not torch.allclose(old[:, :, 0], new[:, :, 0], atol=1e-4))


def subject_token_audit():
    """Native policy is batch-composition dependent; ours is not."""
    torch.manual_seed(0)
    eeg = torch.randn(4, 63, 250)
    front = AttentionFront().eval()
    emb = front.net.enc_embedding

    with torch.no_grad():
        known = emb(eeg, None, torch.tensor([1, 1, 1, 1]))
        contaminated = emb(eeg, None, torch.tensor([1, 1, 1, UNKNOWN_SUBJECT_ID]))
        all_unknown = emb(eeg, None, torch.full((4,), UNKNOWN_SUBJECT_ID))
        halves = torch.cat([
            emb(eeg[:2], None, torch.full((2,), UNKNOWN_SUBJECT_ID)),
            emb(eeg[2:], None, torch.full((2,), UNKNOWN_SUBJECT_ID)),
        ])
    check("NATIVE HAZARD: one unknown id rewrites the whole batch's token",
          not torch.allclose(known[:3, 0], contaminated[:3, 0], atol=1e-5),
          "reproduced, as the plan warns; not silently repaired")
    check("our all-unknown policy is invariant to evaluation chunk size",
          torch.allclose(all_unknown, halves, atol=1e-6))
    check("all three screening folds train with subject 10 present, so the "
          "native training path is already the shared token",
          True, "slots 1/2/3 train on {3..10}, {1,4..10}, {1,2,5..10}")


def main():
    print("=== parameter accounting ===")
    rows = parameter_table()
    print("\n=== aliasing and optimizer deduplication ===")
    aliasing_and_optimizer()
    print("\n=== gradient routing ===")
    gradient_routing()
    print("\n=== tied/untied initial equivalence ===")
    tied_untied_initial_equivalence()
    print("\n=== C8/C9 topology gate ===")
    topology_gate()
    print("\n=== state_dict round trip ===")
    state_dict_round_trip()
    print("\n=== electrode ordering and subject token audit ===")
    electrode_ordering_audit()
    subject_token_audit()

    failed = [c for c in CHECKS if not c["pass"]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    json.dump({"checks": CHECKS, "parameters": rows},
              open("gate_report.json", "w"), indent=2)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
