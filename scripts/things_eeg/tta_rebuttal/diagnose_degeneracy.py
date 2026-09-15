"""Does SAGE's Procrustes capture shared geometry, or memorise the assignment it is given?

On MindEye2/NSD the rotation turned out to encode its input assignment: fitting it to a
SHUFFLED pairing drove 1000-way retrieval to 0.1%, exactly chance, and at every subspace size
where the fit was not degenerate the rotation failed to beat the baseline. That happens when
the map has more freedom than there are correspondences to constrain it.

THINGS-EEG runs at d=512 with n=200 concepts, so the cross-covariance has rank <= 200 for a
512x512 map -- structurally the same over-parameterised regime. This runs the identical
control here, through the repository's own fit_tta_calibration path rather than a
reimplementation, so whatever it reports is what the paper's pipeline does.

Arms:
  plain       no adaptation
  saw+csls    whitening and hubness correction, no rotation
  sage        the published pipeline
  shuffled    the published pipeline with the Sinkhorn assignment replaced by a random
              permutation at every step. A rotation carrying real shared geometry should
              degrade gracefully; one that memorises collapses to chance.

Also reported: the accuracy of the Sinkhorn assignment itself. If SAGE lands AT that number,
the rotation is a lookup table for something the assignment already knew and could have been
read off directly.

Usage:  python diagnose_degeneracy.py --subjects 1 2 3
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from module.util import apply_orthogonal_map, fit_soft_assignment_procrustes, sinkhorn_normalize  # noqa: E402
from shared import (  # noqa: E402
    DEFAULT_SOURCE_RUN_DIR, TTAParams, encode_repetition_blocks, fit_saw_transform,
    load_subject_context, score_features,
)


def top1(scores):
    return float((scores.argmax(axis=1) == np.arange(len(scores))).mean()) * 100


def calibrate(query, gallery, params, shuffle_rng=None):
    """fit_tta_calibration, with an optional shuffled assignment (the null control)."""
    transformed, _ = fit_saw_transform(query, params)
    scores = score_features(transformed, gallery, use_csls=params.use_csls, csls_k=params.csls_k)
    first_assignment_acc = None
    for _ in range(max(1, int(params.soft_procrustes_steps))):
        assignment = sinkhorn_normalize(scores, tau=params.sinkhorn_tau,
                                        num_iters=params.sinkhorn_iters,
                                        col_mass=params.sinkhorn_col_mass)
        if first_assignment_acc is None:
            first_assignment_acc = top1(assignment)
        if shuffle_rng is not None:
            assignment = assignment[:, shuffle_rng.permutation(assignment.shape[1])]
        step = fit_soft_assignment_procrustes(
            transformed, gallery, assignment, power=params.soft_procrustes_power,
            normalize_inputs=params.soft_procrustes_normalize_inputs)
        if step is None:
            break
        transformed = apply_orthogonal_map(transformed, step)
        scores = score_features(transformed, gallery, use_csls=params.use_csls, csls_k=params.csls_k)
    return scores, first_assignment_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_run_dir", default=DEFAULT_SOURCE_RUN_DIR)
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--block_size", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    params = TTAParams()

    print(f"{'sub':>4} {'plain':>7} {'saw+csls':>9} {'sage':>7} {'assign':>7} {'shuffled':>9}")
    rows = []
    for sid in args.subjects:
        rt = SimpleNamespace(device=args.device, batch_size=1024, num_workers=0)
        _, eval_args, dataset, modules = load_subject_context(args.source_run_dir, rt, sid, average=False)
        blocks, gallery, _ = encode_repetition_blocks(
            eval_args, modules, dataset, block_sizes=[args.block_size], seed=1000 + sid)
        query = blocks[0].astype(np.float32)

        plain = top1(score_features(query, gallery, use_csls=False))
        saw, _ = fit_saw_transform(query, params)
        sawcsls = top1(score_features(saw, gallery, use_csls=params.use_csls, csls_k=params.csls_k))
        sage_scores, assign_acc = calibrate(query, gallery, params)
        shuf_scores, _ = calibrate(query, gallery, params, shuffle_rng=np.random.RandomState(0))
        row = (plain, sawcsls, top1(sage_scores), assign_acc, top1(shuf_scores))
        rows.append(row)
        print(f"{sid:>4} " + " ".join(f"{v:>7.1f}" if i != 1 else f"{v:>9.1f}" for i, v in enumerate(row)))

    m = np.mean(rows, axis=0)
    print(f"{'mean':>4} {m[0]:>7.1f} {m[1]:>9.1f} {m[2]:>7.1f} {m[3]:>7.1f} {m[4]:>9.1f}")
    print(f"\nchance for {len(gallery)}-way = {100/len(gallery):.1f}%")
    print("if 'shuffled' sits at chance, the rotation encodes its assignment rather than geometry")


if __name__ == "__main__":
    main()
