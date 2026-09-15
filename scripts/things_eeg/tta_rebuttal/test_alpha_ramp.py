"""Check for the coverage ramp: alpha is set from t and K alone, no labels."""
from types import SimpleNamespace

import numpy as np

from run_streaming_refit import arm_alphas


def test_coverage_ramp():
    args = SimpleNamespace(alphas=[0.5], alpha_ramp=(0.8, 1.5))
    names = dict(arm_alphas(args, t=100, n_gallery=100))
    assert names["alpha0.5"] == 0.5
    assert np.isclose(names["ramp"], 0.8 * 1.0 / (1.0 + 1.5))       # u = 1
    assert np.isclose(dict(arm_alphas(args, 0, 100))["ramp"], 0.0)   # empty buffer, no rotation
    # monotone in the buffer size, and capped
    strengths = [dict(arm_alphas(args, t, 100))["ramp"] for t in (10, 100, 1000, 100_000)]
    assert strengths == sorted(strengths) and strengths[-1] < 0.8
    # CONST = 0 degenerates to a constant CAP, which is what the runtime check relied on
    flat = SimpleNamespace(alphas=None, alpha_ramp=(0.5, 0.0))
    assert np.isclose(dict(arm_alphas(flat, 37, 100))["ramp"], 0.5)


if __name__ == "__main__":
    test_coverage_ramp()
    print("ok  test_coverage_ramp")
