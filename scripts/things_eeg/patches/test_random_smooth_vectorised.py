"""RandomSmooth was vectorised for speed; it must still be the same augmentation.

The original nested loop is reproduced here verbatim as the reference. Both must agree
elementwise AND consume the RNG identically, otherwise the smoothing arm is not the
augmentation the repository defines.
"""
import numpy as np

from module.eeg_augmentation import RandomSmooth


def reference(eeg_data, kernel_size, smooth_prob):
    ch, time_len = eeg_data.shape
    smoothed = np.copy(eeg_data)
    for c in range(ch):
        if np.random.rand() < smooth_prob:
            for t in range(time_len):
                left = max(0, t - kernel_size // 2)
                right = min(time_len, t + kernel_size // 2 + 1)
                smoothed[c, t] = np.mean(eeg_data[c, left:right])
    return smoothed


def test_matches_reference_and_rng():
    for kernel, prob, shape in [(5, 0.3, (63, 250)), (5, 1.0, (8, 17)),
                                (3, 0.5, (10, 40)), (7, 0.0, (4, 9)), (5, 0.5, (6, 3))]:
        x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)

        np.random.seed(123)
        want = reference(x, kernel, prob)
        rng_after_reference = np.random.rand()

        np.random.seed(123)
        got = RandomSmooth(kernel_size=kernel, smooth_prob=prob)(x)
        rng_after_vectorised = np.random.rand()

        assert got.dtype == x.dtype, (got.dtype, x.dtype)
        np.testing.assert_allclose(got, want, atol=1e-6, rtol=1e-5)
        # same number of draws consumed, in the same order
        assert rng_after_reference == rng_after_vectorised, (kernel, prob, shape)


def test_smoothing_actually_happens():
    """A guard against 'fast because it does nothing': smoothing must reduce variance."""
    x = np.random.default_rng(1).standard_normal((32, 200)).astype(np.float32)
    np.random.seed(0)
    out = RandomSmooth(kernel_size=5, smooth_prob=1.0)(x)
    assert out.var() < 0.5 * x.var()
    np.random.seed(0)
    untouched = RandomSmooth(kernel_size=5, smooth_prob=0.0)(x)
    np.testing.assert_array_equal(untouched, x)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
