"""Small numerical checks for the SPD covariance diagnostic."""

import numpy as np

from synthetic_subject_experiments.analyze_covariance_manifold import (
    _regularized_second_moment,
    _spd_log,
    _symmetric_vector,
)


def test_covariance_and_log_are_symmetric_positive_finite():
    rng = np.random.default_rng(4)
    eeg = rng.normal(size=(30, 6, 40)).astype(np.float32)
    covariance = _regularized_second_moment(eeg, shrink=0.05, chunk_size=7)
    assert np.allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() > 0
    tangent = _spd_log(covariance / np.trace(covariance))
    assert np.allclose(tangent, tangent.T)
    assert np.isfinite(tangent).all()


def test_symmetric_vector_preserves_frobenius_norm():
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(7, 7))
    matrix = 0.5 * (matrix + matrix.T)
    assert np.allclose(np.linalg.norm(_symmetric_vector(matrix)), np.linalg.norm(matrix))


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'ok  {name}')
    print('all covariance-manifold self-checks passed')
