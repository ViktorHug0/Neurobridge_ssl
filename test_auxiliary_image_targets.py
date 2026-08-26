"""Unit checks for training-only heterogeneous image targets."""
import os
import tempfile

import numpy as np

from module.dataset import _load_image_target_blocks


def test_primary_is_preserved_and_auxiliary_is_normalized():
    with tempfile.TemporaryDirectory() as root:
        directories = [os.path.join(root, name) for name in ('primary', 'aux1', 'aux2')]
        for directory in directories:
            os.makedirs(directory)
        primary = np.array([[[2.0, 4.0]]], dtype=np.float16)
        aux1 = np.array([[[3.0, 4.0, 0.0]]], dtype=np.float32)
        aux2 = np.array([[[0.0, 12.0]]], dtype=np.float32)
        for directory, values in zip(directories, (primary, aux1, aux2)):
            np.save(os.path.join(directory, 'image_train.npy'), values)

        combined, dims = _load_image_target_blocks(directories, 'image_train.npy')
        assert dims == [2, 3, 2]
        np.testing.assert_array_equal(combined[..., :2], primary.astype(np.float32))
        np.testing.assert_allclose(np.linalg.norm(combined[..., 2:5], axis=-1), 1.0)
        np.testing.assert_allclose(np.linalg.norm(combined[..., 5:], axis=-1), 1.0)


if __name__ == '__main__':
    test_primary_is_preserved_and_auxiliary_is_normalized()
    print('auxiliary image target checks passed')
