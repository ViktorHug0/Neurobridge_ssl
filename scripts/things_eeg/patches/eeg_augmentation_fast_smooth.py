import numpy as np
import random

# Use English instead of Chinese comments

class RandomTimeShift:
    """
    Randomly shift the EEG signal along the time axis (axis=-1).
    max_shift indicates the maximum shift amount (forward or backward), in terms of number of samples.
    """
    def __init__(self, max_shift=5):
        self.max_shift = max_shift

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        # eeg_data shape: (..., time)
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift != 0:
            eeg_data = np.roll(eeg_data, shift, axis=-1)
        return eeg_data


class RandomGaussianNoise:
    """
    Adds random Gaussian noise to the EEG signal.
    std indicates the standard deviation of the noise.
    """
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.std, size=eeg_data.shape)
        return eeg_data + noise


class RandomChannelDropout:
    """
    Randomly drop some channels (set them to zero).
    drop_prob indicates the probability of dropping a channel.
    Assume eeg_data shape: (channel, time) or (channel, time, ...).
    """
    def __init__(self, drop_prob=0.1):
        self.drop_prob = drop_prob

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        # Assuming the first dimension is the channel dimension
        channels = eeg_data.shape[0]
        for ch in range(channels):
            # Randomly decide whether to drop this channel
            if random.random() < self.drop_prob:
                eeg_data[ch] = 0
        return eeg_data


class RandomSmooth:
    """
    A simple smoothing operation, which can be understood as a simple convolution / moving average along the time axis.
    kernel_size indicates the size of the moving average kernel.
    """
    def __init__(self, kernel_size=5, smooth_prob=0.5):
        self.kernel_size = kernel_size
        self.smooth_prob = smooth_prob

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        # Vectorised form of the per-channel/per-timepoint loop this class used to run:
        # same truncated moving average (the window shrinks at the edges), same one
        # np.random.rand() draw per channel in the same order, so the augmentation
        # distribution and the RNG stream are unchanged. The loop cost 74 ms per trial,
        # which dominated training; this is ~0.1 ms. Cumsum runs in float64 because a
        # float32 running sum over the time axis loses the precision that the original
        # short-slice np.mean kept. Verified equal to the loop to <1e-6 (see
        # test_random_smooth_vectorised.py).
        ch, time_len = eeg_data.shape
        keep = np.random.rand(ch) < self.smooth_prob
        smoothed = np.copy(eeg_data)
        if not keep.any():
            return smoothed
        half = self.kernel_size // 2
        t = np.arange(time_len)
        left = np.maximum(0, t - half)
        right = np.minimum(time_len, t + half + 1)
        cumulative = np.concatenate(
            [np.zeros((ch, 1)), np.cumsum(eeg_data, axis=1, dtype=np.float64)], axis=1
        )
        window_mean = (cumulative[:, right] - cumulative[:, left]) / (right - left)
        smoothed[keep] = window_mean[keep].astype(eeg_data.dtype, copy=False)
        return smoothed



class RandomApply:
    """
    Randomly apply a given transform to the EEG data with a probability p.
    """
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return self.transform(eeg_data)
        return eeg_data