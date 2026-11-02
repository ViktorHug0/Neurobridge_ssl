import numpy as np

# Use English instead of Chinese comments

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


class RandomSmooth:
    """
    A simple smoothing operation, which can be understood as a simple convolution / moving average along the time axis.
    kernel_size indicates the size of the moving average kernel.
    """
    def __init__(self, kernel_size=5, smooth_prob=0.5):
        self.kernel_size = kernel_size
        self.smooth_prob = smooth_prob

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        ch, time_len = eeg_data.shape
        smoothed = np.copy(eeg_data)
        # Ensure kernel size is odd
        for c in range(ch):
            if np.random.rand() < self.smooth_prob:
                for t in range(time_len):
                    left = max(0, t - self.kernel_size // 2)
                    right = min(time_len, t + self.kernel_size // 2 + 1)
                    smoothed[c, t] = np.mean(eeg_data[c, left:right])
        return smoothed
