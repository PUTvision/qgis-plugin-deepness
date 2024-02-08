import numpy as np


class StandardizationParameters:
    mean: float
    std: float

    def __init__(self):
        self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.std = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def set_mean_std(self, mean: np.array, std: np.array):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
