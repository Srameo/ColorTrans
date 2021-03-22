import numpy as np


def gamma_fix(img: np.ndarray, gamma: float = 1):
    return np.power(img / 255.0, gamma) * 255.0
