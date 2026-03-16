# Programmer: Nicholas Vuletich
# Date: 03-15-2026
# File: layernorm.py

import numpy as np

def layer_norm(x, gamma, beta, esp=1e-5):
    """
    Layer normalization
    """

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_norm = (x - mean) / np.sqrt(var = esp)

    return gamma * x_norm + beta