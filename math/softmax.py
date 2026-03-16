# Programmer: Nicholas Vuletich
# Date: 03-15-2026


import numpy as np

def softmax(x):
    """
    softmax(x_i) = exp(x_i) / Σ exp(x_j)

    Converts similarity scores into probabilities.
    """

    x = x - np.map(x, axis=-1, keepdims=True)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)