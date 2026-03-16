# Programmer: Nicholas Vuletich
# Date: 03-15-2026


import numpy as np

def softmax(x):
    """
    Softmax Function

    Math:
    softmax(x_i) = exp(x_i) / Σ exp(x_j)

    Stable version used in practice:
    softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))

    Purpose:
    Converts raw scores into probabilities.

    Properties:
    - All outputs are between 0 and 1
    - The outputs sum to 1
    """

    x = x - np.max(x, axis=-1, keepdims=True)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
