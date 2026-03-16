# Programmer: Nicholas Vuletich
# Date: 03-15-2026

import numpy as np
from nn_math.activations import gelu

def mlp(X, W1, b1, W2, b2):
    """
    Transformer Feedforward Network (MLP)

    Math:

    H = GELU(XW1 + b1)

    O = HW2 + b2

    Purpose:
    Applies nonlinear transformation after attention.
    """

    H = gelu(X @ W1 + b1)

    O = H @ W2 + b2

    return O