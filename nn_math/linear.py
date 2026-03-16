# Programmer: Nicholas Vuletich
# Date: 03-15-2026

import numpy as np

def linear(X, W, b):
    """
    Linear Layer

    Math:
    Y = XW + b

    Where:
    X = input matrix          (tokens, input_dim)
    W = weight matrix         (input_dim, output_dim)
    b = bias vector           (output_dim)

    Output:
    Y = transformed matrix    (tokens, output_dim)

    Purpose:
    Transforms input vectors into a new feature space using a learned linear transformation.
    """

    return X @ W + b

