# Programmer: Nicholas Vuletich
# Date: 03-15-2026

import numpy as np

from nn_math.linear import linear

def test_linear():
    tokens = 6
    input_dim = 5
    output_dim = 4

    X = np.random.randn(tokens, input_dim)
    W = np.random.randn(input_dim, output_dim)
    b = np.random.randn(output_dim)

    Y = linear(X, W, b)

    assert Y.shape == (tokens, output_dim)
    assert np.allclose(Y, X @ W + b)

    print("Completed linear test!!!")
