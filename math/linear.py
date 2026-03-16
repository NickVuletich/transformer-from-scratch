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

#----------TESTS----------

if __name__ == "__main__":

    tokens = 6
    input_dim = 5
    output_dim = 4

    X = np.random.randn(tokens, input_dim)
    W = np.random.randn(input_dim, output_dim)
    b = np.random.randn(output_dim)

    Y = linear(X, W, b)

    print(f"Input shape: {X.shape}") 
    print(f"Output shape: {Y.shape}")

