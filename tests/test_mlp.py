# Programmer: Nicholas Vuletich
# Date: 03-15-2026

import numpy as np

from transformer.mlp import mlp

def test_mlp_output_shape():

    tokens = 4
    embed_dim = 8
    hidden_dim = 18

    X = np.random.randn(tokens, embed_dim)

    W1 = np.random.randn(embed_dim, hidden_dim)
    b1 = np.random.randn(hidden_dim)

    W2 = np.random.randn(hidden_dim, embed_dim)
    b2 = np.random.randn(embed_dim)

    output = mlp(X, W1, b1, W2, b2)

    assert output.shape == (tokens, embed_dim)

    print("Completed mlp output shape test!!!")


