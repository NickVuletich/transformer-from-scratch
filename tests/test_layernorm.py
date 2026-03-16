# Programmer: Nicholas Vuletich
# Date: 03-15-2026

import numpy as np

from nn_math.layernorm import layer_norm

def test_layer_norm():
    tokens = 6
    embed_dim = 8

    x = np.random.randn(tokens, embed_dim)

    gamma = np.ones(embed_dim)
    beta = np.ones(embed_dim)

    y = layer_norm(x, gamma, beta)

    assert y.shape == (tokens, embed_dim)
    assert not np.isnan(y).any()
    assert np.isfinite(y).all()

    print("Completed layernorm test!!!")

