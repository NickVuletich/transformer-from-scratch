# Programmer: Nicholas Vuletich
# Date: 03-15-2026

import numpy as np

from transformer.transformer_block_postnorm import transformer_block_postnorm


def test_transformer_block_postnorm():

    tokens = 4
    embed_dim = 8
    hidden = 16

    X = np.random.randn(tokens, embed_dim)

    Wq = np.random.randn(embed_dim, embed_dim)
    Wk = np.random.randn(embed_dim, embed_dim)
    Wv = np.random.randn(embed_dim, embed_dim)

    W1 = np.random.randn(embed_dim, hidden)
    b1 = np.random.randn(hidden)

    W2 = np.random.randn(hidden, embed_dim)
    b2 = np.random.randn(embed_dim)

    gamma1 = np.ones(embed_dim)
    beta1 = np.zeros(embed_dim)

    gamma2 = np.ones(embed_dim)
    beta2 = np.zeros(embed_dim)

    output = transformer_block_postnorm(X, Wq, Wk, Wv, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2)

    assert output.shape == (tokens, embed_dim)
    assert not np.isnan(output).any()
    assert np.isfinite(output).all()

    print("Completed transformer block postnorm test!!!")