# Programmer: Nicholas Vuletich
# Date: 03-15-2026

import numpy as np

from attention.self_attention import self_attention

def test_self_attention():
    tokens = 4
    embed_dim = 8

    X = np.random.randn(tokens, embed_dim)

    Wq = np.random.randn(embed_dim, embed_dim)
    Wk = np.random.randn(embed_dim, embed_dim)
    Wv = np.random.randn(embed_dim, embed_dim)

    output = self_attention(X, Wq, Wk, Wv)

    assert output.shape == (tokens, embed_dim)
    assert not np.isnan(output).any()
    assert np.isfinite(output).all()

    print("Completed self-attention test!!!")
