# Programmer: Nicholas Vuletich
# Date: 03-15-2026

import numpy as np
from nn_math.softmax import softmax

def self_attention(X, Wq, Wk, Wv, mask=None):

    """
    Self Attention

    Math:

    Q = XW_Q
    K = XW_K
    V = XW_V

    scores = (QK^T) / sqrt(d_k)

    weights = softmax(scores)

    output = weightsV

    Where:
    X  = input embeddings
    Wq = query weights
    Wk = key weights
    Wv = value weights

    Purpose:
    Allows each token to attend to every other token and mix their information.
    """
    Q = X @ Wq # (tokens, d_k)
    K = X @ Wk # (tokens, d_k)
    V = X @ Wv # (tokens, d_k)

    d_k = K.shape[-1]

    scores = (Q @ K.T) / np.sqrt(d_k) # (tokens, tokens)

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores) # (tokens, tokens)

    output = weights @ V # (tokens, d_k)

    return output, weights


