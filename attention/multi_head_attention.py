# Programmer: Nicholas Vuletich
# Date: 03-31-2026
# File: multi_head_attention.py

import numpy as np
from nn_math.softmax import softmax

def multi_head_attention(X, Wq, Wk, Wv, mask=None):
    Q = X @ Wq 
    K = X @ Wk
    V = X @ Wv

    d_k = K.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)

    seq_len = X.shape[0]

    if mask is None:
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9

    scores = scores + mask
    
    weights = softmax(scores)
    output = weights @ V

    return output, weights

