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

    if mask is not None:
        scores = scores + mask
    
    weights = softmax(scores)

    output = weights @ V

    return output, weights

