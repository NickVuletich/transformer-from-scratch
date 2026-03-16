# Programmer: Nicholas Vuletich
# Date: 03-15-2026

import numpy as np
import math as m

def self_attention(X, Wq, Wk, Wv):

    """
    Self attention mechanism
    """ 
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    d_k = K.shape[-1]

    scores = (Q @ K.T) / np.sqrt(d_k)

    weights = m.softmax(scores)

    output = weights @ V

    return output