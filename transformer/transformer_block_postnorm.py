# Programmer: Nicholas Vuletich
# Date: 03-15-2026

from attention.self_attention import self_attention
from nn_math.layernorm import layer_norm
from transformer.mlp import mlp

def transformer_block_postnorm(
        X,
        Wq, Wk, Wv,
        W1, b1,
        W2, b2, 
        gamma1, beta1,
        gamma2, beta2
):
    """
    Post-Norm Transformer Block
    (Original 2017 Transformer)

    Z1 = LayerNorm(X + Attention(X))
    Z2 = LayerNorm(Z1 + MLP(Z1))
    """

    attention, weights = self_attention(X, Wq, Wk, Wv)
    Z1 = layer_norm(X + attention, gamma1, beta1)

    mlp_output = mlp(X, W1, b1, W2, b2)
    Z2= layer_norm(X + mlp_output, gamma2, beta2)

    return Z2, weights