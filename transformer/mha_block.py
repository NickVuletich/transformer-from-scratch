# Programmer: Nicholas Vuletich
# Date: 03-15-2026

from attention.multi_head_attention import multi_head_attention
from nn_math.layernorm import layer_norm
from transformer.mlp import mlp

def MHA_transformer_block_prenorm(
        X,
        Wq, Wk, Wv,
        W1, b1,
        W2, b2, 
        gamma1, beta1,
        gamma2, beta2
):
    """
    Pre-Norm Transformer Block
    (Used in GPT-style models)

    Z1 = X + Attention(LayerNorm(X))
    Z2 = Z1 + MLP(LayerNorm(Z1))
    """

    norm1 = layer_norm(X, gamma1, beta1)
    attention, weights = multi_head_attention(norm1, Wq, Wk, Wv)
    Z1 = X + attention

    norm2 = layer_norm(Z1, gamma2, beta2)
    mlp_output = mlp(norm2, W1, b1, W2, b2)
    Z2 = Z1 + mlp_output

    return Z2, weights