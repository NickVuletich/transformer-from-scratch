# Programmer: Nicholas Vuletich
# Date: 03-15-2026
# File: layernorm.py

import numpy as np

def layer_norm(x, gamma, beta, esp=1e-5):
    """
    Layer Normalization

    Math:

    Mean:
    μ = (1/n) Σ x_i

    Variance:
    σ² = (1/n) Σ (x_i - μ)²

    Output:
    y = γ * (x - μ) / sqrt(σ² + ε) + β

    Where:
    γ(gamma) = scale parameter
    β(beta) = shift parameter
    ε(epislon) = small constant for numerical stability

    Purpose:
    Normalizes features across the embedding dimension to stabilize training.
    """


    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_norm = (x - mean) / np.sqrt(var + esp)

    return gamma * x_norm + beta

if __name__ == "__main__":

    tokens = 6
    embed_dim = 8

    x = np.random.randn(tokens, embed_dim)

    gamma = np.ones(embed_dim)
    beta = np.ones(embed_dim)

    y = layer_norm(x, gamma, beta)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")