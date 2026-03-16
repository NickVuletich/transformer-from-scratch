# Programmer: Nicholas Vuletich
# Date: 03-15-2026

import numpy as np

def tanh(x):
    """
    Hyperbolic Tangent

    Math:
    tanh(x) = (e^x - e^-x) / (e^x + e^-x)

    Output range:
    [-1, 1]
    """
    
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu(x):
    """
    ReLU Activation

    Math:
    ReLU(x) = max(0, x)

    Purpose:
    Introduces non-linearity into neural networks.
    """

    return np.maximum(0, x)

def gelu(x):
    """
    GELU Activation

    Approximation used in Transformers.

    Math:
    GELU(x) = 0.5x(1 + tanh( sqrt(2/pi)(x + 0.044715x^3) ))
    """

    return 0.5 * x * (1 + tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))