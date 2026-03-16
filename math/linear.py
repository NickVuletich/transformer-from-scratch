# Programmer: Nicholas Vuletich
# Date: 03-15-2026


"""
Linear Layer

Math:
Y = XW + b

Where:
X = (tokens, input_dim)
W = (input_dim, output_dim)
b = (output_dim)
"""

def linear(X, W, b):
    Y = X @ W + b
    return Y