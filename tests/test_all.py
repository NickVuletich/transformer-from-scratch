# Programmer: Nicholas Vuletich
# Date: 03-15-2026

from tests.test_activations import test_tanh, test_gelu, test_relu
from tests.test_mlp import test_mlp_output_shape
from tests.test_linear import test_linear
from tests.test_softmax import test_softmax
from tests.test_self_attention import test_self_attention
from tests.test_layernorm import test_layer_norm

if __name__ == "__main__":
    print("Testing activation functions...")
    test_tanh()
    test_gelu()
    test_relu()
    print("\n")

    print("Testing MLP output shape...")
    test_mlp_output_shape()
    print()

    print("Testing linear function...")
    test_linear()
    print()

    print("Testing softmax function...")
    test_softmax()
    print()

    print("Testing self-attention function...")
    test_self_attention()
    print()

    print("Testing layernorm function...")
    test_layer_norm()
    print()

    print("Completed all tests sucessfully!!!")