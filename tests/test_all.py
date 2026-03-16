# Programmer: Nicholas Vuletich
# Date: 03-15-2026

from tests.test_activations import test_tanh, test_gelu, test_relu
from tests.test_mlp import test_mlp_output_shape

if __name__ == "__main__":
    test_tanh()
    test_gelu()
    test_relu()
    test_mlp_output_shape()
    print("Completed all tests sucessfully!!!")