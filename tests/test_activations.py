# Programmer: Nicholas Vuletich
# Date: 03-15-2026

import numpy as np

from nn_math.activations import tanh, relu, gelu

def test_tanh():
    x = np.array([0.0])
    assert np.allclose(tanh(x), np.array([0.0]))
    print("Completed tanh test!!!")
    

def test_relu():
    x = np.array([-2, -1, 0, 1, 2])
    expected = np.array([0, 0, 0, 1, 2])
    assert np.allclose(relu(x), expected)
    print("Completed relu test!!!")
    

def test_gelu():
    x = np.array([0.0])
    assert np.allclose(gelu(x), np.array([0.0]))
    print("Completed gelu test!!!")

    