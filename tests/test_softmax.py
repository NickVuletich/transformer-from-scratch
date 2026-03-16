# Programmer: Nicholas Vuletich
# Date: 03-15-2026

import numpy as np

from nn_math.softmax import softmax


def test_softmax():

    x = np.array([1.0, 2.0, 3.0])

    probs = softmax(x)

    assert np.isclose(np.sum(probs), 1.0)
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)

    assert probs.shape == x.shape

    print("Completed softmax test!!!")
