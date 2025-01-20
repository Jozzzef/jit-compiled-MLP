#activation functions

import numpy as np
from numba import njit, jit, float64
import numba as nb

@jit(nopython=True)
def sigmoid(x):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-x))

@jit(nopython=True)
def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    sx = sigmoid(x)
    return sx * (1 - sx)

activ_enum = { 
    "sigmoid": [sigmoid, sigmoid_derivative]
}

