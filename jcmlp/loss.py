import numpy as np
from numba import jit, float64
import numba as nb

@jit(nopython=True)
def mse(y_hat, y):
    """Mean Squared Error"""
    return np.mean((y_hat - y) ** 2.0)

@jit(nopython=True)
def mse_derivative(y_hat, y):
    """Partial Derivative of the Mean Squared Error with respect to y_hat"""
    return 2.0 * (y_hat - y) / y_hat.shape[0] # 2(y_hat - y)/n

@jit(nopython=True)
def binary_cross_entropy(y_hat, y):
   """Binary Cross Entropy"""
   return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

@jit(nopython=True)
def binary_cross_entropy_derivative(y_hat, y):
   """Derivative of Binary Cross Entropy with respect to y_hat"""
   return (y_hat - y) / (y_hat * (1 - y_hat)) / y_hat.shape[0]

loss_enum = { 
    "mse": [mse, mse_derivative],
    "binary_cross_entropy": [binary_cross_entropy, binary_cross_entropy_derivative]
}

