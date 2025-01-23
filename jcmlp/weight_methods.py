import numpy as np
from numba import jit, float64
import numba as nb
import math


@jit(nopython=True)
def init_weights(input_size, hidden_size, output_size):
    """
    initialize weights with random values
    
    w = weight
    b = bias
    index #1 is for the hidden layer
    index #2 is for the output layer
    """
    # select from normal distribution | variance = 1/numNeurons_[layer-1] (Xavier initialization)
    w1 = np.random.randn(input_size, hidden_size) * math.sqrt(1/input_size) 
    b1 = np.zeros(hidden_size) # zero vs random init for bias is unconsequential
    w2 = np.random.randn(hidden_size, output_size) * math.sqrt(1/hidden_size) 
    b2 = np.zeros(output_size)

    return w1, b1, w2, b2

#@jit(nopython=True)
def forward_pass(X, w1, b1, w2, b2, activ):
    """
    forward propagation
   
    z = weighted sum
    a = activation output
    index #1 is for the hidden layer
    index #2 is for the output layer
    """
    # Step A: compute the forward pass through the network
    # Hidden layer
    z1 = np.dot(X, w1) + b1 # weighted sum, pre activation
    a1 = activ(z1) # nueron activation output
    
    # Output layer
    z2 = np.dot(a1, w2) + b2 # weighted sum
    a2 = activ(z2) # nueron activation output
    
    return z1, a1, z2, a2

#@jit(nopython=True, parallel=True)
def backward_pass(X, y, w1, b1, w2, b2, z1, a1, z2, a2, learning_rate, activ_deriv, loss):
    """
    backward propagation

    index #1 is for the hidden layer
    index #2 is for the output layer
    """
    # Step C: Backward pass = gradient computation 
    # Simple mini-batch gradient descent used
    # therefore all gradients are averaged over batch size m for stability.
    m = X.shape[0]
   
    # Output layer gradients
    # our goal equations (following the chain rule):
        # ∂L/∂w2 = ∂z2/∂w2 * (∂a2/∂z2 * ∂L/∂a2) 
        # ∂L/∂b2 = ∂z2/∂b2 * (∂a2/∂z2 * ∂L/∂a2)
            # parentheses used to show how the same block is used twice
            # therefore let ∂L/∂z2 = ∂L/∂a2 * ∂a2/∂z2 for ease of use
    # d_z2 represents ∂L/∂z2 (partial derivative of Loss with respect to z2)
    d_z2 = loss[1](a2, y) * activ_deriv(z2) 
    # represents ∂L/∂w2 | how much each weight in b2 contributed to the error
    # when deriving: w_l * a()_{l-1} + b_l, with respect to w_l it is just a()_{l-1}, and l-1 = 1 here
    d_w2 = np.dot(a1.T, d_z2) / m  
    # represents ∂L/∂b2 | how much each bias in b2 contributed to the error
    # when deriving: w_l * a()_{l-1} + b_l, with respect to b_l it is just 1
    # instead of using the costly dot product, we can just sum since the dot 
        # product would just be multiply all gradients dz by 1 and then sum it, so its the same
    d_b2 = np.sum(d_z2, axis=0) / m 
    
    # Hidden layer gradients, same logic for the above but with a few changes:
    # ∂L/∂z1 = ∂L/∂a1 * ∂a1/∂z1
        # Since a1 affects L through z2, we need to use the chain rule again:
            # ∂L/∂a1 = ∂L/∂z2 * ∂z2/∂a1
            # When we take ∂z2/∂a1, we get w2. This is why we need to multiply dz2 (which is ∂L/∂z2) by w2.T
    d_z1 = np.dot(d_z2, w2.T) * activ_deriv(z1)
    # the previous layer is now the input, hence X
    d_w1 = np.dot(X.T, d_z1) / m
    d_b1 = np.sum(d_z1, axis=0) / m
    
    # Step D: update the weights and biases
    w1 -= learning_rate * d_w1
    b1 -= learning_rate * d_b1
    w2 -= learning_rate * d_w2
    b2 -= learning_rate * d_b2
    
    return w1, b1, w2, b2
