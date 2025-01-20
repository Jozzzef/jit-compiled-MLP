# entry point
import sys
import numpy as np
from numba import jit, float64
import numba as nb
from activations import *
from loss import *
import weight_methods as wm

class JITMLP:
    def __init__(self, input_size, hidden_size, output_size, activ, loss, learning_rate=0.01):
        """Initialize the JIT compiled MLP"""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.w1, self.b1, self.w2, self.b2 = wm.init_weights(input_size, 
                                                             hidden_size, 
                                                             output_size)
        self.activ = activ
        self.loss = loss

    def fit(self, X, y, epochs=100000):
        """Train the neural network"""
        X = X.astype(np.float64)
        y = y.astype(np.float64)
       
        print(self.activ)

        for epoch in range(epochs):
            # Forward pass
            z1, a1, z2, a2 = wm.forward_pass(
                X, 
                self.w1, 
                self.b1, 
                self.w2, 
                self.b2,
                self.activ[0]
            )
            
            # Backward pass and update weights
            self.w1, self.b1, self.w2, self.b2 = wm.backward_pass(
                X, 
                y, 
                self.w1, 
                self.b1, 
                self.w2, 
                self.b2,
                z1, 
                a1, 
                z2, 
                a2, 
                self.learning_rate,
                self.activ[1], 
                self.loss
            )
            
            if epoch % 1000 == 0:
                loss_output = self.loss[0](a2, y)
                print(f'Epoch {epoch}, Loss: {loss_output:.4f}')
    
    def predict(self, X):
        """Make predictions"""
        X = X.astype(np.float64)
        _, _, _, predictions = wm.forward_pass(X, self.w1, self.b1, self.w2, self.b2, self.activ[0])
        return (predictions > 0.5).astype(np.float64)

