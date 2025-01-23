# entry point
import numpy as np
from .weight_methods import * 

class JITMLP:
    def __init__(self, input_size, hidden_size, output_size, activ, loss, learning_rate=0.01):
        """Initialize the JIT compiled MLP
        
        input_size := how many neurons are in the input layer
        hidden_size := how many neurons are in the hidden layer
        output_size :=  how many neurons are in the output layer
        activ := the activation function, please choose one from the activ_enum
        loss := the loss function, please choose one from the loss_enum

        available methods for this class:
            fit() := train the MLP with your data
            predict() := see how well the MLP predicts with your given input 
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.w1, self.b1, self.w2, self.b2 = init_weights(input_size, 
                                                             hidden_size, 
                                                             output_size)
        self.activ = activ
        self.loss = loss

    def fit(self, X, y, epochs=100000, verbosity=10000):
        """Train the neural network

        X := your training vectors all in one matrix
        y := your vector of labels corresponding to X
        epochs := the amount of training loops
        verbosity := the amount of lines in console showing the training progress; in interval [1, epochs]
        """
        X = X.astype(np.float64)
        y = y.astype(np.float64)
       
        for epoch in range(epochs):
            # Forward pass
            z1, a1, z2, a2 = forward_pass(
                X, 
                self.w1, 
                self.b1, 
                self.w2, 
                self.b2,
                self.activ[0]
            )
            
            # Backward pass and update weights
            self.w1, self.b1, self.w2, self.b2 = backward_pass(
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
            
            if epoch % verbosity == 0:
                loss_output = self.loss[0](a2, y)
                print(f'Epoch {epoch}, Loss: {loss_output:.4f}')
    
    def predict(self, X):
        """Make predictions with given input X"""
        X = X.astype(np.float64)
        _, _, _, predictions = forward_pass(X, self.w1, 
                                               self.b1, 
                                               self.w2, 
                                               self.b2, 
                                               self.activ[0])
        return (predictions > 0.5).astype(np.float64)

