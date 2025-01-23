# jit-compiled-MLP
* a simple Multi Layer Perceptron which uses jit to compile the code, speeding up run time
* performance increase up to 41.74% in testing

### How To Use
* see example.py for the most intuitive understanding.
* below see a simplified view of the main objects to be used
```python
from jcmlp import JITMLP, activ_enum, loss_enum

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

# this enumeration of activation functions expose all which are available
activ_enum = { 
    "sigmoid": [sigmoid, sigmoid_derivative],
    ...
}

# this enumeration of loss functions expose all which are available 
loss_enum = { 
    "mse": [mse, mse_derivative],
    ...
}
```

### Project Roadmap
* add regularization
* easier support for nesting MLPs
* additional activation and loss functions 
