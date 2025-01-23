#example usage of the MLP
import numpy as np
from jcmlp import JITMLP, activ_enum, loss_enum
import time

# Generate sample data
np.random.seed(0)
X = np.random.randn(100, 2)
y = ((X[:, 0] + X[:, 1]) > 0).astype(np.float64).reshape(-1, 1)

# First Call (Before Compilation) ===================

# Create and train the model
start = time.time()
mlp = JITMLP(
        input_size=2, 
        hidden_size=4, 
        output_size=1, 
        activ=activ_enum["sigmoid"], 
        loss=loss_enum["binary_cross_entropy"])
mlp.fit(X, y)

# Make predictions
predictions = mlp.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.4f}")
print(f"#1 ran in {time.time() - start:.4f} seconds")

# Second Call (After Compilation) ===================

# Create and train the model
start_2 = time.time()
mlp = JITMLP(
        input_size=2, 
        hidden_size=4, 
        output_size=1, 
        activ=activ_enum["sigmoid"], 
        loss=loss_enum["binary_cross_entropy"])
mlp.fit(X, y)

# Make predictions
predictions = mlp.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.4f}")
print(f"#2 ran in {time.time() - start_2:.4f} seconds")

