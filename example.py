#example usage of the MLP
import numpy as np
from main import JITMLP
from activations import activ_enum
from loss import loss_enum

# Generate sample data
np.random.seed(0)
X = np.random.randn(100, 2)
y = ((X[:, 0] + X[:, 1]) > 0).astype(np.float64).reshape(-1, 1)

# Create and train the model
mlp = JITMLP(input_size=2, hidden_size=4, output_size=1, activ=activ_enum["sigmoid"], loss=loss_enum["binary_cross_entropy"])
mlp.fit(X, y)

# Make predictions
predictions = mlp.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.4f}")
