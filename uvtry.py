# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "requests", "fuzzywuzzy","pandas"]
# ///

import numpy as np

def linear_regression(X, y, learning_rate=0.01, epochs=1000):
    # Reshape X if it's 1D
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    
    # Add bias term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Initialize parameters
    theta = np.random.randn(X_b.shape[1])
    
    # Training loop
    for epoch in range(epochs):
        # Compute predictions
        y_pred = np.dot(X_b, theta)
        
        # Compute gradients
        gradients = 2/X_b.shape[0] * X_b.T.dot(y_pred - y)
        
        # Update parameters
        theta -= learning_rate * gradients
        
        # Compute MSE loss (optional)
        if epoch % 100 == 0:
            mse = np.mean((y_pred - y) ** 2)
            print(f'Epoch {epoch}, MSE: {mse:.4f}')
    
    return theta

# Example usage
X = np.array([2000, 3000, 4000, 5000, 6000])  # Input features (e.g., house size)
y = np.array([300000, 450000, 600000, 750000, 900000])  # Target values (e.g., house prices)

# Train the model
theta = linear_regression(X, y, learning_rate=1e-7, epochs=1000)

# Make predictions
X_new = np.array([3500])
X_new_b = np.c_[np.ones((1, 1)), X_new]
prediction = np.dot(X_new_b, theta)
print(f'\nPrediction for {X_new[0]}: {prediction[0]:.2f}')


