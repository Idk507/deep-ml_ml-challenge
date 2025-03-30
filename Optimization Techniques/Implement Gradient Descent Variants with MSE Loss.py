"""
Implement Gradient Descent Variants with MSE Loss

In this problem, you need to implement a single function that can perform three variants of gradient descentâStochastic Gradient Descent (SGD), Batch Gradient Descent, and Mini-Batch Gradient Descentâusing Mean Squared Error (MSE) as the loss function. The function will take an additional parameter to specify which variant to use.

Example:
Input:
import numpy as np

# Sample data
X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
y = np.array([2, 3, 4, 5])

# Parameters
learning_rate = 0.01
n_iterations = 1000
batch_size = 2

# Initialize weights
weights = np.zeros(X.shape[1])

# Test Batch Gradient Descent
final_weights = gradient_descent(X, y, weights, learning_rate, n_iterations, method='batch')
# Test Stochastic Gradient Descent
final_weights = gradient_descent(X, y, weights, learning_rate, n_iterations, method='stochastic')
# Test Mini-Batch Gradient Descent
final_weights = gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size, method='mini_batch')
Output:
[float,float]
[float, float]
[float, float]

"""
import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
    m = len(y)
    
    for _ in range(n_iterations):
        if method == 'batch':
            # Calculate the gradient using all data points
            predictions = X.dot(weights)
            errors = predictions - y
            gradient = 2 * X.T.dot(errors) / m
            weights = weights - learning_rate * gradient
        
        elif method == 'stochastic':
            # Update weights for each data point individually
            for i in range(m):
                prediction = X[i].dot(weights)
                error = prediction - y[i]
                gradient = 2 * X[i].T.dot(error)
                weights = weights - learning_rate * gradient
        
        elif method == 'mini_batch':
            # Update weights using sequential batches of data points without shuffling
            for i in range(0, m, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                predictions = X_batch.dot(weights)
                errors = predictions - y_batch
                gradient = 2 * X_batch.T.dot(errors) / batch_size
                weights = weights - learning_rate * gradient
                
    return weights


