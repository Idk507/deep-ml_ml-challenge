"""
Implement Elastic Net Regression using gradient descent, combining L1 and L2 penalties to handle multicollinearity and encourage sparsity in the feature weights.

Example:
Input:
X = np.array([[0, 0], [1, 1], [2, 2]]); y = np.array([0, 1, 2])
Output:
(array([0.37, 0.37]), 0.25)
Reasoning:
The model learns a nearly perfect linear relationship with regularization controlling weight magnitude. The weights converge around 0.37 with a bias around 0.25.
"""
def elastic_net_gradient_descent(X, y, alpha1=0.1, alpha2=0.1, learning_rate=0.01, max_iter=1000, tol=1e-4):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # Initialize weights
    b = 0  # Initialize bias

    for _ in range(max_iter):
        y_pred = X.dot(w) + b
        error = y_pred - y

        # Compute gradients
        dw = (1 / n_samples) * X.T.dot(error) + alpha1 * np.sign(w) + 2 * alpha2 * w
        db = (1 / n_samples) * np.sum(error)

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

        # Check for convergence
        if np.linalg.norm(dw, ord=1) < tol:
            break

    return w, b
