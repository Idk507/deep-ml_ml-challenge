import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Implements Gaussian Naive Bayes classifier.
    
    Args:
        X_train: Training features (shape: N_train x D)
        y_train: Training labels (shape: N_train)
        X_test: Test features (shape: N_test x D)
    
    Returns:
        Predicted class labels for X_test (shape: N_test)
    """
    eps = 1e-9  # small value for numerical stability
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_features = X_train.shape[1]

    # --- Training Phase ---
    priors = {}
    means = {}
    variances = {}

    for c in classes:
        X_c = X_train[y_train == c]
        priors[c] = X_c.shape[0] / X_train.shape[0]
        means[c] = np.mean(X_c, axis=0)
        variances[c] = np.var(X_c, axis=0) + eps

    # --- Prediction Phase ---
    predictions = []
    for x in X_test:
        log_posteriors = []
        for c in classes:
            # log prior
            log_prior = np.log(priors[c])
            # log likelihoods for each feature
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variances[c]))
            log_likelihood -= 0.5 * np.sum(((x - means[c]) ** 2) / variances[c])
            # total log posterior
            log_posterior = log_prior + log_likelihood
            log_posteriors.append(log_posterior)
        # choose class with max posterior
        predictions.append(classes[np.argmax(log_posteriors)])

    return np.array(predictions)
