def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute the Huber Loss between true and predicted values.

    Parameters:
    - y_true: scalar or list of true values
    - y_pred: scalar or list of predicted values
    - delta: threshold parameter (float)

    Returns:
    - Average Huber Loss (float)
    """
    # Ensure inputs are lists for uniform processing
    if not isinstance(y_true, (list, tuple)):
        y_true = [y_true]
    if not isinstance(y_pred, (list, tuple)):
        y_pred = [y_pred]

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    losses = []
    for yt, yp in zip(y_true, y_pred):
        error = yt - yp
        abs_error = abs(error)

        if abs_error <= delta:
            loss = 0.5 * (error ** 2)
        else:
            loss = delta * (abs_error - 0.5 * delta)

        losses.append(loss)

    return sum(losses) / len(losses)
