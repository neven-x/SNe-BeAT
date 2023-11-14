import numpy as np


def make_design_matrix(data, history=25):
    """
    Create time-lagged design matrix from stimulus intensity vector.

    Args:
        data (1D array): Stimulus intensity at each time point.
        history (number): Number of time lags to use.

    Returns
        X (2D array): GLM design matrix with shape T, d
    """

    # Create version of stimulus vector with zeros before onset
    padded_data = np.concatenate([np.zeros(history - 1), data])

    # Construct a matrix where each row has the d frames of
    # the stimulus proceeding and including timepoint t
    T = len(data)  # Total number of timepoints (number of stimulus frames)
    X = np.zeros((T, history))
    for t in range(T):
        X[t] = padded_data[t:t + history]

    return X
