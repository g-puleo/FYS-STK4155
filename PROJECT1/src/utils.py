import numpy as np

def MSE(z, ztilde):
    """
    Calculates the mean squared error, the average squared difference between the estimated values and the actual value

    Args:
        z (ndarray) : Target data
        ziltde (ndarray) : Predicted z-value

    Returns:
        MSE (float) : S
    """
    z = z.ravel()
    ztilde = ztilde.ravel()
    return (1/len(z))*np.sum((z-ztilde)**2)

def R2(z,ztilde):
    """
    Calculates the R2 score.

    Args:
        z (ndarray) : Target data
        ziltde (ndarray) : Predicted z-value

    Returns:
        R2_score (float) : Score, max as 1.
    """
    z = z.ravel()
    ztilde = ztilde.ravel()
    return 1- (MSE(z,ztilde))/np.var(z)

def FrankeFunction(x,y):
    """
    Calculates the z-value for the given frankiefunction with input x and y

    Args:
        x (ndarray) : All x datapoints
        y (ndarray) : All y datapoints

    Return:
        z (ndarray) : Calculated z-values
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def singleBootstrap(design_matrix, target):
    """
    Resamples the dataset with replacement and returns one dataset.

    Args:
        design_matrix (ndarray) : Data to resample
        target (ndarray) : Corresponding target values

    Returns:
        design_matrix[ bootstrap_indices,:] (ndarray) : Bootstrapped dataset
        target[bootstrap_indices] (ndarray) : Bootstrapped target data
    """
    nrows, ncols = design_matrix.shape
    indices = np.arange(nrows)
    bootstrap_indices = np.random.choice(indices, size=nrows)
    return (design_matrix[bootstrap_indices, :], target[bootstrap_indices])
