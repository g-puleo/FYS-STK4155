
import numpy as np



def MSE(z, ztilde):
    z = z.ravel()
    ztilde = ztilde.ravel()
    return (1/len(z))*np.sum((z-ztilde)**2)

def R2(z,ztilde):
    return 1- (MSE(z,ztilde))/np.var(z)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def singleBootstrap( design_matrix, target ):
    '''inputs:
        design_matrix (numpy.ndarray)
        target (numpy.array)
        '''
    nrows, ncols = design_matrix.shape
    indices = np.arange(nrows)
    bootstrap_indices = np.random.choice(indices, size=nrows)

    return (design_matrix[ bootstrap_indices, : ], target[ bootstrap_indices])
