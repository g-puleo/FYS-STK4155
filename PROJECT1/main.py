


#One function to run examples and plot all the different stuff.
import plotting_functions as plotfnc
from franke_fit import *
import numpy as np
import matplotlib.pyplot as plt
import utils

def generate_reults():
    methods = [OLS, Ridge, Lasso]
    np.random.seed(3463223)
    np.random.seed(133)
    # Make data.
    Nx_ = 16
    Ny_ = 16
    maxdeg = 12
    #generate x,y data from uniform distribution
    x__ = np.random.rand(Nx_, 1)
    y__ = np.random.rand(Ny_, 1)
    x_, y_ = np.meshgrid(x__,y__)
    z_ = (utils.FrankeFunction(x_, y_) + 0.1*np.random.randn(Nx_,Ny_)).reshape(-1,1)

    #methods = [OLS]
    MSE_list_train = []
    MSE_list_test = []
    titles = []
    for met in methods:
        titles.append(met.__name__)
        lamb_val = 0.1
        #MSE plot example
        degrees_list, MSE_train_list, MSE_test_list, bias, variance, _, _, _ = Solver(x_, y_, z_, Nx_, Ny_, met, lamb = lamb_val, useBootstrap=False, useCrossval=False, maxdegree = maxdeg)
        MSE_list_train.append(MSE_train_list)
        MSE_list_test.append(MSE_test_list)

    plotfnc.MSE_plot(degrees_list, MSE_list_train, MSE_list_test, titles_ = titles, savefig = True)
    plt.show()

    """
    #Bias var plot example
    degrees_list, MSE_train_list, MSE_test_list, bias, variance, _, _, _ = Solver(x_, y_, z_, Nx_, Ny_, OLS, useBootstrap=True, useCrossval=False, maxdegree = maxdeg)
    plotfnc.bias_var_plot(degrees_list, bias, variance, MSE_test_list, title="Custom1", savefig = True)
    plt.show()
    """

    """
    #Betamatrix plot
    deg_toplot = 6
    nr_of_betas = 6
    degrees_list, MSE_train_list, MSE_test_list, bias, variance, beta_matrix, _, _ = Solver(x_, y_, z_, Nx_, Ny_, OLS, useBootstrap=False, useCrossval=False, lamb=0.0001, maxdegree = maxdeg)
    plotfnc.betaval_plot(degrees_list, beta_matrix, nr_of_betas, maxdeg = deg_toplot, title = f"Betavalues_{nr_of_betas}", savefig = True)
    plt.show()
    """

    #Attempt to show frankie func which dont look too good
    #plotfnc.function_plot(x_, y_, z__)
    """
    #Gridsearch
    lambda_vals = np.logspace(-6, 0, 7)
    mindeg = 3
    maxdeg = 10
    method = Ridge
    MSE_2d = np.zeros(shape=(maxdeg+1-mindeg ,len(lambda_vals)))

    method = OLS
    useBoot = False
    useCross = False
    #Fill array with MSE values. x-axis lambda, y-axis degree
    for i in range(len(lambda_vals)):
        degrees_list, MSE_train_list, MSE_test_list, _, _, _, _ , _ =  Solver(x_, y_, z_, Nx_, Ny_, Ridge, useBootstrap=useBoot, useCrossval=useCross, lamb=lambda_vals[i], mindegree = mindeg, maxdegree = maxdeg)
        for j in range(maxdeg-mindeg+1):
            MSE_2d[j,i] = MSE_test_list[mindeg+j] #fix indexing cause of length

    plotfnc.gridsearch_plot(MSE_2d, lambda_vals, mindeg, maxdeg, title="boot_ridge", savefig = True)
    plt.show()
    """

    #z__ = utils.FrankeFunction(x__, y__)
    #frankie everything
    #setupdata
    #run solver
    #plot MSE, bias etc.
    return

def main():
    #Run to generate all the plots using the frankie function.
    generate_reults()

    #Plot whatever you want.

    #gen x, y ,z data
    # Call on solver with parameters
    # Call on what to plot.

    #Terrain Data example.
    #terrain plotting.
    return

if __name__ == "__main__":
    main()
