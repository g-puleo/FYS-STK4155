


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
    #np.random.seed(12)
    # Make data.
    Nx_ = 16
    Ny_ = 16
    maxdeg = 12
    #generate x,y data from uniform distribution
    x__ = np.random.rand(Nx_, 1)
    y__ = np.random.rand(Ny_, 1)
    x_, y_ = np.meshgrid(x__,y__)
    z_ = (utils.FrankeFunction(x_, y_) + 0.1*np.random.randn(Nx_,Ny_)).reshape(-1,1)

    """
    #1 MSE AND R2 SCORES AS FUNCTION OF THE POLYNOMIAL DEGREE, OLS
    list_train = []
    test = []
    degrees_list, MSE_train_list, MSE_test_list, _, _, _, R2_train_list, R2_test_list, _ \
    = Solver(x_, y_, z_, Nx_, Ny_, OLS, lamb = 0, useBootstrap=False, useCrossval=False, maxdegree = maxdeg)
    plotfnc.MSE_R2_plot(degrees_list, MSE_train_list, MSE_test_list, R2_train_list, R2_test_list, savefig = True)
    """

    """
    # (2) BETA PARAMETERS, AS FUNCTION OF POLYDEG, OLS
    #Betamatrix plot
    deg_toplot = 6
    nr_of_betas = 6
    degrees_list, MSE_train_list, MSE_test_list, bias, variance, beta_matrix, _, _ = Solver(x_, y_, z_, Nx_, Ny_, OLS, useBootstrap=False, useCrossval=False, lamb=0.0001, maxdegree = maxdeg)
    plotfnc.betaval_plot(degrees_list, beta_matrix, nr_of_betas, maxdeg = deg_toplot, title = f"Betavalues_{nr_of_betas}", savefig = True)
    plt.show()
    """

    """
    #(3) Bias-variance tradeoff, AS FUNCTION OF POLYDEG USING ONLY BOOTSTRAP, and only OLS
    maxdeg_ = 10
    degrees_list, MSE_train_list, MSE_test_list, bias, variance, beta_matrix, R2_train_list, R2_test_list, z_pred \
    = Solver(x_, y_, z_, Nx_, Ny_, OLS, lamb = 0, useBootstrap=True, useCrossval=False, maxdegree = maxdeg_)
    plotfnc.bias_var_plot(degrees_list, bias, variance, MSE_test_list,  savefig = True)
    """

    """
    #(4)COMPARISON BETWEEN ESTIMATES OF MSE IN CROSSVAL AND BOOTSTRAP, OLS. 2 plots to reuse plotting func
    #Bootstrap value goes a bit crazy for higher complexity which used to a problem which i thought was fixed
    maxdeg = 8
    MSE_list_train = []
    MSE_list_test = []
    titles = ["MSE Bootstrap", "MSE Crossval"]

    degrees_list, MSE_tr_boot, MSE_te_boot, _, _, _, _, _, _ \
    = Solver(x_, y_, z_, Nx_, Ny_, OLS, lamb = 0, useBootstrap=True, useCrossval=False, maxdegree = maxdeg)
    MSE_list_train.append(MSE_tr_boot)
    MSE_list_test.append(MSE_te_boot)

    degrees_list, MSE_tr_cross, MSE_te_cross, _, _, _, _, _, _ \
    = Solver(x_, y_, z_, Nx_, Ny_, OLS, lamb = 0, useBootstrap=False, useCrossval=True, maxdegree = maxdeg)
    MSE_list_train.append(MSE_tr_cross)
    MSE_list_test.append(MSE_te_cross)

    plotfnc.MSE_plot(degrees_list, MSE_list_train, MSE_list_test, titles_ = titles, savename = "bootcross", savefig = True)
    #plt.show()
    """

    """
    #(5)(E-F: CROSSVAL), (5) [2 figures, one ridge, one lasso]
    # MSE COLORPLOT IN RIDGE AND LASSO (gridsearch)
    maxdeg = 13
    crossval = True
    lambda_vals = np.logspace(-6, 0, 7)
    mindeg = 3
    maxdeg = 10
    method = Ridge
    MSE_2d = np.zeros(shape=(maxdeg+1-mindeg ,len(lambda_vals)))
    #Fill array with MSE values. x-axis lambda, y-axis degree
    for i in range(len(lambda_vals)):
        degrees_list, MSE_train_list, MSE_test_list, _, _, _, _, _, _ = \
        Solver(x_, y_, z_, Nx_, Ny_, Ridge, useBootstrap=False, useCrossval=True, lamb=lambda_vals[i], mindegree = mindeg, maxdegree = maxdeg)
        for j in range(maxdeg-mindeg+1):
            MSE_2d[j,i] = MSE_test_list[j] #fix indexing cause of length

    plotfnc.gridsearch_plot(MSE_2d, lambda_vals, mindeg, maxdeg, savename="Ridge_grid", savefig = True)
    #plt.show()

    lambda_vals = np.logspace(-12, -3, 10)
    method = Lasso
    MSE_2d = np.zeros(shape=(maxdeg+1-mindeg ,len(lambda_vals)))
    #Fill array with MSE values. x-axis lambda, y-axis degree
    for i in range(len(lambda_vals)):
        degrees_list, MSE_train_list, MSE_test_list, _, _, _, _, _, _ = \
        Solver(x_, y_, z_, Nx_, Ny_, method, useBootstrap=False, useCrossval=True, lamb=lambda_vals[i], mindegree = mindeg, maxdegree = maxdeg)
        for j in range(maxdeg-mindeg+1):
            MSE_2d[j,i] = MSE_test_list[j] #fix indexing cause of length

    plotfnc.gridsearch_plot(MSE_2d, lambda_vals, mindeg, maxdeg, savename="Lasso_grid", savefig = True)
    #plt.show()
    """


    #6 [2 figures, or 1 figure with 2 axis side by side]
    #BIAS-VARIANCE TRADEOFF, AS FUNCTION OF POLYDEG USING ONLY BOOTSTRAP,for both RIDGE AND LASSO

    lambda_vals = np.logspace(-2, 0, 3)
    ridge = []
    lasso = []
    mindeg = 0
    maxdeg = 8
    i = 0

    for lambda_ in lambda_vals:
        degrees_list, MSE_train_list, MSE_test_list, bias, variance, _, _, _, _ = \
        Solver(x_, y_, z_, Nx_, Ny_, Ridge, useBootstrap=True, useCrossval=False, lamb=lambda_, mindegree = mindeg, maxdegree = maxdeg)
        ridge.append([MSE_test_list, bias, variance])

        degrees_list, MSE_train_list, MSE_test_list, bias, variance, _, _, _, _ = \
        Solver(x_, y_, z_, Nx_, Ny_, Lasso, useBootstrap=True, useCrossval=False, lamb=lambda_, mindegree = mindeg, maxdegree = maxdeg)
        lasso.append([MSE_test_list, bias, variance])

    plotfnc.bias_var_lambdas(degrees_list, ridge, lasso, lambda_vals)
    plt.show()

    #bias_var_lambdas(degrees_list, bias, variance, MSE_test_list, lambdas, title = "BiasVar_lambdas", savefig = True, savename = "test")

    """
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

    """
    #Bias var plot example
    degrees_list, MSE_train_list, MSE_test_list, bias, variance, _, _, _ = Solver(x_, y_, z_, Nx_, Ny_, OLS, useBootstrap=True, useCrossval=False, maxdegree = maxdeg)
    plotfnc.bias_var_plot(degrees_list, bias, variance, MSE_test_list, title="Custom1", savefig = True)
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
