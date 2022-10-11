

import sys
sys.path.insert(1, './src') #Search the src folder for the modules
import plotting_functions as plotfnc
from franke_fit import *
import numpy as np
import matplotlib.pyplot as plt
import utils


def generate_reults(showfigs = False):
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

    #1 MSE AND R2 SCORES AS FUNCTION OF THE POLYNOMIAL DEGREE, OLS
    print("Plotting MSE and R2 score for OLS.")
    list_train = []
    test = []
    degrees_list, MSE_train_list, MSE_test_list, _, _, _, R2_train_list, R2_test_list, _ \
    = Solver(x_, y_, z_, Nx_, Ny_, OLS, lamb = 0, useBootstrap=False, useCrossval=False, maxdegree = maxdeg)
    plotfnc.MSE_R2_plot(degrees_list, MSE_train_list, MSE_test_list, R2_train_list, R2_test_list, savefig = True)

    # (2) BETA PARAMETERS, AS FUNCTION OF POLYDEG, OLS
    #Betamatrix plot
    print("Plotting betavalues, OLS.")
    deg_toplot = 6
    nr_of_betas = 6
    degrees_list, MSE_train_list, MSE_test_list, bias, variance, beta_matrix, _, _, _ = Solver(x_, y_, z_, Nx_, Ny_, OLS, useBootstrap=False, useCrossval=False, lamb=0.0001, maxdegree = maxdeg)
    plotfnc.betaval_plot(degrees_list, beta_matrix, nr_of_betas, maxdeg = deg_toplot, title = f"Betavalues_{nr_of_betas}", savefig = True)

    #(3) Bias-variance tradeoff, AS FUNCTION OF POLYDEG USING ONLY BOOTSTRAP, and only OLS
    print("Plotting biasvariance tradeoff, OLS w/bootstrap.")
    maxdeg_ = 10
    degrees_list, MSE_train_list, MSE_test_list, bias, variance, beta_matrix, R2_train_list, R2_test_list, z_pred \
    = Solver(x_, y_, z_, Nx_, Ny_, OLS, lamb = 0, useBootstrap=True, useCrossval=False, maxdegree = maxdeg_)
    plotfnc.bias_var_plot(degrees_list, bias, variance, MSE_test_list,  savefig = True)

    #(4)COMPARISON BETWEEN ESTIMATES OF MSE IN CROSSVAL AND BOOTSTRAP, OLS. 2 plots to reuse plotting func
    #Bootstrap value goes a bit crazy for higher complexity which used to a problem which i thought was fixed
    print("Plotting MSE for OLS w/boot and w/crossval.")
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

    #(5)(E-F: CROSSVAL), (5) [2 figures, one ridge, one lasso]
    # MSE COLORPLOT IN RIDGE AND LASSO (gridsearch)
    print("Plotting gridsearch MSE, ridge and lasso.")
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


    #6 [2 figures, or 1 figure with 2 axis side by side]
    #BIAS-VARIANCE TRADEOFF, AS FUNCTION OF POLYDEG USING ONLY BOOTSTRAP,for both RIDGE AND LASSO
    print("Plotting bias-variance bootstrapped ridge and lasso.")
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

    if showfigs:
        plt.show()

    return


def get_bool(var):
    if var == "y":
        var_ = True
    elif var == "n":
        var_ = False
    else:
        sys.exit("Wrong input, must be 'y' or 'n'. Exiting run. ")
    return var_

def main():
    #Run to generate all the plots using the frankie function.
    generate_reults(showfigs = False)
    """
    bool_gen = input("Do you want to generate all figures for FrankeFunction? (y/n)")
    bool_gen = get_bool(bool_gen)
    if bool_gen:
        showfigs_ = input("Do you want to show all figures? (y/n)")
        bool_show = get_bool(showfigs_)
        generate_reults(showfigs = bool_show)
    """


    #plot_terrain()


    #Plot whatever you want.

    #gen x, y ,z data
    # Call on solver with parameters
    # Call on what to plot.

    #Terrain Data example.
    #terrain plotting.
    return

if __name__ == "__main__":
    main()
