#File to generate all the plots and results for the report.

#Including frankefunction 3d, betavalues, MSE, bias-variance, gridsearch

# Bonus: visualize the predicted z-values in 3d-space.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib
from franke_fit import *
import utils
from mpl_toolkits import mplot3d
import matplotlib
#matplotlib.rcParams['text.usetex'] = True

colorpal = sns.color_palette("deep")


sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes


def MSE_plot(degrees_list, MSE_train_list, MSE_test_list, mindegree= 0, titles_ = ["MSE"], savefig = False, savename = "MSE", path = "./Plots/MSE"):
    fig, axs = plt.subplots(nrows = 1, ncols = len(MSE_train_list), sharey = True, tight_layout=True, figsize=(7*len(MSE_train_list),5))
    plt.gca().set_ylim(bottom=0)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    for i in range(len(MSE_train_list)):
        #A fix for if length is 1 as u cant call upon axs[0]
        if len(MSE_train_list) > 1:
            axs_ = axs[i]
        else:
            axs_ = axsq
        axs_.autoscale(enable=True, axis="y", tight=False)
        axs_.plot(degrees_list, MSE_train_list[i][mindegree:], label = "Train", color = colorpal[0])
        axs_.plot(degrees_list, MSE_test_list[i][mindegree:], label = "Test", color = colorpal[1])
        axs_.title.set_text(f"{titles_[i]}")
        axs_.set_xlabel("Order of polynomial")
        axs_.legend()

    if len(MSE_train_list) > 1:
        axs[0].set_ylabel("MSE")
    else:
        axs.set_ylabel("MSE")
    plt.grid(True)
    if savefig:
        plt.savefig(f"{path}/{savename}.png", dpi = 300)
    return


def MSE_R2_plot(degrees_list, MSE_train_list, MSE_test_list, R2_train_list, R2_test_list, savefig = False, path = "./Plots/MSER2"):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, sharey = False, tight_layout=True, figsize=(7*2,5))
    plt.gca().set_ylim(bottom=0)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    #axs.autoscale(enable=True, axis="y", tight=False)
    axs[0].plot(degrees_list, MSE_train_list, label = "Train", color = colorpal[0])
    axs[0].plot(degrees_list, MSE_test_list, label = "Test", color = colorpal[1])
    axs[1].plot(degrees_list, R2_train_list, label = "Train", color = colorpal[0])
    axs[1].plot(degrees_list, R2_test_list, label = "Test", color = colorpal[1])
    axs[0].title.set_text(f"MSE")
    axs[1].title.set_text(f"R2")
    for i in range(2):
        axs[i].set_xlabel("Order of polynomial")
        axs[i].legend()
    axs[0].set_ylabel("MSE Error")
    axs[1].set_ylabel("R2 Error")
    plt.grid(True)
    if savefig:
        plt.savefig(f"{path}/MSER2_OLS.png", dpi = 300)
    return

def bias_var_plot(degrees_list, bias, variance, MSE_test_list, savename = "biasvar", savefig = False, path = "./Plots/BiasVar"):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7,5), tight_layout=True)
    plt.plot(degrees_list, bias, label = "Bias", color = colorpal[0])
    plt.plot(degrees_list, variance, label = "Variance", color = colorpal[1])
    plt.plot(degrees_list, MSE_test_list, label = "MSE", color = colorpal[2])
    plt.xlabel("Order of polynomial")
    plt.ylabel("Numerical estimate")
    plt.grid(True)
    plt.legend()
    if savefig:
        plt.savefig(f"{path}/{savename}.png", dpi = 300)
    return

def bias_var_lambdas(degrees_list, lists , lambdas, title = "BiasVar", savefig = False, savename = "fig", path = "./Plots/BiasVar"):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, sharey = False, tight_layout=True, figsize=(7*2,5))
    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    methods = ['Ridge', 'Lasso']
    for i in range(2):
        for j in range(len(lambdas)):
            axs[i].plot(degrees_list, lists[i][j][0], label = f"$\\lambda$={lambdas[j]}", linestyle=linestyles[0], color = colorpal[j])
            axs[i].plot(degrees_list, lists[i][j][1], linestyle=linestyles[1], color = colorpal[j])
            axs[i].plot(degrees_list, lists[i][j][2], linestyle=linestyles[2], color = colorpal[j])
        axs[i].legend()
        axs[i].set_xlabel("Order of polynomial")
        axs[i].set_yscale("log")

        axs[i].set_title(methods[i])

    if savefig:
        plt.savefig(f"{path}/{title}.png", dpi = 300)
    return


def betaval_plot(degrees_list, beta_mat, nr_ofbeta, maxdeg, title = "Betavalues", savefig = False, path = "./Plots/Betamatrix"):
    plt.figure(figsize=(7,5), tight_layout=True)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    for i in range(nr_ofbeta):
    	plt.plot(degrees_list[:maxdeg], beta_mat[i,:maxdeg], label=f"$\\beta_{i}$", color = colorpal[i])
    plt.grid(True)
    plt.xlabel("Order of polynomial")
    plt.ylabel("Optimal parameter")
    plt.legend()
    if savefig:
        plt.savefig(f"{path}/{title}.png", dpi = 300)
    return

def gridsearch_plot(MSE_2d_values, lambda_vals, mindeg, maxdeg, savefig = False, savename = "grid", path = "./Plots/Gridsearch"):
    plt.figure(figsize=(7,5), tight_layout=True)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(MSE_2d_values, columns= lambda_vals, index = np.arange(mindeg, maxdeg+1))
    df.round(2)
    fig = sns.heatmap(df, cbar_kws={'label': 'MSE'})
    fig.set(xlabel="Lambda", ylabel="Order of polynomial")
    if savefig:
        plt.savefig(f"{path}/{savename}.png", dpi = 300)
    return
