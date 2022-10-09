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
colorpal = sns.color_palette("deep")


sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes

def MSE_plot(degrees_list, MSE_train_list, MSE_test_list, mindegree= 0, title = "MSE", savefig = False, path = "./Plots/MSE"):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7,5), tight_layout=True)
    plt.title(f"{title}")
    plt.plot(degrees_list, MSE_train_list[mindegree:], label = "Train", color = colorpal[0])
    plt.plot(degrees_list, MSE_test_list[mindegree:], label = "Test", color = colorpal[1])
    plt.grid(True)
    plt.legend()
    if savefig:
        plt.savefig(f"{path}/{title}.png", dpi = 300)
    return

def bias_var_plot(degrees_list, bias, variance, MSE_test_list, title = "BiasVar", savefig = False, path = "./Plots/BiasVar"):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7,5), tight_layout=True)
    plt.title(f"{title}")
    plt.plot(degrees_list, bias, label = "Bias", color = colorpal[0])
    plt.plot(degrees_list, variance, label = "Variance", color = colorpal[1])
    plt.plot(degrees_list, MSE_test_list, label = "MSE Error", color = colorpal[2])
    plt.grid(True)
    plt.legend()
    if savefig:
        plt.savefig(f"{path}/{title}.png", dpi = 300)
    return

def betaval_plot():
    return

def gridsearch_plot(MSE_2d_values, lambda_vals, mindeg, maxdeg, title = "gridsearch", savefig = False, path = "./Plots/Gridsearch"):
    plt.figure(figsize=(7,5), tight_layout=True)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    df= pd.DataFrame(MSE_2d_values, columns= lambda_vals, index = np.arange(mindeg, maxdeg+1))
    fig = sns.heatmap(df, cbar_kws={'label': 'MSE'})
    fig.set(xlabel="Lambda", ylabel="Degree of complexity")
    if savefig:
        plt.savefig(f"{path}/{title}.png", dpi = 300)
    return


def function_plot(x, y, z):
    # Doesent work too well.
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')

    # Creating color map
    my_cmap = plt.get_cmap('hot')

    # Creating plot
    surf = ax.plot_surface(x, y, z,
                           cmap = my_cmap,
                           edgecolor ='none')

    fig.colorbar(surf, ax = ax,
                 shrink = 0.5, aspect = 5)

    ax.set_title('Surface plot')

    # show plot
    plt.show()
    return
