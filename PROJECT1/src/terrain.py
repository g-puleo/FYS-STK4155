"""
FYS-STK4155
Project 1
Analysing terrain data



"""
from select import select
import franke_fit as ff
import numpy as np
import matplotlib
from imageio import imread
import plotting_functions as plotfnc
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pathlib

saveFigs = True
path = "./Plots/Terrain"
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

filename = 'SRTM_data_Norway_2.tif'

def prepare_grid(filename, location, selection_size, ravel=True):
    ''' Reads .tif file with altitude data.
        returns x,y, in meshgrid format, together with corresponding altitude h
        Inputs:
            filename,
            location = (x,y) pixels of upper left corner of region of interest
            selection size: select a N x N square grid, subset of the whole image.
    '''
    xmin = location[0]
    ymin = location[1]
    xmax = xmin + selection_size
    ymax = ymin + selection_size

    x = np.linspace(-1,1,selection_size) # keep in mind pixel indices are arbitrary units
    y = np.linspace(-1,1,selection_size)
    x, y = np.meshgrid(x,y)

    h = imread(filename)[ymin:ymax, xmin:xmax]

    if ravel:
        h = h.reshape(-1,1) # want array of shape (N,1)
        
    return x, y, h


def plot_raw_terrain(saveFigs=saveFigs):

    region_pos = [(1270,1290), (1180,160), (0,0)]
    selection_size = [  50, 50 , 1800]
    cmaps = ['gray']*2 + ['YlOrBr']
    fig, axs = plt.subplots(1,len(region_pos), figsize=(13,4))

    #normalize scale so that all figures have same colorscale
    h = [ prepare_grid(filename, region_pos[ii], selection_size[ii], ravel=False )[2] \
           for ii in range(len(region_pos)) ]


    #mynorm = matplotlib.cm.colors.Normalize(vmin=np.min( [np.min(h[ii]) for ii in range(2)]), \
    #    vmax = np.max( [np.max(h[ii]) for ii in range(2)] ) )

    for ii, h_ in enumerate(h):
        #output[2] of prepare_grid is the altitude profile as 2d matrix
        im = axs[ii].imshow(h_, cmap=cmaps[ii]) 
        axs[ii].grid(visible=False)
        fig.colorbar(im, ax=axs[ii], label="Altitude [m]")
        if ii==2:
            for jj in range(2):
                x_idx, y_idx = region_pos[jj]
                #y_idx -= selection_size[jj]
                rect = plt.Rectangle((x_idx,y_idx),selection_size[jj],selection_size[jj], fill=False, color="black")
                axs[ii].add_patch(rect)
    fig.tight_layout()
    if saveFigs:
        plt.savefig(f"{path}/terrain_data.pdf", dpi = 300)
    return fig, axs



def crossval_modelselection(locations, selection_size, method,lambda_vals,mindeg,maxdeg,saveFigs=saveFigs):
    '''
    Performs a model selection for Ridge/Lasso.
    Note:
        This function assumes that the crossval method in the solver function has a set randomstate
        otherwise when selecting the lambdas we would get different/random folds when fixing degree and varying lambda
        so not optimal, but just for illustration and further development
    -Uses the franke_fit.Solver method with cross-validation to get a MSE_validation score
    -Then selects for each degree the lambda that gave smallest validation score
    -Then tests each of these models (degree, lambda) (using no resampling)
    -Optimal model is the one with the smallest test MSE
    -Inputs:
        - locations: list of terrain locations (x,y)
        - selection_size
        - Ridge and Lasso as methods
    requires array of lambda values to scan
    and which polynomial complexities to fit
    saveFigs optional to save plots
    returns optimal prediction + optimal parameters for each location
    '''
    predictions = []
    optimal_params = []
    fig, axs = plt.subplots(1,len(locations),figsize=(14,5)) # figure for plotting MSE_test
    fig_lambda, axs_lambda = plt.subplots(1,len(locations),figsize=(14,5)) # figure for plotting MSE as fn of lambda
    for jj in range(len(locations)):
        
        ax = axs[jj]
        ax_lambda = axs_lambda[jj]

        # prepare data
        location = locations[jj]
        x,y,h = prepare_grid(filename, location, selection_size)

        ##########perform Cross Validation for different values of lambda:
        
        # run cross validation to asses error for different lambdas:
        n_lambdas = len(lambda_vals)
        validation_MSE = np.empty((maxdeg-mindeg+1,n_lambdas))

        for ii, lamb in enumerate(lambda_vals):
            current_validation_MSE_list = ff.Solver(x, y, h, Nx=selection_size, Ny=selection_size, method=method, \
                    lamb = lamb, useBootstrap = False, useCrossval = True,\
                    mindegree=mindeg,maxdegree=maxdeg)[2]
            # ff.Solver()[2] corresponds to the MSE_test_list, which contains the MSE_test estimates
            # for various polynomial degrees.
            validation_MSE[:,ii] = np.array(current_validation_MSE_list)
        
        # plot scaled val. MSE for degree with minimal val. MSE as fn of lambda
        # to illustrate dependecy on lambda
        row_idx = np.argmin(validation_MSE)//validation_MSE.shape[1]
        ax_lambda.plot(lambda_vals, validation_MSE[row_idx]/np.var(h))
        ax_lambda.set_xlabel("$\lambda$")
        ax_lambda.set_ylabel("Val. MSE / Var(terrain)")
        ax_lambda.set_xscale("log")
        ax_lambda.yaxis.set_major_formatter(FormatStrFormatter('%.03f'))

        #find optimal lambdas (minimum of each row)
        optimal_lambdas = lambda_vals[np.argmin(validation_MSE, axis=1)]
        
        # now test each model (degree, lambda_opt)
        MSE_test_list = []
        z_pred_list = []
        for degree in range(mindeg, maxdeg+1):
            lamb = optimal_lambdas[degree-mindeg]
            _, _, MSE_test, _, _, _, _, _, z_pred = ff.Solver(x, y, h, Nx=selection_size, Ny=selection_size, method=method, \
                    lamb = lamb, useBootstrap = False, useCrossval = False,\
                    mindegree=degree,maxdegree=degree)
            MSE_test_list.append(MSE_test)
            z_pred_list.append(z_pred[0])
        
        # plot relative MSE (= 1 - R2 score)
        ax.plot(np.arange(mindeg, maxdeg+1), MSE_test_list/np.var(h))
        ax.set_xlabel("Order of polynomial"); ax.set_ylabel("Test MSE / Var(terrain)")
        ax.xaxis.get_major_locator().set_params(integer=True) # force x-axis to show integer degrees

        # save optimal parameters as a tuple (degree, lambda)
        optimal_degree = np.argmin(MSE_test_list) + mindeg
        optimal_lambda = optimal_lambdas[np.argmin(MSE_test_list)]
        optimal_params.append((optimal_degree,optimal_lambda))

        # save predicted terrain
        prediction = z_pred_list[np.argmin(MSE_test_list)]
        predictions.append(prediction)

        print(f"For terrain in location {location} the optimal model parameters using {method.__name__} are: degree {optimal_degree}, lambda = {optimal_lambda}.")

    fig.tight_layout()
    fig_lambda.tight_layout()
    if saveFigs:
        fig.savefig(f"{path}/terrain_MSE_test_with_{method.__name__}.pdf", dpi=300)
        fig_lambda.savefig(f"{path}/model_selection_with_{method.__name__}.pdf", dpi=300)
    return predictions, optimal_params

def get_mean_relerror(pred,data):
    # returns the mean rel. error for predictions of data
    relerror_array = np.abs((pred - data)/data)
    return np.linalg.norm(relerror_array, ord=1)

# plotting data
plot_raw_terrain()

# setting some parameters
mindeg = 3
maxdeg = 26
locations = [(1270,1290), (1180,160)]
selection_size = 50
methods = [ff.OLS, ff.Ridge, ff.Lasso]

# setting up plotting of optimal predictions for each location and method
fig, axs = plt.subplots(len(locations), len(methods), figsize=(14,5))
ims = [None]*len(locations)*len(methods) # saving ims to set up colorbars

# also create figure to plot MSEs for OLS
fig_OLS, axs_OLS = plt.subplots(1, len(locations), figsize=(14,5))

mean_relerror_dict = {}

for jj, method in enumerate(methods):
    # plot predictions for each method
    if method!=ff.OLS:
        # if method is Ridge or Lasso perform a model selection for best degree and lambda
        n_lambdas = 20
        if method==ff.Ridge:
            lambda_vals = np.logspace(-7,1,n_lambdas)
        else:
            lambda_vals = np.logspace(-14,-6,n_lambdas) # smaller lambdas for Lasso
        predictions, optimal_params = crossval_modelselection(locations,selection_size,method,lambda_vals,mindeg,maxdeg)
    
    mean_relerror_dict[method.__name__] = [] # create a list containing mean rel. error for given method
    # loop over locations:
    for ii, location in enumerate(locations):
        ax = axs[ii,jj]
        x, y, h = prepare_grid(filename, location, selection_size)
        if method==ff.OLS:
            degrees_list, MSE_train_list, MSE_test_list, _, _, _, _, _, z_pred_list = \
                ff.Solver(x, y, h, Nx=selection_size, Ny=selection_size, method=method, \
                    lamb = 0., useBootstrap = False, useCrossval = False,\
                    mindegree=mindeg,maxdegree=maxdeg)
            # plot relative MSE (= 1 - R2 score)
            ax_OLS = axs_OLS[ii]
            ax_OLS.plot(degrees_list, MSE_test_list/np.var(h),label="Test")
            ax_OLS.plot(degrees_list, MSE_train_list/np.var(h),label="Train")
            ax_OLS.set_xlabel("Order of polynomial"); ax_OLS.set_ylabel("MSE / Var(terrain)")
            ax_OLS.xaxis.get_major_locator().set_params(integer=True) # force x-axis to show integer degrees
            ax_OLS.legend()

            modeldegree = np.argmin(MSE_test_list) + mindeg
            prediction = z_pred_list[np.argmin(MSE_test_list)]
            ax.set_title(f"OLS, p={modeldegree}", fontsize=10)
            #ax.annotate(f"p = {modeldegree}", xy=(selection_size//10,selection_size//10), color="red")
        else:
            prediction = predictions[ii] # jj=1 = Ridge, jj=2 = Lasso
            modeldegree = optimal_params[jj-1][0]
            lamb = optimal_params[ii][1]
            ax.set_title("{:s}, p={:.0f}, $\lambda$={:.2e}".format(method.__name__, modeldegree, lamb),\
                fontsize=10)
            #ax.annotate("p = {:.0f}".format(modeldegree), xy=(selection_size//10,selection_size//10), color="red")
            #ax.annotate("$\lambda$={:.2e}".format(lamb), xy=(selection_size//10, selection_size-selection_size//10), color="red")
        
        ims[ii+jj] = ax.imshow(prediction, cmap="gray")
        ax.set_xticks([]); ax.set_yticks([]) # remove pixel ticks on terrain figure
        
        mean_relerror_dict[method.__name__].append(get_mean_relerror(prediction,h.reshape(selection_size,selection_size)))
        

fig_OLS.tight_layout()
fig.tight_layout()

for i in range(len(locations)):
    fig.colorbar(ims[i*len(methods)], ax = axs[i,:], label="Altitude [m]")

if saveFigs:
    fig_OLS.savefig(f"{path}/terrain_MSE_test_with_OLS.pdf", dpi=300)
    fig.savefig(f"{path}/terrain_predictions.pdf", dpi=300)

plt.show()

print(mean_relerror_dict)








#terrain = terrain.reshape(-1,1) # want array of shape (n,1)
#terrain = utils.FrankeFunction(x,y) + 0.*np.random.randn(selection_size, selection_size)

"""

print("zbar",terrain[selection_size//2,selection_size//2])
terrain = terrain.reshape(-1,1)
# using cross-validation as resampling technique
maxdeg = 20

####
#### O L S
####



optimal_model = np.argmin(MSE_test_list)
print("Optimal polynomial has degree ", optimal_model)
print("zbar",z_pred_list[0][selection_size//2,selection_size//2])
print("zbar",z_pred_list[optimal_model][selection_size//2,selection_size//2])
fig2 = plt.figure()
plt.imshow(z_pred_list[optimal_model],cmap="gray")
plt.colorbar()

print("no of points", z_pred_list[optimal_model].shape)

from franke_fit import *
useCrossval = False
z_pred_list, degrees_list, MSE_train_list, MSE_test_list, bias, variance, _, _, _ = \
    Solver(x, y, terrain, Nx=selection_size, Ny=selection_size, method=Ridge, \
        lamb = 0.0, useBootstrap = useCrossval, useCrossval = useCrossval, maxdegree=maxdeg)

optimal_model = np.argmin(MSE_test_list)
print("Optimal polynomial has degree ", optimal_model)
print("zbar",z_pred_list[0][selection_size//2,selection_size//2])
print("zbar",z_pred_list[optimal_model][selection_size//2,selection_size//2])
plt.figure()
plt.imshow(z_pred_list[optimal_model],cmap="gray")
plt.colorbar()
print("no of points", z_pred_list[optimal_model].shape)

plt.show()

plt.plot(degrees_list, MSE_test_list, label="MSE_Test")
plt.plot(degrees_list, MSE_train_list, label="MSE_Train")
plt.plot(degrees_list, bias, label="bias")
plt.plot(degrees_list, variance, label="variance")
plt.plot(degrees_list, bias - variance, label="bias-var")
plt.grid()
plt.legend()
plt.show()
"""


"""

#################
## Ridge
##
##################
fig, axs = plt.subplots(nrows=2)

i = 0
for lamb in np.logspace(-8,2,11):#[1e-7, 1e-6, 1e-3, 1e-1]:
    print(lamb)
    degrees_list, MSE_train_list, MSE_test_list, bias, variance, error = \
        Solver(x, y, terrain, Nx=selection_size, Ny=selection_size, method=Ridge, \
        lamb = lamb, useBootstrap = False, useCrossval = True, maxdegree=maxdeg)
    axs[i].plot(degrees_list, MSE_test_list, label=f"{lamb}")#MSE_test")
    #axs[i].plot(degrees_list, error, label="error")
    #axs[i].plot(degrees_list, bias, label="Bias")
    #axs[i].plot(degrees_list, variance, label="Variance")
    axs[i].set_ylabel(f"$\lambda = $ {lamb}")
    axs[i].legend()
    #i += 1
#print(error - MSE_test_list)
axs[-1].set_xlabel("Degree")
plt.show()
"""
'''
i = 0
for lamb in [0]:#1e-7, 1e-6, 1e-5]:#, 1e-4]:
    print(lamb)
    degrees_list, MSE_train_list, MSE_test_list, bias, variance, error, beta, X = \
        Solver(x, y, terrain, Nx=selection_size, Ny=selection_size, method=Ridge, \
        lamb = lamb, useBootstrap = False, useCrossval = False, maxdegree=maxdeg)
    #axs[i].plot(degrees_list, MSE_test_list, label="MSE_test")
    axs[i].plot(degrees_list, error, label="error")
    axs[i].plot(degrees_list, bias, label="Bias")
    axs[i].plot(degrees_list, variance, label="Variance")
    axs[i].set_ylabel(f"$\lambda = $ {lamb}")
    axs[i].legend()
    beta = beta[:,-1]
    print(beta.shape, X.shape)
    z = X@beta
    z = z.reshape(selection_size, selection_size)
    print(z)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,\
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    i += 1
print(error - MSE_test_list)
axs[-1].set_xlabel("Degree")
plt.show()

'''