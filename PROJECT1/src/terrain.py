"""
FYS-STK4155
Project 1
Analysing terrain data
"""

"""
-input: filename, pixel location, selection_size, method
-output: optimal fit params (degree), plotted optimal fit, MSE
gridsearch_plot(MSE_2d_values, lambda_vals, mindeg, maxdeg, title = "gridsearch", savefig = False, path = "./Plots/Gridsearch"):

"""
import franke_fit as ff
import numpy as np
import matplotlib
from imageio import imread
import plotting_functions as plotfnc
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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
    # # plot terrain data
    # fig1 = plt.figure()
    # plt.imshow(h,cmap="gray")
    # #plt.savefig(f"terrain plot {filename}.pdf")
    # plt.colorbar()

    if ravel:
        h = h.reshape(-1,1) # want array of shape (N,1)
        
    return x, y, h
"""
def terrain_func(method):
    
    
    if method==OLS:
        lambda_vals = [0.]
    else:
        lambda_vals = [1e-7]# np.logspace(-8,0,1)

    mindeg = 0
    maxdeg = 26

    useCrossval = False
    MSE_2d_values = np.zeros((maxdeg-mindeg + 1, len(lambda_vals)))

    for lamb_idx, lamb in enumerate(lambda_vals):
        degrees_list, MSE_train_list, MSE_test_list, bias, variance, _, _, _, z_pred_list = \
            Solver(x, y, terrain, Nx=selection_size, Ny=selection_size, method=method, \
            lamb = lamb, useBootstrap = useCrossval, useCrossval = useCrossval, mindegree=mindeg,maxdegree=maxdeg)
        MSE_2d_values[:,lamb_idx] = MSE_test_list

    #fig = plotfnc.gridsearch_plot(MSE_2d_values, lambda_vals, mindeg, maxdeg, title = "gridsearch", savefig = False, path = "./Plots/Gridsearch")
    #from matplotlib.colors import LogNorm, Normalize
    #plt.figure(figsize=(7,5), tight_layout=True)
    #fig = plt.figure()
    #axes = plt.gca()
    #axes.imshow(MSE_2d_values, cmap="autumn", norm=LogNorm())
    #axes.set_xlabel("Lambda"); axes.set_ylabel("Order of polynomial")
    

    optimal_model_idx = np.argmin(MSE_2d_values)
    optimal_degree = optimal_model_idx//len(lambda_vals) + mindeg
    optimal_lambda_idx = optimal_model_idx%len(lambda_vals)
    optimal_lambda = lambda_vals[optimal_lambda_idx]
    print("Optimal polynomial has degree ", optimal_degree)
    if method!=OLS:
        print("with optimal lambda = ", optimal_lambda)
    plt.figure()
    plt.imshow(z_pred_list[optimal_degree-mindeg],cmap="gray")
    plt.colorbar()
    fig, ax = plt.subplots()
    ax.plot(degrees_list, MSE_2d_values[:,0], label="MSE_test")
    ax.plot(degrees_list, MSE_train_list, label="MSE_train")
    plt.legend()
    for degree in range(mindeg,maxdeg+1):
        cool_function = lambda lamb: Solver(x,y,terrain,selection_size,selection_size,method,lamb,useCrossval=True,mindegree=degree,maxdegree=degree)

    return
"""
##probably will be convenient to move this function to another file (plotting_functions.py)
filename = 'SRTM_data_Norway_2.tif'

def plot_raw_terrain():

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
        fig.colorbar(im, ax=axs[ii])
        if ii==2:
            for jj in range(2):
                x_idx, y_idx = region_pos[jj]
                #y_idx -= selection_size[jj]
                rect = plt.Rectangle((x_idx,y_idx),selection_size[jj],selection_size[jj], fill=False, color="black")
                axs[ii].add_patch(rect)
    fig.tight_layout()
    plt.savefig("terrain_data.pdf")
    return fig, axs




plot_raw_terrain()
plt.show()

'''
##THIS IS WHERE WE WILL DO CROSSVAL MODELSELECTION:
###import our data
location = 
selection_size = 50

x,y,h = prepare_grid(filename, location, selection_size)

##########perform Cross Validation for different values of lambda:

#set up range of polydegree to scan
mindeg = 1
maxdeg = 25

#set up grid of parameters to scan
lambda_vals = np.logspace(-8,0,1)
optimal_lambda = np.empty(maxdeg-mindeg+1)
#run cross validation to asses error for different lambdas:

validation_MSE = np.empty((maxdeg-mindeg+1,len(lambda_vals)))
for ii, lamb in enumerate(lmabda_vals):

    current_validation_MSE_list = ff.Solver(x, y, h, Nx=selection_size, Ny=selection_size, method=Ridge, \
            lamb = lamb, useBootstrap = False, useCrossval = True,\
             mindegree=mindeg,maxdegree=maxdeg)[3]
    #ff.Solver()[3] corresponds to the MSE_test_list, which contains the MSE_test estimates
    #for various polynomial degrees.
    validation_MSE[:,ii] = np.array(current_validation_MSE_list)

#find optimal lambda (minimum of each row)

optimal_lambda = np.min(validation_MSE, axis=1)


plt.show()

'''





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
