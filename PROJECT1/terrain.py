"""
FYS-STK4155
Project 1
Analysing terrain data
"""

from franke_fit import *
from imageio import imread

filename = 'SRTM_data_Norway_2.tif'

selection_size = 20
xmin = 1000
xmax = xmin + selection_size
ymin = 1000
ymax = ymin + selection_size

terrain = imread(filename)[ymin:ymax, xmin:xmax]
x = np.arange(xmin, xmax)
y = np.arange(ymin, ymax)
x, y = np.meshgrid(x,y)
terrain = terrain.reshape(-1,1) # want array of shape (n,1)
#terrain = utils.FrankeFunction(x,y) + np.random.randn(selection_size, selection_size)
#terrain = terrain.reshape(-1,1)
# using cross-validation as resampling technique
maxdeg = 16

####
#### O L S
####


degrees_list, MSE_train_list, MSE_test_list, bias, variance, error = \
    Solver(x, y, terrain, Nx=selection_size, Ny=selection_size, method=Ridge, \
        lamb = 0, useBootstrap = False, useCrossval = True, maxdegree=maxdeg)

plt.plot(degrees_list, MSE_test_list, label="MSE_Test")
plt.plot(degrees_list, MSE_train_list, label="MSE_Train")
plt.plot(degrees_list, bias, label="bias")
plt.plot(degrees_list, variance, label="variance")
plt.plot(degrees_list, bias - variance, label="bias-var")
plt.grid()
plt.legend()
plt.show()
print(variance)


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