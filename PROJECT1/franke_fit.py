# Here we test the performance of OLS fit using the Franke Function to generate our dataset
# adding a random gaussian uncorrelated noise. Polynomials up to degree 5 are used as train


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
Nx = 1000
Ny = 1000
maxdegree = 5
MSE_train_list = np.zeros(maxdegree+1)
MSE_test_list = np.zeros(maxdegree+1)
R2_train_list = np.zeros(maxdegree+1)
R2_test_list = np.zeros(maxdegree+1)

for degree in range(maxdegree+1):
	x = np.random.rand(Nx, 1)
	y = np.random.rand(Ny, 1)
	x, y = np.meshgrid(x,y)

	#define own functions for MSE and R2 score
	def MSE(z, ztilde):
		return (1/len(z))*np.sum((z-ztilde)**2)

	def R2(z,ztilde):
		return 1- (MSE(z,ztilde))/np.var(z)

	def FrankeFunction(x,y):
		term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
		term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
		term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
		term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
		return term1 + term2 + term3 + term4

	#generate target data
	z = (FrankeFunction(x, y) + 1*np.random.randn(Nx,Ny)).reshape(-1,1)

	#set up design matrix for a polynomial of given degree
	X = np.zeros(( Nx*Ny, (degree+1)**2))
	for ii in range(degree+1):
		for jj in range(degree+1):
			column_index = (degree+1)*ii+jj
			X[:,column_index:column_index+1]  = (x**ii * y**jj).reshape(-1,1)

	#split data into train and test using scikitlearn
	X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2)

	#find optimal parameters using OLS
	beta_opt = np.linalg.pinv(X_train.T @ X_train)@X_train.T @ z_train
	z_tilde_train = X_train @ beta_opt
	z_tilde_test  = X_test @ beta_opt

	#evaluate MSE
	MSE_train_list[degree]  = MSE(z_train, z_tilde_train)
	MSE_test_list[degree]  = MSE(z_test, z_tilde_test)
	R2_train_list[degree]  = r2_score(z_train, z_tilde_train)
	R2_test_list[degree]   = r2_score(z_test, z_tilde_test)
# # Plot the surface.
# surf = ax.scatter(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# # Customize the z axis.
# ax.set_zlim(-0.10, 1.40)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()
fig, axs = plt.subplots(1,2)
degrees_x = np.arange(maxdegree+1)
axs[0].plot(degrees_x, MSE_train_list, label="train")
axs[0].plot(degrees_x, MSE_test_list, label="test")
axs[0].set_ylabel("MSE")

axs[1].plot(degrees_x, R2_train_list,  label="train")
axs[1].plot(degrees_x, R2_test_list,  label="test")
axs[1].set_ylabel("R2")

for ii in range(2):
	axs[ii].set_xlabel("degree of polynomial")
	axs[ii].grid(visible=True)
	axs[ii].legend()

plt.show()