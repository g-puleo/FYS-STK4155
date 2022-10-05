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
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import utils
import sys

np.random.seed(110222)
fig = plt.figure()
# Make data.
Nx = 100
Ny = 100
maxdegree = 10
MSE_train_list = np.zeros(maxdegree+1)
MSE_test_list = np.zeros(maxdegree+1)
R2_train_list = np.zeros(maxdegree+1)
R2_test_list = np.zeros(maxdegree+1)
beta_matrix = np.zeros( ( (maxdegree+1)**2, maxdegree+1 ) )
bias = np.zeros(maxdegree+1)
variance = np.zeros(maxdegree+1)
error = np.zeros(maxdegree+1)
#generate x,y data from uniform distribution
x = np.random.rand(Nx, 1)
y = np.random.rand(Ny, 1)
x, y = np.meshgrid(x,y)




def OLS(X_train, X_test, z_train):
	beta_opt = np.linalg.pinv(X_train.T @ X_train)@X_train.T @ z_train
	#this matrix contains values of beta which we aim to plot
	#each column of this matrix will contain values of the parameters beta
	#we are then plotting some of the rows
	z_tilde_train = X_train @ beta_opt
	z_tilde_test  = X_test @ beta_opt
	return beta_opt, z_tilde_train, z_tilde_test

def Ridge():
	print("yo")


def Lasso():
	print("yo")


def Solver(method, lamb = -1, useBootstrap = False, useCrossval = False, useScaling = True):
	#generate target data
	z = (utils.FrankeFunction(x, y) + 0.1*np.random.randn(Nx,Ny)).reshape(-1,1)

	if method not in [OLS, Ridge, Lasso]:
		sys.exit(f"Error: Method [{method}] is not compatible. Must be in [OLS, Ridge, Lasso]")
	if method != OLS:
		if lamb < 0:
			sys.exit("Error: Lambda must have >=0 value if using Ridge or Lasso")

	for degree in range(maxdegree+1):
		print(f"Degree {degree}/{maxdegree}")
		#set up design matrix for a polynomial of given degree
		X = np.zeros(( Nx*Ny, (degree+1)*(degree+2)//2))
		counter = 0
		for S in range(degree+1):
			for ii in range(S+1):
				# need to figure out what the right column index is = (degree+1)*ii+jj
				X[:,counter:counter+1]  = (x**ii * y**(S-ii)).reshape(-1,1)
				counter+=1

		#split data into train and test using scikitlearn
		X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2, random_state=26092022)

		if useScaling:
			scaler = StandardScaler()
			scaler.fit(X_train)
			X_train = scaler.transform(X_train)
			X_test = scaler.transform(X_test)
			X_train[:,0:1] = 1
			X_test[:,0:1] = 1

		if useBootstrap:
			#NEXT TIME: compute average MSE on all models obtained by bootstrap.
			#-> PLOT : TRAIN_MSE AND TEST_MSE as function of polynomial complexity
			#-> 		think of the bias - variance tradeoff
			N_bootstraps = X_train.shape[0]
			MSE_avg_train = 0
			MSE_avg_test = 0
			z_pred = np.empty((X_test.shape[0], N_bootstraps))
			#bias = np.empty()

			for i in range(N_bootstraps):
				X_train_b, z_train_b = utils.singleBootstrap(X_train, z_train)
				beta_opt, z_tilde_train, z_tilde_test = method(X_train_b, X_test, z_train_b)
				MSE_avg_train += utils.MSE(z_train_b , z_tilde_train)
				MSE_avg_test += utils.MSE(z_test, z_tilde_test)
				z_pred[:,i:i+1] = z_tilde_test

			MSE_test=MSE_avg_test/N_bootstraps
			MSE_train=MSE_avg_train/N_bootstraps

			bias[degree] = np.mean((z_test - np.mean(z_pred, axis=1, keepdims=True))**2)
			variance[degree] = np.mean(np.var(z_pred, axis=1, keepdims=True))
			error[degree] = np.mean( np.mean( (z_pred-z_test)**2, axis=1, keepdims=True))

		elif useCrossval:

			MSE_avg_train = 0
			MSE_avg_test = 0
			#X, z = shuffle(X, z) #shuffle dataset corresponding
			# Seperate data into k folds
			k = 5
			kfold = KFold(n_splits = k, shuffle = True)
			for train_inds, test_inds in kfold.split(X):
				X_train, X_test = X[train_inds], X[test_inds]
				z_train, z_test = z[train_inds], z[test_inds]

				beta_opt, z_tilde_train, z_tilde_test = method(X_train, X_test, z_train)

				MSE_avg_train += utils.MSE(z_train , z_tilde_train)
				MSE_avg_test += utils.MSE(z_test, z_tilde_test)

				#print(utils.MSE(z_train , z_tilde_train))

			MSE_test=MSE_avg_test/k
			MSE_train=MSE_avg_train/k

		else:
			#find optimal parameters using OLS
			beta_opt, z_tilde_train, z_tilde_test = method(X_train, X_test, z_train)
			beta_matrix[0:(degree+1)*(degree+2)//2, degree:degree+1 ] = beta_opt

			MSE_train = utils.MSE(z_train, z_tilde_train)
			MSE_test = utils.MSE(z_test, z_tilde_test)
			R2_train_list[degree]  = utils.R2(z_train, z_tilde_train)
			R2_test_list[degree]   = utils.R2(z_test, z_tilde_test)

		#evaluate MSE
		MSE_train_list[degree]  = MSE_train
		MSE_test_list[degree]  = MSE_test

	#Basic plot of MSE scores for train and test.
	plt.plot(np.arange(maxdegree+1), MSE_train_list, label = "Train")
	plt.plot(np.arange(maxdegree+1), MSE_test_list, label = "Test")
	plt.legend()
	plt.show()

	"""
	if useBootstrap:
		plot(bias)
		plot(variance)
	"""

	"""
	# # Plot the surface.
	# surf = ax.scatter(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	# # Customize the z axis.
	# ax.set_zlim(-0.10, 1.40)
	# ax.zaxis.set_major_locator(LinearLocator(10))
	# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	# # Add a color bar which maps values to colors.
	# fig.colorbar(surf, shrink=0.5, aspect=5)
	# plt.show()
	#print(beta_matrix)
	fig, axs = plt.subplots(1,2)
	degrees_x = np.arange(maxdegree+1)
	axs[0].plot(degrees_x, MSE_train_list, label="train")
	axs[0].plot(degrees_x, MSE_test_list, label="test")
	axs[0].set_ylabel("MSE")

	axs[1].plot(degrees_x, R2_train_list,  label="train")
	axs[1].plot(degrees_x, R2_test_list,  label="test")
	axs[1].set_ylabel("R2")

	for ii in range(1):
		axs[ii].set_xlabel("degree of polynomial")
		axs[ii].grid(visible=True)
		axs[ii].legend()

	#plt.show()

	# fig_beta, axs_beta = plt.subplots(1,1)

	# for jj in range(6):
	# 	axs_beta.plot( np.arange(maxdegree+1), beta_matrix[jj,:], label=f"$beta {jj}$")

	# axs_beta.grid(visible=True)
	# axs_beta.legend()


	fig_bvt, axs_bvt = plt.subplots(1,1)

	axs_bvt.plot(np.arange(maxdegree+1), error, label='error')
	axs_bvt.plot(np.arange(maxdegree+1), bias, label='bias')
	axs_bvt.plot(np.arange(maxdegree+1), variance, label='variance')
	axs_bvt.legend()
	plt.show()
	"""

Solver(OLS, useBootstrap=False, useCrossval=False, useScaling = False)
#Solver(OLS, useBootstrap=False, useCrossval=False)
#Solver(OLS, useBootstrap=True, useCrossval=False)



"""
def bootstrap:
	DONE

def crossval:
	TO DO

def OLS():
	DONE

def Lasso
	TO DO

def Ridge
	TO DO


methods avialble = OLS, Ridge, Lasso

solver(Method):
	if bootstrap:
		method(data)

	elif crossval:
		method(data)


plot(Beta values as you increase degree) #just for OLS without resampling
plot(R2_degrees) #just for OLS without resampling

plot(MSE_degrees) # For all 3
plot(bias, variance) # All 3 methods, bootstrapped

MAIN FILE:

all variables
func = asdasdad
x = 0 -> 1000
y = 0 ->1000
main(OLS, bootstrap=True):

"""
