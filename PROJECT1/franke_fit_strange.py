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
from sklearn import linear_model

from sklearn.metrics import mean_squared_error
import utils
import sys

import warnings
warnings.filterwarnings("ignore")


np.random.seed(1102229789)
fig = plt.figure()
# Make data.
Nx = 15
Ny = 15
maxdegree = 15
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


def scale(X_train, X_test, z_train):
	#Scale data and return it + mean value from target train data.
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train_ = scaler.transform(X_train)
	X_test_ = scaler.transform(X_test)
	z_mean_train = np.mean(z_train)
	#X_train[:,0] = 0
	#_test[:,0] = 0
	return X_train_, X_test_, z_mean_train

def OLS(X_train, X_test, z_train, lamb=0):
	X_train_, X_test_, z_mean_train = scale(X_train, X_test, z_train)
	z_train_ = z_train - z_mean_train

	beta_opt = np.linalg.pinv(X_train_.T @ X_train_)@X_train_.T @ z_train_
	z_tilde_train = X_train_ @ beta_opt + z_mean_train
	z_tilde_test  = X_test_ @ beta_opt + z_mean_train
	#z_tilde_train = np.ravel(z_tilde_train)
	#z_tilde_test= np.ravel(z_tilde_test)
	return beta_opt, z_tilde_train, z_tilde_test

def Ridge(X_train, X_test, z_train, lamb):
	X_train_, X_test_, z_mean_train = scale(X_train, X_test, z_train)
	#Subtract mean from z to remove intercept
	#Find beta opt wtih new equation
	#Add mean of z to prediction_z
	z_train_ = z_train - z_mean_train

	tmp = X_train_.T @ X_train_
	beta_opt = np.linalg.pinv(tmp + lamb* np.eye(tmp.shape[0]))@X_train_.T @ z_train_
	z_tilde_train = X_train_ @ beta_opt + z_mean_train
	z_tilde_test  = X_test_ @ beta_opt + z_mean_train
	#z_tilde_train = np.ravel(z_tilde_train)
	#z_tilde_test= np.ravel(z_tilde_test)

	return beta_opt, z_tilde_train, z_tilde_test


def Lasso(X_train, X_test, z_train, lamb):
	X_train_, X_test_, z_mean_train = scale(X_train, X_test, z_train)
	clf = linear_model.LassoCV(alphas=np.array([lamb]), fit_intercept=True)
	clf.fit(X_train_, z_train)
	z_tilde_train  = clf.predict(X_train_) #questionable, u get back original z_train?
	z_tilde_test = clf.predict(X_test_)
	beta_opt = clf.coef_
	#print(beta_opt)
	#print(z_tilde_test)
	return beta_opt, z_tilde_train, z_tilde_test


def Solver(method, lamb = 0, useBootstrap = False, useCrossval = False):
	#generate target data
	z = (utils.FrankeFunction(x, y) + 0.1*np.random.randn(Nx,Ny)).reshape(-1,1)
	#print(z.shape)

	#Print info
	print(f"Running solver with {method.__name__}. Degrees: {maxdegree}.", end = "")
	if useBootstrap:
		print("Using bootstrap ", end ="")
	elif useCrossval:
		print(f" Using crossvalidation with 5 k-folds.")
	print("\n")

	#Check if correct input
	if method not in [OLS, Ridge, Lasso]:
		sys.exit(f"Error: Method [{method}] is not compatible. Must be in [OLS, Ridge, Lasso]")

	#Make sure Lambda is positive when using Ridge/Lasso
	if method != OLS and lamb < 0:
		sys.exit("Error: Lambda must have >=0 value if using Ridge or Lasso")

	for degree in range(0, maxdegree+1):
		#print(f"Degree {degree}/{maxdegree}")

		#set up design matrix for a polynomial of given degree
		X = np.zeros(( Nx*Ny, (degree+1)*(degree+2)//2))
		counter = 0
		for S in range(degree+1):
			for ii in range(S+1):
				# need to figure out what the right column index is = (degree+1)*ii+jj
				X[:,counter:counter+1]  = (x**ii * y**(S-ii)).reshape(-1,1)
				counter+=1


		#X = X[:,1:] #remove first column to remove intercept.
		X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2, random_state=26092022) #Split into train and test

		if useBootstrap:
			N_bootstraps = X_train.shape[0]
			MSE_avg_train = 0
			MSE_avg_test = 0
			z_pred = np.empty((X_test.shape[0], N_bootstraps))
			"""
			if Lasso:
				bootstrap()
				scitkit.Lasso(data)
			"""
			for i in range(N_bootstraps):
				#When ridge, new mean value so we have to scale here. Dont add 1 column.
				X_train_b, z_train_b = utils.singleBootstrap(X_train, z_train)
				beta_opt, z_tilde_train, z_tilde_test = method(X_train_b, X_test, z_train_b, lamb)
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

				beta_opt, z_tilde_train, z_tilde_test = method(X_train, X_test, z_train, lamb)

				MSE_avg_train += utils.MSE(z_train , z_tilde_train)
				MSE_avg_test += utils.MSE(z_test, z_tilde_test)

				#print(utils.MSE(z_train , z_tilde_train))

			MSE_test=MSE_avg_test/k
			MSE_train=MSE_avg_train/k

		else:
			#find optimal parameters using OLS
			beta_opt, z_tilde_train, z_tilde_test = method(X_train, X_test, z_train, lamb)
			beta_matrix[0:(degree+1)*(degree+2)//2, degree ] = beta_opt.ravel()
			#For Lasso get 2 diff values using diff MSE functions.
			MSE_train = utils.MSE(z_train, z_tilde_train)
			MSE_train2 = mean_squared_error(z_train, z_tilde_train)
			MSE_test = utils.MSE(z_test, z_tilde_test)
			#MSE_test = mean_squared_error(z_test, z_tilde_test)
			R2_train_list[degree]  = utils.R2(z_train, z_tilde_train)
			R2_test_list[degree]   = utils.R2(z_test, z_tilde_test)

		#evaluate MSE
		MSE_train_list[degree]  = MSE_train
		MSE_test_list[degree]  = MSE_test

	#Basic plot of MSE scores for train and test.
	plt.title(f"{method.__name__} boot: {useBootstrap}, cross: {useCrossval}")
	plt.plot(np.arange(maxdegree+1), MSE_train_list, label = "Train")
	plt.plot(np.arange(maxdegree+1), MSE_test_list, label = "Test")
	plt.legend()
	plt.grid(True)

	"""
	plt.figure(2)
	plt.plot(np.arange(maxdegree+1), R2_train_list, label = "Train")
	plt.plot(np.arange(maxdegree+1), R2_test_list, label = "Test")
	plt.legend()
	plt.show()
	"""


#Solver(OLS, useBootstrap=False, useCrossval=False, useScaling = False)
plt.figure(1)
Solver(Lasso, useBootstrap=False, useCrossval=False, lamb=0.01)
plt.savefig("MSE_ridge.png")
plt.show()


#Solver(OLS, useBootstrap=False, useCrossval=False)
#Solver(OLS, useBootstrap=True, useCrossval=False)



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
