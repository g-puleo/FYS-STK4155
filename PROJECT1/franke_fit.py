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
import seaborn as sns
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


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
	beta_opt, z_tilde_train, z_tilde_test = Ridge(X_train, X_test, z_train, lamb = 0)
	return beta_opt, z_tilde_train, z_tilde_test

def Ridge(X_train, X_test, z_train, lamb):
	X_train_, X_test_, z_mean_train = scale(X_train, X_test, z_train)
	X_train_ = X_train_[:, 1:]
	X_test_ = X_test_[:, 1:]
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
	beta_opt = np.insert(beta_opt, 0, z_mean_train)
	return beta_opt, z_tilde_train, z_tilde_test

def Ridge_scikit(X_train, X_test, z_train, lamb):
	X_train_, X_test_, z_mean_train = scale(X_train, X_test, z_train)
	clf = linear_model.Ridge(alpha=lamb, fit_intercept=True)
	clf.fit(X_train_, z_train)
	z_tilde_train  = clf.predict(X_train_)
	z_tilde_test = clf.predict(X_test_)
	beta_opt = clf.coef_
	return beta_opt, z_tilde_train, z_tilde_test


def Lasso(X_train, X_test, z_train, lamb):
	X_train_, X_test_, z_mean_train = scale(X_train, X_test, z_train)
	#shift the prediction and remove first column of design matrix
	X_train_ = X_train_[:, 1:]
	X_test_ = X_test_[:, 1:]
	z_train_ = z_train - z_mean_train

	clf = linear_model.Lasso(alpha = lamb, fit_intercept=True)
	clf.fit(X_train_, z_train_)
	z_tilde_train  = clf.predict(X_train_) #questionable, u get back original z_train?
	#z_tilde_train = np.reshape(z_tilde_train.shape[0],1)
	z_tilde_test = clf.predict(X_test_).reshape(-1,1)
	beta_opt = clf.coef_
	beta_opt = np.insert(beta_opt, 0, z_mean_train)
	#print(beta_opt)
	#print(z_tilde_test)
	return beta_opt, z_tilde_train, z_tilde_test


def Solver(x, y, z, Nx, Ny, method, lamb = 0, useBootstrap = False, useCrossval = False, mindegree = 0, maxdegree = 12):
	#Set up list to store resultss
	z_pred_list = []
	MSE_train_list = np.zeros(maxdegree+1)
	MSE_test_list = np.zeros(maxdegree+1)
	R2_train_list = np.zeros(maxdegree+1)
	R2_test_list = np.zeros(maxdegree+1)
	bias = np.zeros(maxdegree+1)
	variance = np.zeros(maxdegree+1)
	error = np.zeros(maxdegree+1)
	beta_matrix = np.zeros( ( (maxdegree+1)*(maxdegree+2)//2, maxdegree+1 ) )

	#Print info when run
	print(f"Running solver with {method.__name__}. Degrees: {maxdegree}.", end = "")
	if useBootstrap:
		print("Using bootstrap ", end ="")
	elif useCrossval:
		print(f" Using crossvalidation with 5 k-folds.")
	print("\n")

	#Check if correct input
	if method not in [OLS, Ridge, Lasso, Ridge_scikit]:
		sys.exit(f"Error: Method [{method}] is not compatible. Must be in [OLS, Ridge, Lasso]")

	#Make sure Lambda is positive when using Ridge/Lasso
	if method != OLS and lamb < 0:
		sys.exit("Error: Lambda must have >=0 value if using Ridge or Lasso")

	for degree in range(mindegree, maxdegree+1):
		print(f"Degree {degree}/{maxdegree}")

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
			if mindegree == 0:
				beta_matrix[0:(degree+1)*(degree+2)//2, degree ] = beta_opt.ravel()
			#For Lasso get 2 diff values using diff MSE functions.
			MSE_train = utils.MSE(z_train, z_tilde_train)
			MSE_test = utils.MSE(z_test, z_tilde_test)
			#MSE_test = mean_squared_error(z_test, z_tilde_test)
			R2_train_list[degree]  = utils.R2(z_train, z_tilde_train)
			R2_test_list[degree]   = utils.R2(z_test, z_tilde_test)

		#evaluate MSE
		MSE_train_list[degree]  = MSE_train
		MSE_test_list[degree]  = MSE_test
		# add prediction to list (for plotting purposes)
		X_scaled,_,_ = scale(X,X,z) # want the scaled data
		if degree == 0:
			z_pred_ = beta_opt[0]*np.ones(len(z))
		else:
			z_pred_ = X_scaled[:,1:]@beta_opt[1:] + beta_opt[0]
		z_pred_list.append(z_pred_.reshape(Ny,Nx))

	degrees_list = np.arange(maxdegree+1)
	#Basic plot of MSE scores for train and test.

	"""
	plt.title(f"{method.__name__} boot: {useBootstrap}, cross: {useCrossval}")
	plt.plot(degrees_list, MSE_train_list[mindegree:], label = "Train")
	plt.plot(degrees_list, MSE_test_list[mindegree:], label = "Test")
	plt.legend()
	plt.grid(True)
	"""

	return z_pred_list, degrees_list, MSE_train_list, MSE_test_list, bias, variance, beta_matrix, R2_train_list, R2_test_list

"""
#Solver(OLS, useBootstrap=False, useCrossval=False, useScaling = False)
plt.figure(1)
Solver(Ridge, useBootstrap=False, useCrossval=False, lamb=0.01, maxdegree = maxdeg)
plt.figure(2)
Solver(Ridge_scikit, useBootstrap=False, useCrossval=False, lamb=0.01, maxdegree = maxdeg)
plt.show()
"""

"""
np.random.seed(3463223)
fig = plt.figure()
# Make data.
Nx_ = 16
Ny_ = 16
maxdeg = 12
#generate x,y data from uniform distribution
x_ = np.random.rand(Nx_, 1)
y_ = np.random.rand(Ny_, 1)
x_, y_ = np.meshgrid(x_,y_)
z_ = (utils.FrankeFunction(x_, y_) + 0.1*np.random.randn(Nx_,Ny_)).reshape(-1,1)

beta_matrix = np.zeros( ( (maxdeg+1)*(maxdeg+2)//2, maxdeg+1 ) )

degrees_list, MSE_train_list, MSE_test_list, bias, variance = Solver(x_, y_, z_, Nx_, Ny_, OLS, useBootstrap=False, useCrossval=False, lamb=0.0001, maxdegree = maxdeg)
for i in range(6):
	plt.plot(degrees_list, beta_matrix[i,:], label=f"Beta{i}")
plt.legend()
plt.show()
"""


"""
plt.figure(1)
degrees_list, MSE_train_list, MSE_test_list, bias, variance = Solver(x_, y_, z_, Nx_, Ny_, OLS, useBootstrap=False, useCrossval=False, lamb=0.0001, maxdegree = maxdeg)
plt.show()
"""


"""
Config for a nice plot OLS bias var:
np.random.seed(3463223)
Nx = 16
Ny = 16
maxdeg = 12
#Bias - variance tradeoff plotting:
degrees_list, MSE_train_list, MSE_test_list, bias, variance = Solver(OLS, useBootstrap=True, useCrossval=False, maxdegree = maxdeg)
plt.plot(degrees_list, bias, label="Bias")
plt.plot(degrees_list, variance, label="Variance")
plt.plot(degrees_list, MSE_test_list, label="Error")
plt.legend()
plt.show()
"""


"""
#Gridsearch
np.random.seed(5653456)
# Make data.
Nx_ = 16
Ny_ = 16
maxdeg = 10
#generate x,y data from uniform distribution
x_ = np.random.rand(Nx_, 1)
y_ = np.random.rand(Ny_, 1)
x_, y_ = np.meshgrid(x_,y_)
z_ = (utils.FrankeFunction(x_, y_) + 0.1*np.random.randn(Nx_,Ny_)).reshape(-1,1)

lambda_vals = np.logspace(-6, 0, 14)
mindeg = 3
MSE_2d = np.zeros(shape=(maxdeg+1-mindeg ,len(lambda_vals)))

#Fill array with MSE values. x-axis lambda, y-axis degree
for i in range(len(lambda_vals)):
	degrees_list, MSE_train_list, MSE_test_list, _, _ = Solver(x_, y_, z_, Nx_, Ny_, Ridge, useBootstrap=False, useCrossval=False, lamb=lambda_vals[i], mindegree = mindeg, maxdegree = maxdeg)
	for j in range(maxdeg-mindeg+1):
		MSE_2d[j,i] = MSE_test_list[mindeg+j] #fix indexing cause of length

df= pd.DataFrame(MSE_2d, columns= lambda_vals, index = np.arange(mindeg, maxdeg+1))
fig = sns.heatmap(df, cbar_kws={'label': 'MSE'})
fig.set(xlabel="Lambda", ylabel="Degree of complexity")
plt.show()
"""

#Solver(OLS, useBootstrap=False, useCrossval=False)
#Solver(OLS, useBootstrap=True, useCrossval=False)
