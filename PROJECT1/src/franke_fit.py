import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn import linear_model
import utils
import sys
import warnings
warnings.filterwarnings("ignore")

def scale(X_train, X_test, z_train):
	"""
	Scales the design matrix using sklearn StandardScaler and calculate mean value of z(target data)

	Args:
		X_train (ndarray) : Array containing training data
		X_test (ndarray) : Array containing test data
		z_train (ndarray) : Array containing target data for the test model_selection

	Returns:
		X_train_ (ndarray) : Scaled version of X_train
		X_test_ (ndarray) : Scaled version of X_test
		z_mean_train (float) : Mean value of z_train
	"""
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train_ = scaler.transform(X_train)
	X_test_ = scaler.transform(X_test)
	z_mean_train = np.mean(z_train)
	return X_train_, X_test_, z_mean_train

def OLS(X_train, X_test, z_train, lamb=0):
	"""
	Solves a regression model where the loss function is the linear least squares function with no regularization
	Runs Ridge method with lambda value set to 0.

	Args:
		X_train (ndarray) : Array containing training data
		X_test (ndarray) : Array containing test data
		z_train (ndarray) : Array containing target data for the test model_selection
		lamb (float) : Constant that multiplies the L2 term, controlling regularization strength

	Returns:
		 beta_opt (ndarray) : Array containing optimized paramaters
		 z_tilde_train (ndarray) : Predicted z values using beta_opt on the train data
		 z_tilde_test (ndarray) : Predicted z values using beta_opt on the test data
	"""
	beta_opt, z_tilde_train, z_tilde_test = Ridge(X_train, X_test, z_train, lamb = 0)
	return beta_opt, z_tilde_train, z_tilde_test

def Ridge(X_train, X_test, z_train, lamb):
	"""
	Solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm

	Args:
		X_train (ndarray) : Array containing training data
		X_test (ndarray) : Array containing test data
		z_train (ndarray) : Array containing target data for the test model_selection
		lamb (float) : Constant that multiplies the L2 term, controlling regularization strength

	Returns:
		 beta_opt (ndarray) : Array containing optimized paramaters
		 z_tilde_train (ndarray) : Predicted z values using beta_opt on the train data
		 z_tilde_test (ndarray) : Predicted z values using beta_opt on the test data
	"""
	X_train_, X_test_, z_mean_train = scale(X_train, X_test, z_train)
	X_train_ = X_train_[:, 1:] #Remove first column from data
	X_test_ = X_test_[:, 1:]
	z_train_ = z_train - z_mean_train
	tmp = X_train_.T @ X_train_
	beta_opt = np.linalg.pinv(tmp + lamb* np.eye(tmp.shape[0]))@X_train_.T @ z_train_
	z_tilde_train = X_train_ @ beta_opt + z_mean_train #add back intercept to data by adding z_mean value
	z_tilde_test  = X_test_ @ beta_opt + z_mean_train
	beta_opt = np.insert(beta_opt, 0, z_mean_train) #add back beta0 in beta matrix
	return beta_opt, z_tilde_train, z_tilde_test

def Ridge_scikit(X_train, X_test, z_train, lamb):
	"""
	Solves a regression model where the loss function is the linear least squares function with L2 regularization.
	Makes use of scikit Ridge model. Used to verify results.

	Args:
		X_train (ndarray) : Array containing training data
		X_test (ndarray) : Array containing test data
		z_train (ndarray) : Array containing target data for the test model_selection
		lamb (float) : Constant that multiplies the L2 term, controlling regularization strength

	Returns:
		 beta_opt (ndarray) : Array containing optimized paramaters
		 z_tilde_train (ndarray) : Predicted z values using beta_opt on the train data
		 z_tilde_test (ndarray) : Predicted z values using beta_opt on the test data
	"""
	X_train_, X_test_, z_mean_train = scale(X_train, X_test, z_train)
	clf = linear_model.Ridge(alpha=lamb, fit_intercept=True)
	clf.fit(X_train_, z_train)
	z_tilde_train  = clf.predict(X_train_)
	z_tilde_test = clf.predict(X_test_)
	beta_opt = clf.coef_
	return beta_opt, z_tilde_train, z_tilde_test


def Lasso(X_train, X_test, z_train, lamb):
	"""
	Solves a regression model where the loss function is the linear least squares function with L1 regularization.
	Makes use of scikit Lasso model.

	Args:
		X_train (ndarray) : Array containing training data
		X_test (ndarray) : Array containing test data
		z_train (ndarray) : Array containing target data for the test model_selection
		lamb (float) : Constant that multiplies the L2 term, controlling regularization strength

	Returns:
		 beta_opt (ndarray) : Array containing optimized paramaters
		 z_tilde_train (ndarray) : Predicted z values using beta_opt on the train data
		 z_tilde_test (ndarray) : Predicted z values using beta_opt on the test data
	"""
	X_train_, X_test_, z_mean_train = scale(X_train, X_test, z_train)
	#Shift the prediction and remove first column of design matrix
	X_train_ = X_train_[:, 1:]
	X_test_ = X_test_[:, 1:]
	z_train_ = z_train - z_mean_train

	clf = linear_model.Lasso(alpha = lamb, fit_intercept=False)
	clf.fit(X_train_, z_train_)
	z_tilde_train  = clf.predict(X_train_) + z_mean_train
	z_tilde_test = clf.predict(X_test_) + z_mean_train
	beta_opt = clf.coef_
	beta_opt = np.insert(beta_opt, 0, z_mean_train)
	return beta_opt, z_tilde_train, z_tilde_test


def Solver(x, y, z, Nx, Ny, method, lamb = 0, useBootstrap = False, useCrossval = False, mindegree = 0, maxdegree = 12, showruninfo = False, useRandomState = True):
	"""
	Solves regression problem using OLS, Ridge or Lasso method over multiple degrees of polynomial complexity.
	Makes use of cross-validation with k folds or bootstrapping if requested.
	Calculates MSE, beta_matrix and z_prediction for all methods. For bootstrap also calculates bias and variance.
	For no resampling additionally the R2 score is calculated. Returns arrays of 0 if not calculated for specific method.

	Args:
		x (ndarray) : x-values meshgrid coordinate matrix
		y (ndarray) : y-values meshgrid coordinate matrix
		z (ndarray) : target data
		Nx (int) : Number of data points along x-axis
		Ny (int) : Number of data points along y-axis
		method (function) : OLS, Ridge, Ridge_scikit or Lasso as the method used in solver
		lamb (float) : Value multiplied with the regularization term. , default = 0
		useBootstrap (bool) : True for using bootstrap method. , default = False
		useCrossval (bool) : True for using cross-validation method, default = False
		mindegree (int) : Min degree of complexity for the polynomial. , default = 0
		maxdegree (int) : Max degree of complexity for the polynomial. , default = 12
		showruninfo (bool) : Show basic info on run, method and resampling method. , default=False
		useRandomState (bool) : meaning static state for splitting train and test data. False meaning random shuffle. , default=True

	Returns:
		degrees_list (ndarray) : 1D array containing degrees from mindeg to maxdeg
		MSE_train_list (ndarray) : Contains the MSE values from the training data
		MSE_test_list (ndarray) : Contains the MSE values predicted with the test data
		bias (ndarray) : Contains the values of the bias. Calculated when doing bootstrapping.
		variance (ndarray) : Contains the values of the variance. Calculated when doing bootstrapping.
		beta_matrix (ndarray) : Contains the optimized paramters of beta used when calculating the prediction values
		R2_train_list (ndarray) : R2 score calculated with OLS on the train data
		R2_test_list (ndarray) : R2 score calculated with OLS on the test data
		z_pred_list (ndarray) : Predicted z-data using the test set.
	"""
	#Set up lists to store results
	z_pred_list = []
	MSE_train_list = []
	MSE_test_list = []
	R2_train_list = np.zeros(maxdegree-mindegree+1)
	R2_test_list = np.zeros(maxdegree-mindegree+1)
	bias = np.zeros(maxdegree-mindegree+1)
	variance = np.zeros(maxdegree-mindegree+1)
	error = np.zeros(maxdegree-mindegree+1)
	beta_matrix = np.zeros( ( (maxdegree+1)*(maxdegree+2)//2, maxdegree-mindegree+1 ) )

	#Print basic run info : method, maxdegree and resampling method.
	if showruninfo:
		print(f"Running solver with {method.__name__}. Degrees: {maxdegree}.", end = "")
		if useBootstrap:
			print("Using bootstrap ", end ="")
		elif useCrossval:
			print(f"Using 5-fold cross validation .")

		print("\n")

	#Check if valid method in input.
	if method not in [OLS, Ridge, Lasso, Ridge_scikit]:
		sys.exit(f"Error: Method [{method}] is not compatible. Must be in [OLS, Ridge, Lasso]")

	if method != OLS and lamb < 0:
		sys.exit("Error: Lambda must have >=0 value if using Ridge or Lasso")

	if useRandomState:
		random_state = 26092022 #Set static state for splititng test and test data

	for degree in range(mindegree, maxdegree+1):
		#Set up design matrix for a polynomial of given degree
		X = np.zeros(( Nx*Ny, (degree+1)*(degree+2)//2))
		counter = 0
		for S in range(degree+1):
			for ii in range(S+1):
				# need to figure out what the right column index is = (degree+1)*ii+jj
				X[:,counter:counter+1]  = (x**ii * y**(S-ii)).reshape(-1,1)
				counter+=1
		#Split into train and test
		if useRandomState:
			X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2, random_state=random_state)
		else:
			X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2)

		#Bootstrap method
		if useBootstrap:
			N_bootstraps = X_train.shape[0]
			MSE_avg_train = 0
			MSE_avg_test = 0
			z_pred = np.empty((X_test.shape[0], N_bootstraps))
			for i in range(N_bootstraps):
				X_train_b, z_train_b = utils.singleBootstrap(X_train, z_train)
				if degree==0:
					# if deg=0 just have the intercept = mean(z)
					z_mean_train = np.array([[np.mean(z_train)]])
					beta_opt, z_tilde_train, z_tilde_test = z_mean_train, z_mean_train, z_mean_train
				else:
					beta_opt, z_tilde_train, z_tilde_test = method(X_train_b, X_test, z_train_b, lamb)
				MSE_avg_train += utils.MSE(z_train_b , z_tilde_train)
				MSE_avg_test += utils.MSE(z_test, z_tilde_test)
				z_pred[:,i:i+1] = z_tilde_test.reshape(-1,1)

			#Calculate average MSE over all bootstraps
			MSE_test=MSE_avg_test/N_bootstraps
			MSE_train=MSE_avg_train/N_bootstraps

			bias[degree-mindegree] = np.mean((z_test - np.mean(z_pred, axis=1, keepdims=True))**2)
			variance[degree-mindegree] = np.mean(np.var(z_pred, axis=1, keepdims=True))
			error[degree-mindegree] = np.mean( np.mean( (z_pred-z_test)**2, axis=1, keepdims=True))

		#Cross-validation method
		elif useCrossval:
			MSE_avg_train = 0
			MSE_avg_test = 0
			# Seperate data into k folds
			k = 5
			if useRandomState:
				kfold = KFold(n_splits = k, shuffle = True, random_state=random_state)
			else:
				kfold = KFold(n_splits = k, shuffle = True)
			for train_inds, test_inds in kfold.split(X_train):
				X_train_k, X_test_k = X_train[train_inds], X_train[test_inds]
				z_train_k, z_test_k = z_train[train_inds], z_train[test_inds]
				if degree==0:
					# if deg=0 just have the intercept = mean(z)
					z_mean_train = np.array([[np.mean(z_train_k)]])
					beta_opt, z_tilde_train, z_tilde_test = z_mean_train.reshape(-1), z_mean_train, z_mean_train
				else:
					beta_opt, z_tilde_train, z_tilde_test = method(X_train_k, X_test_k, z_train_k, lamb)

				MSE_avg_train += utils.MSE(z_train_k , z_tilde_train)
				MSE_avg_test += utils.MSE(z_test_k, z_tilde_test)

			MSE_test=MSE_avg_test/k
			MSE_train=MSE_avg_train/k

		else:
			if degree==0:
				# if deg=0 just have the intercept = mean(z)
				z_mean_train = np.array([[np.mean(z_train)]])
				beta_opt, z_tilde_train, z_tilde_test = z_mean_train, z_mean_train, z_mean_train
			else:
				beta_opt, z_tilde_train, z_tilde_test = method(X_train, X_test, z_train, lamb)
			if mindegree == 0:
				beta_matrix[0:(degree+1)*(degree+2)//2, degree ] = beta_opt.ravel()
			MSE_train = utils.MSE(z_train, z_tilde_train)
			MSE_test = utils.MSE(z_test, z_tilde_test)
			R2_train_list[degree-mindegree]  = utils.R2(z_train, z_tilde_train)
			R2_test_list[degree-mindegree]   = utils.R2(z_test, z_tilde_test)

		#Add MSE value for each degree/iteration
		MSE_train_list.append(MSE_train)
		MSE_test_list.append(MSE_test)
		#Add prediction to list (for plotting purposes)
		X_scaled,_,_ = scale(X,X,z) # want the scaled data
		if degree == 0:
			z_pred_ = beta_opt[0]*np.ones(len(z))
		else:
			z_pred_ = X_scaled[:,1:]@beta_opt[1:] + beta_opt[0]
		z_pred_list.append(z_pred_.reshape(Ny,Nx))

	degrees_list = np.arange(mindegree, maxdegree+1)

	return degrees_list, MSE_train_list, MSE_test_list, bias, variance, beta_matrix, R2_train_list, R2_test_list, z_pred_list
