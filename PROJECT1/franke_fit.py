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

fig = plt.figure()
# Make data.
Nx = 10
Ny = 10
maxdegree = 5
MSE_train_list = np.zeros(maxdegree+1)
MSE_test_list = np.zeros(maxdegree+1)
R2_train_list = np.zeros(maxdegree+1)
R2_test_list = np.zeros(maxdegree+1)
beta_matrix = np.zeros( ( (maxdegree+1)**2, maxdegree+1 ) )
#generate x,y data from uniform distribution
x = np.random.rand(Nx, 1)
y = np.random.rand(Ny, 1)
x, y = np.meshgrid(x,y)
print(f"Franke(0,0)={FrankeFunction(0,0)}")
for degree in range(maxdegree+1):
	
	#define own functions for MSE and R2 score
	
	#generate target data
	z = (FrankeFunction(x, y) + 1.5*np.random.randn(Nx,Ny)).reshape(-1,1)

	#set up design matrix for a polynomial of given degree
	X = np.zeros(( Nx*Ny, (degree+1)**2))
	counter = 0
	for S in range(degree+1):
		for ii in range(S+1):
			# need to figure out what the right column index is = (degree+1)*ii+jj
			X[:,counter:counter+1]  = (x**ii * y**(S-ii)).reshape(-1,1)
			counter+=1
	#split data into train and test using scikitlearn
	X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2)

	# scaler = StandardScaler()
	# scaler.fit(X_train)
	# X_train = scaler.transform(X_train)
	# X_test = scaler.transform(X_test)


	#find optimal parameters using OLS
	beta_opt = np.linalg.pinv(X_train.T @ X_train)@X_train.T @ z_train
	#this matrix contains values of beta which we aim to plot
	#each column of this matrix will contain values of the parameters beta
	#we are then plotting some of the rows
	beta_matrix[0:(degree+1)**2, degree:degree+1 ] = beta_opt
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
#print(beta_matrix)
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

fig_beta, axs_beta = plt.subplots(1,1)

for jj in range(6):
	axs_beta.plot( np.arange(maxdegree+1), beta_matrix[jj,:], label=f"$beta {jj}$")


axs_beta.grid(visible=True)
axs_beta.legend()
	

plt.show()