# from scipy import stats
# from astropy.stats import jackknife_resampling
# from astropy.stats import jackknife_stats
# import math
# import scipy.linalg as la
from sklearn import linear_model
from sklearn.utils import resample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Make multivariate normal data. The inputs allow the means, variances and covariances to be adjusted
# The size of the data is determined by the size of the mean and cov matrix inputs

# Assume that the last column is the data and the rest are predictors

plt.figure()
plt.hist(betalist[:,0])
plt.hist(betalist[:,1])

sns.pairplot(pd.DataFrame(data))

NSimMC = 100

NBoot = 2000
N = 100
alpha = 0.05

means = [1,1,1]
Cov = 0.25
covs = [[1,Cov,Cov],[Cov,1,Cov],[Cov,Cov,1]]
data = MakeMultiVariableData(N,means, covs)
PointEstimate = Calculate_Beta_Sklearn(data)
BSbetalist,BSinterlist = Resample_Sklearn(NBoot,Calculate_Beta_Sklearn, data)
JKbetalist, JKinterlist = JackKnife(Calculate_Beta_Sklearn, data)

print(PointEstimate)
index = 0
print(CalculateBCaCI(BSbetalist[:,index], JKbetalist[:,index], PointEstimate[0][index], alpha))
index = 1
print(CalculateBCaCI(BSbetalist[:,index], JKbetalist[:,index], PointEstimate[0][index], alpha))

np.corrcoef(data.T)

MClist = CalculatePower(NSimMC, NBoot, N, alpha, means, covs)
MClist.sum(0)/NSimMC

def CalculatePower(NSimMC, NBoot, N, alpha, means, covs):
    # How many predictor columns are there?
    M = np.size(means) - 1 
    # Prepare a matrix for counting significant findings
    MClist = np.zeros((NSimMC,M))
    # Repeatedly generate data for Monte Carlo simulations 
    for i in range(NSimMC):
        # Make data
        data = MakeMultiVariableData(N,means, covs)
        # Bootstrap Resample
        BSbetalist,BSinterlist = Resample_Sklearn(NBoot,Calculate_Beta_Sklearn, data)
        # Jackknife
        JKbetalist, JKinterlist = JackKnife(Calculate_Beta_Sklearn, data)
        # Calculate the point estimate
        PointEstimate = Calculate_Beta_Sklearn(data)
        # Calculate the confidence intervals
        for index in range(M):
            BCaCI = CalculateBCaCI(BSbetalist[:,index], JKbetalist[:,index], PointEstimate[0][index], alpha)
            if BCaCI[0]*BCaCI[1] > 0:
                MClist[i, index] = 1
        # Calculate the confidence intervals for the multiplication of the mediation effect

    return MClist
    
def JackKnife(func, data):
    N,M = data.shape
    # prepare output arrays
    betalist = np.zeros([N,M-1])
    interlist = np.zeros([N,1])
    temp = np.arange(N)
    for i in range(0,N):
        JKresample = data[np.delete(temp, i),:]
        # randomly resample the dataset with the original set with replacement
        tempP = func(JKresample)
        betalist[i,:] = tempP[0]
        interlist[i] = tempP[1]
    return betalist,interlist

def CalculateBCaCI(BS, JK, PE, alpha):
    NBoot = BS.shape[0]
    N = JK.shape[0]
    zA = norm.ppf(alpha/2)
    z1mA = norm.ppf(1 - alpha/2)
    # Find resamples less than point estimate
    F = np.sum(BS < PE)
    BCaCI = [-1, 1]
    if F > 0:
        zh0 = norm.ppf(F/NBoot)
        ThetaDiff = JK.sum()/N - JK
        acc = ((ThetaDiff**3).sum())/(6*((ThetaDiff**2).sum())**(3/2))
        Alpha1= norm.cdf(zh0 + (zh0 + zA)/(1 - acc*(zh0 + zA)))
        Alpha2 = norm.cdf(zh0 + (zh0 + z1mA)/(1 - acc*(zh0 + z1mA)))

        PCTlower = np.percentile(BS,Alpha1*100)
        PCTupper = np.percentile(BS,Alpha2*100)
        BCaCI = [PCTlower, PCTupper]
    return BCaCI
    
def MakeMultiVariableData(N = 1000, means = [1,1,1], covs = [[1,0,0],[0,1,0],[0,0,1]]):
    x = np.random.multivariate_normal(means, covs, N)
    return x

def Calculate_Beta_Sklearn(data):
    # using linear regression model from sklearn 
    lm = linear_model.LinearRegression()
    #fit the x,y to the model,x will be a 2d matrix and y is a array
    model = lm.fit(data[:,[0,-2]], data[:,-1])
    return model.coef_, model.intercept_

def Bootstrap_Sklearn(NBoot, func, data):
    # How big is the data
    N,M = data.shape
    # prepare output arrays
    betalist = np.zeros([NBoot,M-1])
    interlist = np.zeros([NBoot,1])
    for i in range(0,NBoot):
        # randomly resample the dataset with the original set with replacement
        a = resample(data, n_samples=N, replace=True, random_state=i)
        temp = func(a)
        betalist[i,:] = temp[0]
        interlist[i] = temp[1]
    return betalist,interlist

# def m1_bootstrap(m,age):
#     coeflist=[]
#     interlist=[]
#     for i in range(0,len(m[0])):
#         mi = m[:,i].reshape(-1,1)
#         result_coef, result_inter = self.Resample_Sklearn(resample_time ,self.Calculate_Beta_Sklearn,mi,age)
#         coeflist.append(result_coef)
#         interlist.append(result_inter)
#     coeflist =np.array(coeflist)
#     interlist =np.array(interlist)
#     #return coeflist, interlist
#     return coeflist
# 
# def m1_jackknife(m,age):
#     coeflist_jackknife=[]
#     interlist_jackknife=[]
#     for i in range(0,len(m[0])):
#         mjack = m[:,i]
#         resamples = jackknife_resampling(mjack)
#         for ii in range(0,len(resamples)):
#             resamplesi = np.append(resamples[ii],[1])
#             resamplesi = resamplesi.reshape(-1,1)
#             result_coef_jackknife,result_inter_jackknife= self.Calculate_Beta_Sklearn(resamplesi,age)
#             coeflist_jackknife.append(result_coef_jackknife)
#             interlist_jackknife.append(result_inter_jackknife)
#         # result = Calculate_Beta_Numpy(mi,age)
#         # coeflist.append(result)
#     coeflist_jackknife =np.array(coeflist_jackknife)
#     interlist_jackknife =np.array(interlist_jackknife)
#     #return coeflist_jackknife,interlist_jackknife
#     return coeflist_jackknife