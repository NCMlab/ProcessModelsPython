# from scipy import stats
# from astropy.stats import jackknife_resampling
# from astropy.stats import jackknife_stats
# import math
# import scipy.linalg as la
from sklearn import linear_model
from sklearn.utils import resample
import numpy as np
import pandas as pd

d = MakeThreeVariableData(100,[1,1,1],[[1,0.1,0.1],[0.1,1,0.1],[0.1,0.1,1]])
Calculate_Beta_Sklearn(d[['X','Y']], d['Z'])

d = MakeThreeVariableData(100,[1,1,1],[[1,0.2,0.2],[0.2,1,0.2],[0.2,0.2,1]])
Calculate_Beta_Sklearn(d[['X','Y']], d['Z'])

NBoot = 200


Resample_Sklearn(NBoot,Calculate_Beta_Sklearn,X,y)


def MakeThreeVariableData(N = 1000, means = [1,1,1], covs = [[1,0,0],[0,1,0],[0,0,1]]):
    x, y, z = np.random.multivariate_normal(means, covs, N).T
    df = pd.DataFrame({'X': x, 'Y': y, 'Z':z})
    return df

def Calculate_Beta_Sklearn(X,y):
    # using linear regression model from sklearn 
    lm = linear_model.LinearRegression()
    #fit the x,y to the model,x will be a 2d matrix and y is a array
    model = lm.fit(X,y)
    #default method to calculate the beta
    beta = model.coef_
    inter = model.intercept_
    # print(beta)
    return beta,inter

def Resample_Sklearn(NBoot,func, data):
    N = data.shape[0]
    M = data.shape[1] - 1
    betalist = np.zeros([N,M])
    interlist = np.zeros([N,1])
    for i in range(0,NBoot):
        # randomly resample the dataset with the original set with replacement
        a = resample(X, N, replace=True, random_state=i)
        betalist[i,:],interlist[i] = func(data[[]],data[])#<<<FIX THIS
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