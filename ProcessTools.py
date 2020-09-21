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
# Explore the following:
# strength of the relationships between 
# A and B: -1:0.1:1 (20)
# B and C: -1:0.1:1 (20)
# A and C: -1:0.1:1 (20)
# Show how the size of the mediation effect changes
# Change sample sizes N: 10:10:200 (20)
# TOTAL SIM = 160,000


# A is dichotomous with equal group size and redo
# TOTAL SIM = 8,000

# A is dichotomous with unequal group size and redo
# Group size ratio: 4, 3, 2, 0.5, 0.33, 0.25 (6)

# A is uniformly distributed (1)

# The covariance between A and B varies
# 0:0.1:0.9 (10)

# TOTAL SIM = 48,000
#
# plt.figure()
# plt.hist(betalist[:,0])
# plt.hist(betalist[:,1])
# 
# sns.pairplot(pd.DataFrame(data))

NSimMC = 10
NBoot = 200

N = 100
alpha = 0.05

means3 = [1,1,1]
means4 = [1,1,1,1]
Cov = 0.25
covs = [[1,Cov,Cov],[Cov,1,Cov],[Cov,Cov,1]]
Cov12 = 0.25
Cov13 = 0.25
Cov14 = 0.25
Cov23 = 0.25
Cov24 = 0.25
Cov23 = 0.25
Cov34 = 0.25
covs3 = [[1,Cov12,Cov13],[Cov12,1,Cov23],[Cov13,Cov23,1]]
covs4 = [[1,Cov12,Cov13,Cov14],[Cov12,1,Cov23, Cov24],[Cov13,Cov23,1,Cov34],[Cov14,Cov24,Cov34,1]]

# Make data  where the DV is the last column (3) and a covariate is in column 2
data = MakeMultiVariableData(N,means3, covs3)

# Add the moderator
data = MakeModeratedEffect(data,0,1, effect = 100)


data = MakeMultiVariableData(N,[1, 1], [[1,0.99],[0.99,1]])
print(np.corrcoef(data.T))
Calculate_Beta_Sklearn(data)


data = MakeIndependentData(N, means = [0,0,0], stdev = [1,1,1], weights = [1, 0.5, 0])
print(np.corrcoef(data.T))
data.mean(0)
Calculate_Beta_Sklearn(data)


# Point estimates
ai = 0
bi = 1
COVi = 2
yi = 3
# The last column is the DV
PointEstimate2 = Calculate_Beta_Sklearn(data[:,[ai,COVi, bi]])
# The last column is the DV
PointEstimate3 = Calculate_Beta_Sklearn(data[:,[ai,bi,COVi,yi]])

# Point estimate mediation effects and identify which effects to multiply
IE, TE, DE, a, b = CalculateMediationPEEffect(PointEstimate2, PointEstimate3, 0, 1)

# Bootstrap model 2
BSbetalist2, BSinterlist2 = Bootstrap_Sklearn(NBoot,Calculate_Beta_Sklearn, data[:,[ai,COVi, bi]])
JKbetalist2, JKinterlist2 = JackKnife(Calculate_Beta_Sklearn, data[:,[ai,COVi, bi]])
# Bootstrap model 3
BSbetalist3, BSinterlist3 = Bootstrap_Sklearn(NBoot,Calculate_Beta_Sklearn, data[:,[ai,bi,COVi,yi]])
JKbetalist3, JKinterlist3 = JackKnife(Calculate_Beta_Sklearn, data[:,[ai,bi,COVi,yi]])

# Bootstrap mediation effects
BSIE, BSTE, BSDE, BSa, BSb = CalculateMediationResampleEffect(BSbetalist2, BSbetalist3, 0, 1)
# Jackknifemediation effects
JKIE, JKTE, JKDE, JKa, JKb = CalculateMediationResampleEffect(JKbetalist2, JKbetalist3, 0 , 1)


IECI = CalculateBCaCI(BSIE, JKIE, IE, alpha)
TECI = CalculateBCaCI(BSTE, JKTE, TE, alpha)
DECI = CalculateBCaCI(BSDE, JKDE, DE, alpha)
aCI = CalculateBCaCI(BSa, JKa, a, alpha)
bCI = CalculateBCaCI(BSb, JKb, b, alpha)

# modCI = CalculateBCaCI(BSbetalist3[:,-1], JKbetalist3[:,-1], PointEstimate3[0][-1], alpha)

print("a: %0.3f (%0.3f : %0.3f)"%(a,aCI[0],aCI[1]))
print("b: %0.3f (%0.3f : %0.3f)"%(b,bCI[0],bCI[1]))
print("TE: %0.3f (%0.3f : %0.3f)"%(TE,TECI[0],TECI[1]))
print("DE: %0.3f (%0.3f : %0.3f)"%(DE,DECI[0],DECI[1]))
print("IE: %0.3f (%0.3f : %0.3f)"%(IE,IECI[0],IECI[1]))

np.corrcoef(data.T)

NSimMC = 100
NBoot = 200
N = 200
MClist = CalculatePower(NSimMC, NBoot, N, alpha, means, covs)
MClist.sum(0)/NSimMC

# Keep covs

def MakeModeratedEffect(data,i = 0, j = 1, effect = 0):
    # How big is the data
    N,M = data.shape
    # Remove mean of each column
    iD = data[:,i] - data[:,i].mean()
    print(iD.mean())
    jD = data[:,j] - data[:,j].mean()
    print(jD.mean())
    # Create the moderation regressor. 
    # This regressor is a vector and is converted to an array for the multiplication
    mD = np.array(iD*jD)
    print(mD.mean())
    # Add a moderator effect to the DV
    OutData = np.append(data, np.expand_dims(mD, axis = 1), axis = 1)
    OutData[:,-1] = OutData[:,-1] + mD*effect
    # Append the moderator to the data 
    return OutData


def SetupAllSims():
    NSimMC = 20
    NBoot = 100
    Cov12 = np.arange(-1,1.1,0.1)
    Cov13 = np.arange(-1,1.1,0.1)
    Cov23 = np.arange(-1,1.1,0.1)
    N = np.arange(10,201,10)
    NSimAll = Cov12.shape[0]*Cov13.shape[0]*Cov23.shape[0]*N.shape[0]
    for n in N:
        for i in Cov12:
            for j in Cov13:
                for k in Cov23:
                    covs = [[1,i,j],[i,1,k],[j,k,1]]
                    MClist = CalculatePower(NSimMC, NBoot, n, alpha, means, covs)
                    print(MClist.sum(0)/NSimMC)
                    
                    
def CalculateMediationPEEffect(PointEstimate2, PointEstimate3, ia = 0, ib = 1):
    # Indirect effect
    a = PointEstimate2[0][ia]
    b = PointEstimate3[0][ib]
    IE = a*b
    # Direct effect
    DE = PointEstimate3[0][0]
    TE = DE + IE
    return IE, TE, DE, a, b
    
def CalculateMediationResampleEffect(BSbetalist2, BSbetalist3, ia, ib):
    # Indirect effect
    a = np.squeeze(BSbetalist2[:,ia])
    b = np.squeeze(BSbetalist3[:,ib])
    IE = a*b
    DE = np.squeeze(BSbetalist3[:,0])
    TE = DE + IE
    return IE, TE, DE, a, b
    
def CalculatePower(NSimMC, NBoot, N, alpha, means, covs):

    # Prepare a matrix for counting significant findings
    MClist = np.zeros((NSimMC,5))
    # Repeatedly generate data for Monte Carlo simulations 
    for i in range(NSimMC):
        #print("%d of %d"%(i+1,NSimMC))
        # Make data
        data = MakeMultiVariableData(N,means, covs)
        # Point estimates
        PointEstimate2 = Calculate_Beta_Sklearn(data[:,[0,1]])
        PointEstimate3 = Calculate_Beta_Sklearn(data)
        # Point estimate mediation effects
        IE, TE, DE, a, b = CalculateMediationPEEffect(PointEstimate2, PointEstimate3)
        
        # Bootstrap model 2
        BSbetalist2, BSinterlist2 = Resample_Sklearn(NBoot,Calculate_Beta_Sklearn, data[:,[0,1]])
        JKbetalist2, JKinterlist2 = JackKnife(Calculate_Beta_Sklearn, data[:,[0,1]])
        # Bootstrap model 3
        BSbetalist3, BSinterlist3 = Resample_Sklearn(NBoot,Calculate_Beta_Sklearn, data)
        JKbetalist3, JKinterlist3 = JackKnife(Calculate_Beta_Sklearn, data)
        # Bootstrap mediation effects
        BSIE, BSTE, BSDE, BSa, BSb = CalculateMediationResampleEffect(BSbetalist2, BSbetalist3)
        # Jackknifemediation effects
        JKIE, JKTE, JKDE, JKa, JKb = CalculateMediationResampleEffect(JKbetalist2, JKbetalist3)
        
        IECI = CalculateBCaCI(BSIE, JKIE, IE, alpha)
        TECI = CalculateBCaCI(BSTE, JKTE, TE, alpha)
        DECI = CalculateBCaCI(BSDE, JKDE, DE, alpha)
        aCI = CalculateBCaCI(BSa, JKa, a, alpha)
        bCI = CalculateBCaCI(BSb, JKb, b, alpha)
        if IECI[0]*IECI[1] > 0:
            MClist[i, 0] = 1        
        if TECI[0]*TECI[1] > 0:
            MClist[i, 1] = 1        
        if DECI[0]*DECI[1] > 0:
            MClist[i, 2] = 1        
        if aCI[0]*aCI[1] > 0:
            MClist[i, 3] = 1        
        if bCI[0]*bCI[1] > 0:
            MClist[i, 4] = 1        
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

def MakeIndependentData(N = 1000, means = [0,0,0], stdev = [1,1,1], weights = [0, 0, 0]):
    # Make sure everything is the correct size
    M = len(means)
    S = len(stdev)
    W = len(weights)
    if (M == S) and (M == W):
        data = np.zeros([N,M])
        # Create independent data
        for i in range(M):
            data[:,i] = np.random.normal(means[i], stdev[i], N)
    # Add weights between predictors to DV
    for i in range(M-1):
        data[:,-1] = ddd= data[:,-1] + (data[:,i])*weights[i]
    return data


def Calculate_Beta_Sklearn(data):
    # using linear regression model from sklearn 
    lm = linear_model.LinearRegression()
    M = data.shape[1]
    #fit the x,y to the model,x will be a 2d matrix and y is a array
    # if M > 2:
    model = lm.fit(data[:,0:-1], data[:,-1])
    # else:
    #     model = lm.fit(data[:,[0]], data[:,-1])
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

def MakeData(
def MakeIndependentData(N = 1000, means = [0,0,0], stdev = [1,1,1], weights = [0, 0, 0]):
    # Make sure everything is the correct size
    M = len(means)
    S = len(stdev)
    W = len(weights)
    if (M == S) and (M == W):
        data = np.zeros([N,M])
        # Create independent data
        for i in range(M):
            data[:,i] = np.random.normal(means[i], stdev[i], N)
    # Add weights between predictors to DV
    for i in range(M-1):
        data[:,-1] = ddd= data[:,-1] + (data[:,i])*weights[i]
    return data
