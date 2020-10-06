# from scipy import stats
# from astropy.stats import jackknife_resampling
# from astropy.stats import jackknife_stats
# import math
# import scipy.linalg as la
from sklearn import linear_model
from sklearn.utils import resample
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.stats import norm
import time
import sys
import os

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

def CalculateMediationPEEffect(PointEstimate2, PointEstimate3, ia = 0, ib = 1):
    # Indirect effect
    a = PointEstimate2[0][ia]
    b = PointEstimate3[0][ib]
    IE = a*b
    # Direct effect
    DE = PointEstimate3[0][0]
    TE = DE + IE
    return IE, TE, DE, a, b
    
def CalculateMediationResampleEffect(BSbetalist2, BSbetalist3, ia = 0, ib = 1):
    # Indirect effect
    a = np.squeeze(BSbetalist2[:,ia])
    b = np.squeeze(BSbetalist3[:,ib])
    IE = a*b
    DE = np.squeeze(BSbetalist3[:,0])
    TE = DE + IE
    return IE, TE, DE, a, b
    
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

def ExploringIdea():  
    # How do the variance in the simulated data, the weights, the betas and the Bs 
    # relate to each other?
    N = 100
    data = MakeIndependentData(N, [1,15,1], [10,10,100], [0.5, 0.5, 0.5], 99)
    beta, beta0 = Calculate_Beta_Sklearn(data)
    
    lm = linear_model.LinearRegression()
    X = data[:,0:-2]
    y = data[:,-2]
    model = lm.fit(X,y)
    beta = model.coef_
    
    B = beta*X.std(0)/y.std()
    print(beta)
    print(B)
    
def CalculateKappaEffectSize():
    # https://github.com/NCMlab/ProcessModelsNeuroImage/blob/master/FinalCode/CalculateKappa2.m
    pass

def CalculateRegressionEffectSizes():
    pass
    



def Calculate_Beta_Sklearn(data):
    # using linear regression model from sklearn 
    lm = linear_model.LinearRegression()
    # M = data.shape[1]
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


def MakeIndependentData(N = 1000, means = [0,0,0], stdev = [1,1,1], weights = [0, 0, 0], Atype = 99):
    # weights = AtoC, BtoC, AtoB
    # Make sure everything is the correct size
    M = len(means)
    S = len(stdev)
    W = len(weights)
# try:
    data = np.zeros([N,M])
    # Create independent data
    # columns are A, B, C
    for i in range(M):
        data[:,i] = np.random.normal(means[i], stdev[i], N)
    if Atype == 1:
        data[:,0] = np.random.uniform(20,80,N)
    if Atype == 2:
        data[:,0] = np.concatenate((np.zeros(int(N/2)), np.ones(int(N/2))))
    # Add weights between predictors to DV
    AtoB = weights[-1] # This part is super confusing!!!!
    AtoC = weights[0]
    BtoC = weights[1]
    # Make C data
    # Make a weighted combo of A and B
    temp = np.zeros(N)
    for i in range(M-1):
        temp = temp + (data[:,i])*weights[i]
    # Add thsi weighted combo to C
    data[:,-1] = temp + data[:,-1]
    # Make B data    
    data[:,1] = data[:,1] + (data[:,0])*AtoB
# except:
#     data = -99
    return data


def CalculateIndPower(NBoot, NSimMC, N, typeA, alpha, AtoB, AtoC, BtoC ):
    print("Starting simulations...")
    t = time.time()
    # Prepare a matrix for counting significant findings
    MClist = np.zeros((NSimMC,5))
    # Repeatedly generate data for Monte Carlo simulations 
    for i in range(NSimMC):
        print("%d of %d"%(i+1,NSimMC))
        # Make data
        data = MakeIndependentData(N, [1,1,1], [1,1,1], [AtoB, AtoC, BtoC], typeA)
        # data = MakeMultiVariableData(N,means, covs)
        # Point estimates
        PointEstimate2 = Calculate_Beta_Sklearn(data[:,[0,1]])
        PointEstimate3 = Calculate_Beta_Sklearn(data)
        # Point estimate mediation effects
        IE, TE, DE, a, b = CalculateMediationPEEffect(PointEstimate2, PointEstimate3)
        
        # Bootstrap model 2
        BSbetalist2, BSinterlist2 = Bootstrap_Sklearn(NBoot,Calculate_Beta_Sklearn, data[:,[0,1]])
        JKbetalist2, JKinterlist2 = JackKnife(Calculate_Beta_Sklearn, data[:,[0,1]])
        # Bootstrap model 3
        BSbetalist3, BSinterlist3 = Bootstrap_Sklearn(NBoot,Calculate_Beta_Sklearn, data)
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
    Power = MClist.sum(0)/NSimMC
    # Prepare output data
    # Nboot, Nsim, N, AtoB, AtoC, BtoC, typeA, powIE, powTE, powDE, powa, powb
    outdata = [NBoot, NSimMC, N, AtoB, AtoC, BtoC, typeA, Power[0], Power[1], Power[2], Power[3], Power[4]]
    print("Run time was: %0.2f"%(time.time() - t))
    
    return outdata


def main():
    if len(sys.argv[1:]) != 8:
        print("ERROR")
    else:
        print("Getting ready")
        # Pass the process ID to use for setting the seed
        pid = os.getpid() 
        # Set the seed
        np.random.seed(pid + int(time.time()))
        # Run the sim
        NBoot = int(sys.argv[1:][0])
        NSim = int(sys.argv[1:][1])
        N = int(sys.argv[1:][2])
        Atype = int(sys.argv[1:][3])
        AtoB = float(sys.argv[1:][4])
        AtoC = float(sys.argv[1:][5])
        OutDir = sys.argv[1:][6]
        BtoC = float(sys.argv[1:][7])
        BtoCArray = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
        # If this parameter is too big then assume that the sbatch array was used
        if BtoC > 0.9:
            BtoC = BtoCArray[int(BtoC)-1]
            
        
        alpha = 0.05
        print("Calling the simulator")
        outdata = CalculateIndPower(NBoot,NSim, N, Atype, alpha, AtoB, AtoC, BtoC)
        print("Done with the simulations")
        # Make outputfile name
        clock = time.localtime()
        OutFileName = "SimData_NB_%d_NSim_%d_"%(NBoot,NSim)
        OutFileName = OutFileName+str(clock.tm_hour)+"_"+str(clock.tm_min)+"__"+str(clock.tm_mon)+"_"+str(clock.tm_mday)+"_"+str(clock.tm_year)
        OutFileName = OutFileName+'_pid'+str(pid)+'.csv'
        np.savetxt(os.path.join(OutDir, OutFileName), outdata, delimiter = ',')

if __name__ == "__main__":
    #MakeBatchScripts()
    main()

       
