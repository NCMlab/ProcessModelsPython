#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 09:51:05 2020

@author: jasonsteffener
"""

def CheckMediationPower(NBoot, data):
    alpha = 0.05
    MClist = np.zeros(5)
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
        MClist[0] = 1        
    if TECI[0]*TECI[1] > 0:
        MClist[1] = 1        
    if DECI[0]*DECI[1] > 0:
        MClist[2] = 1        
    if aCI[0]*aCI[1] > 0:
        MClist[3] = 1        
    if bCI[0]*bCI[1] > 0:
        MClist[4] = 1        
    return MClist


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

def MakeData(N = 1000, means = [1,1], covs = np.eye(2), meanDV = 1, stdDV = 1, weights = [0, 0]):
    # Make predictors
    x = np.random.multivariate_normal(means, covs, N)
    y = np.random.normal(meanDV, stdDV, N)

    for i in range(len(weights)):
        y = y + x[:,i]*weights[i]
    data = np.append(x,np.expand_dims(y, axis = 1), axis = 1)
    return data
    
    # Make the predictors related to each other and added a weighted components to the DV
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

def SetupSims(NBoot, NSimMC):
    print("Starting the setup")
    alpha = 0.05
    # NBoot = 100
    # NSimMC = 100
    column_names = ['SampleSize','Atype','AtoB','AtoC','BtoC','IE', 'TE', 'DE', 'a', 'b']

    df = pd.DataFrame(columns = column_names)
    N = np.arange(10,101,10)
    # covAB = np.arange(-1,1.1,0.33)
    typeA = [99,1,2] # cont, unif, dicotomous
    # Aratio = [4,3,2,1,0.5,0.33,0.25]
    # varA = 1#np.arange(0.1,2.01,0.5)
    # varB = 1#np.arange(0.1,2.01,0.5)
    # varC = 1#np.arange(0.1,2.01,0.5)       
    AtoB = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]#np.arange(-0.5,0.1,0.5)
    AtoC = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]# np.arange(-1.0,1.01,0.5)
    BtoC = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]# np.arange(-1.0,1.01,0.5)    
        
    count = 0
    for i1 in N:
        for i3 in typeA:
            for i8 in AtoB:
                for i9 in AtoC:
                    for i10 in BtoC:
                        count += 1
    NAllSims = count
    print("There will be %d simulations run"%(NAllSims))
    SimData = np.zeros([NAllSims,10])
    count = 0
    t = time.time()
    for i1 in N:
        for i3 in typeA:
            for i8 in AtoB:
                for i9 in AtoC:
                    for i10 in BtoC:
                        print()
                        
                        MClist = np.zeros((NSimMC,5))  
                        for j in range(NSimMC):    
                            data = MakeIndependentData(i1, [1,1,1], [1,1,1], [i8, i9, i10], i3)                            
                        # MakeIndependentData(N = 1000, means = [0,0,0], stdev = [1,1,1], weights = [0, 0, 0], Atype = 99):
                  
                            MClist[j,:] =  CheckMediationPower(NBoot, data)
                        tempMC = MClist.sum(0)
                        new_row = [i1, i3, i8, i9, i10, tempMC[0], tempMC[1], tempMC[2], tempMC[3], tempMC[4]]
                        print("%d out of %d in %0.3f sec"%(count, NAllSims, time.time() - t))
                        SimData[count,:] = new_row
                        # this_column = df.columns[count]
                        # df[this_column] = new_row
                        count += 1
                        t = time.time()
    return SimData
    