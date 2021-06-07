#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 08:53:26 2020

@author: jasonsteffener
"""
import os
import numpy as np
import pandas as pd
import time
from itertools import product
def MakeBatchScripts():
    # Make all of the submission scripts.
    # Also make a dataframe to keep track of all of the expected simulations
    # What are the values for each parameter that we will use?
    sP = [1,0.66,0.33,0,-0.33, -0.66, -1]
    NSamples = np.arange(20,201,20)
    NumSampleSizes = NSamples.shape[0]
    NBoot = 1000
    NPower = 1000
    NParams = 8 # How many parameters are there for input to the model?
    Params = product(sP,sP,sP,sP,sP,sP,sP,sP)
    NSims = (7**8)*NSamples.shape[0]

    cNames = ['NBoot','NPower','N']
    for i in range(NParams):
        cNames.append("param%03d"%(i+1))
    cNames.append('SimID')
    cNames.append('Completed')


    dfOut = pd.DataFrame(columns=cNames)

    BaseDir = "/home/steffejr/scratch/Project"
    BaseDir = "/home/steffejr/Data"
    OutDir = "/home/steffejr/Data/Results"
    #OutDir = '/Users/jasonsteffener/Documents/GitHub'
    CodeDir = "/home/steffejr/scratch/ProcessModelsPython"
    SubmissionListFileName = os.path.join(OutDir,'SubmissionList.csv')    
    count = 0
    for i in Params:
        count += 1
        if count > -1:
            # Create parameters as input
            b1 = i[0]
            b2 = i[1]
            b3 = i[2]
            b4 = i[3]
            b5 = i[4]
            b6 = i[5]
            b7 = i[6]
            b8 = i[7]            
            # Create the script file
            SimID = '%06d'%(count)
            fileName = "submit_Process_%s"%(SimID)
            f = open(os.path.join(BaseDir, 'jobs', fileName+".sh"), "w")
            f.write("#!/bin/bash\n")                 
            f.write("#SBATCH --job-name=%s.job\n"%(os.path.join(BaseDir, 'jobs', fileName)))
            f.write("#SBATCH --output=%s.out\n"%(os.path.join(BaseDir, 'out', fileName)))
            f.write("#SBATCH --error=%s.err\n"%(os.path.join(BaseDir, 'out', fileName)))
            f.write("#SBATCH --time=06:00:00\n")
            f.write("#SBATCH --account=def-steffejr-ab\n")
            f.write("#SBATCH --mem-per-cpu=256M\n\n")
            # Added an array for at least one dimension of simulations
            f.write("#SBATCH --array=1-10\n")
            f.write("source ~/ENV/bin/activate\n")
            f.write("python %s %d %d %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %s %s\n" %(os.path.join(CodeDir, "ProcessTools.py"), NBoot,NPower,b1,b2,b3,b4,b5,b6,b7,b8, OutDir, '$SLURM_ARRAY_TASK_ID'))
            f.close()
            # Add this sim to the dataframe keeping track
            for j in NSamples:
                li = [NBoot, NPower,b1,b2,b3,b4,b5,b6,b7,b8,SimID, 0]
                row = pd.Series(li, index = cNames)
                dfOut = dfOut.append(row, ignore_index = True)
                            
                            # submit the file to the queue
                            # os.system('sbatch %s'%(os.path.join(BaseDir, 'jobs', fileName+".sh")))
                            # time.sleep(0.1)
    print("Saving submission list to file: %s"%SubmissionListFileName)
    dfOut.to_csv(SubmissionListFileName)
    print(count)

if __name__ == "__main__":
    MakeBatchScripts()
