#!/usr/bin/env python
#SBATCH --output=%x-%j.log
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks-per-node=32

import sys
import os

sys.path.append(os.getcwd())

import process_models.tools as t
import time
import multiprocessing as mp
import pandas as pd
import numpy as np

# run with this command sbatch --export=ALL,BRAIN_DATA_PATH=/home/abare/scratch/python-distrubuted-test/AllDataCon0004.nii.gz,MASK_PATH=/home/abare/scratch/python-distrubuted-test/mask.nii.gz,COGNITIVE_PATH=/home/abare/scratch/python-distrubuted-test/data.csv,RESULTS_PATH=/home/abare/scratch/python-distrubuted-test,N_BOOTSTRAPS=100 cluster_bootstraping.py

if __name__ == '__main__':

    BRAIN_DATA_PATH = os.environ.get('BRAIN_DATA_PATH')
    MASK_PATH = os.environ.get('MASK_PATH')
    COGNITIVE_PATH = os.environ.get('COGNITIVE_PATH')
    RESULTS_PATH = os.environ.get('RESULTS_PATH')
    N_BOOTSTRAPS = int(os.environ.get('N_BOOTSTRAPS'))

    #Create the brain imaging data from the file
    brain_imaging = t.load_img_data(BRAIN_DATA_PATH)
    #Create the mask imaging data from the file
    mask = t.load_img_data(MASK_PATH)
    
    #Create the file n * m matrix where n is the number of patients and m the length of voxels
    M = t.create_brain_data(brain_imaging,mask)
    #Retrive the shape of the matrix
    n, m = M.shape

    #Load the cognitive scores and age scores from the path
    cog_scores = pd.read_csv(COGNITIVE_PATH)
    #cognitive scores
    Y = cog_scores.values[:,-1]
    #age scores
    X = cog_scores.values[:,-2]
    #combining all of the data
    combined_data = np.array([[X[i],M[i],Y[i]] for i in range(len(M))],dtype=object)
    #pool object with mp.cpu_count() amount of workers
    pool = mp.Pool(mp.cpu_count())
    #job list 1
    jobs1 = []
    #job list 2
    jobs2 = []
    # iterate n_bootstrap times
    for j in range(N_BOOTSTRAPS):
        #boostrap values X,Y,M
        bootstrap_value = t.boot_sample(combined_data,n,int(time.time()) + j)
        #if its the first iteration take the Point Estimate
        if j == 0:
            formated_M, _xy = t.flatten_data(M, X, Y)
        else:     
            formated_M, _xy = t.flatten_data(np.array([i.tolist() for i in bootstrap_value[:,1]]), np.array(combined_data[:,0], dtype='float'), np.array(combined_data[:,2], dtype='float'))


        #First equation jobs
        job1 = pool.apply_async(t.calculate_beta,args=(_xy[0].reshape(-1,1),formated_M))
        #Second eqation jobs
        job2 = pool.apply_async(t.calculate_beta,args=(t.combine_array(_xy[0],formated_M),_xy[1]))

        #append jobs
        jobs1.append(job1)
        jobs2.append(job2)
    pool.close()
    pool.join()
    t.save_results(f'{RESULTS_PATH}/results-1-{os.environ["SLURM_JOBID"]}.pkl',jobs1) 
    t.save_results(f'{RESULTS_PATH}/results-2-{os.environ["SLURM_JOBID"]}.pkl',jobs2)