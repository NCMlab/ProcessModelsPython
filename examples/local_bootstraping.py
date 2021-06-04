import sys
import os

sys.path.append(os.getcwd())

import process_models.tools as t
import time
import multiprocessing as mp
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Load up the 4-D image data  
    BaseDir = '/Volumes/GoogleDrive/Shared drives/JasonData/Steffener_CSI4900/Jeremy_Arik'
    image = t.load_img_data(os.path.join(BaseDir, "AllDataCon0004.nii.gz"))
    # Load up the mask image
    mask = t.load_img_data(os.path.join(BaseDir, "mask.nii.gz"))
    # mask and flatten the 4-D data
    M = t.create_brain_data(image,mask)
    # Read in the one-dimensional data, i.e. age, cognitive scores
    df = pd.read_csv(os.path.join(BaseDir, "data.csv"))
    # This is assuming that the mediation model has brain data as the mediator M
    # Y = df.values[:,-1]
    Y = np.random.rand(39)
    X = np.random.rand(39)
    # X = df.values[:,-2]
    # What is the size of the brain data
    n,m = M.shape
    
    combined_data = np.array([[X[i],M[i],Y[i]] for i in range(len(M))],dtype=object)
    # Create a pool of workers based on the number cpus 
    pool = mp.Pool(mp.cpu_count())
    # There is one job for each equation that needs to be estimated
    # For simple mediation, there is are two equations: X --> M and X + M --> Y
    jobs1 = []
    jobs2 = []
    N_BOOTSTRAPS = 10
    # iterate n_bootstrap times
    for j in range(N_BOOTSTRAPS):
        bootstrap_value = t.boot_sample(combined_data,n,int(time.time()) + j)

        formated_M, _xy = t.flatten_data(np.array([i.tolist() for i in bootstrap_value[:,1]]), np.array(combined_data[:,0], dtype='float'), np.array(combined_data[:,2], dtype='float'))
        # Calculate the linear regression beta weights by passing two arguments: 
            # the output and the model
        job1 = pool.apply_async(t.calculate_beta,args=(_xy[0].reshape(-1,1),formated_M))

        job2 = pool.apply_async(t.calculate_beta,args=(t.combine_array(_xy[0],formated_M),_xy[1]))

        jobs1.append(job1)
        jobs2.append(job2)
    # Close the pool
    pool.close()
    # By joining the pool this will wait until all jobs are completed
    pool.join()
    t.save_results(f'results-1.pkl',jobs1) 
    t.save_results(f'results-2.pkl',jobs2) 