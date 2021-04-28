import sys
import os

sys.path.append(os.getcwd())

import process_models.tools as t
import time
import multiprocessing as mp
import pandas as pd
import numpy as np

if __name__ == '__main__':
    
    image = t.load_img_data("/Users/arikbarenboim/Documents/ProcessModelsPython/AllDataCon0004.nii.gz")
    mask = t.load_img_data("/Users/arikbarenboim/Documents/ProcessModelsPython/mask.nii.gz")
    M = t.create_brain_data(image,mask)

    df = pd.read_csv("/Users/arikbarenboim/Documents/ProcessModelsPython/data.csv")

    Y = df.values[:,-1]
    X = df.values[:,-2]

    n,m = M.shape

    combined_data = np.array([[X[i],M[i],Y[i]] for i in range(len(M))],dtype=object)

    pool = mp.Pool(mp.cpu_count())
    jobs1 = []
    jobs2 = []
    for j in range(10):
        bootstrap_value = t.boot_sample(combined_data,n,int(time.time()) + j)

        formated_M, _xy = t.flatten_data(np.array([i.tolist() for i in bootstrap_value[:,1]]), np.array(combined_data[:,0], dtype='float'), np.array(combined_data[:,2], dtype='float'))

        job1 = pool.apply_async(t.calculate_beta,args=(_xy[0].reshape(-1,1),formated_M))

        job2 = pool.apply_async(t.calculate_beta,args=(t.combine_array(_xy[0],formated_M),_xy[1]))

        jobs1.append(job1)
        jobs2.append(job2)
    pool.close()
    pool.join()
    t.save_results(f'results-1.pkl',jobs1) 
    t.save_results(f'results-2.pkl',jobs2) 