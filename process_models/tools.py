#==================================
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
from sklearn import linear_model
import pandas as pd
import time
#==================================
# load image (4D) [X,Y,Z_slice,time]
#nii_img  = nib.load("/Users/arikbarenboim/Documents/ProcessModelsPython/AllDataCon0004.nii.gz")
#nii_data = nii_img.get_fdata()

#nii_binary = np.where(np.isnan(nii_data), 0, 1)

def load_img_data(path):
    nii_img = nib.load(path)
    return nii_img.get_fdata()

def create_mask(img_data):
    return np.where(np.isnan(img_data), 0, 1)

def create_brain_data(img_data, mask):
    # Check if the mask and img_data are the same size later
    print(img_data.shape)
    brain_data = []
    for i in range(img_data.shape[3]):
        #print(img_data[:,:,:,i].shape)
        p = np.multiply(img_data[:,:,:,i], mask)
        pf = p.flatten()
        #print(pf)
        #print(pf.shape)
        pfn = pf[np.logical_not(np.isnan(pf))]
        #print(pfn)
        brain_data.append(pfn)
    return np.array(brain_data)

l = load_img_data("/Users/arikbarenboim/Documents/ProcessModelsPython/AllDataCon0004.nii.gz")
m = load_img_data("/Users/arikbarenboim/Documents/ProcessModelsPython/mask.nii.gz")


c = create_brain_data(l,m)

df = pd.read_csv("/Users/arikbarenboim/Documents/ProcessModelsPython/data.csv")

print(df.head())

#Yi=cXi+ei

Y = df.values[:, -1]

x = c[:,2]

print(x)
n,m = c.shape

start_time = time.time()

lm = linear_model.SGDRegressor()
for i in range(m):
    lm.partial_fit(c[:,i].reshape(-1, 1),Y)

print(time.time() - start_time)

print(lm.coef_, lm.intercept_)
#print(Y)

#X = np.random.choice(c[0], 3081).reshape((39, -1))

#lm = linear_model.LinearRegression().fit(X, Y)

#print(lm.coef_)

#c = create_brain_data(l,m)



#fig = plt.figure()
#x = fig.add_subplot()

#ax.imshow(l[:,:,30,3],cmap='gray', interpolation=None)

#plt.scatter(X,Y)

#plt.show()

#fig, ax = plt.subplots(num="MRI_demo")
#ax.imshow(nii_data[:,:,0,0], cmap="gray")
#ax.axis('off')

#plt.scatter(x, Y)
#plt.show()