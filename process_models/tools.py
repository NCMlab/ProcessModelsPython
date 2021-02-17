#==================================
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
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
        print(img_data[:,:,:,i].shape)
        p = np.multiply(img_data[:,:,:,i], mask)
        pf = p.flatten()
        print(pf)
        print(pf.shape)
        pfn = pf[np.logical_not(np.isnan(pf))]
        print(pfn)
        brain_data.append(pfn)
    return np.array(brain_data)

l = load_img_data("/Users/arikbarenboim/Documents/ProcessModelsPython/AllDataCon0004.nii.gz")
m = load_img_data("/Users/arikbarenboim/Documents/ProcessModelsPython/mask.nii.gz")

c = create_brain_data(l,m)
np.savetxt("filename",c,newline="\n")

#fig = plt.figure()
#ax = fig.add_subplot()

#ax.imshow(nii_data[:,:,30,3],cmap='gray', interpolation=None)

#for j in range (nii_data.shape[3]):
    #for i in range(nii_data[:,:,:,j].shape[2]):
         #ax.imshow(nii_data[:,:,i,j],cmap='gray', interpolation=None)
         #plt.pause(0.03)

#plt.show()

#fig, ax = plt.subplots(num="MRI_demo")
#ax.imshow(nii_data[:,:,0,0], cmap="gray")
#ax.axis('off')


#===================================================
# number_of_slices = 3
# number_of_frames = 4

# fig, ax = plt.subplots(number_of_frames, number_of_slices,constrained_layout=True)
# fig.canvas.set_window_title('4D Nifti Image')
# fig.suptitle('4D_Nifti 10 slices 30 time Frames', fontsize=16)
# #-------------------------------------------------------------------------------
# mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()

# for slice in range(number_of_slices):
#     for frame in range(number_of_frames):
#         ax[frame, slice].imshow(nii_data[:,:,slice,frame],cmap='gray', interpolation=None)
#         ax[frame, slice].set_title("layer {} / frame {}".format(slice, frame))
#         ax[frame, slice].axis('off')

# plt.show()    