import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
import nibabel as nib
from nilearn.masking import apply_mask, unmask
import os

num_subs = 1


mask = nib.load("/export/mialab/users/nlewis/HCP_ZFU/Fixed_MNI152_T1_3mm_brain_mask.nii.gz")

comps = np.loadtxt("comps.txt",delimiter=',')

fil = '/export/mialab/users/nlewis/HCP_ZFU/101410.nii'
img = nib.load(fil)
img_data = apply_mask(img,mask)

img_shape = img_data.shape
x=img_data[5:,:]
'''
print(x.shape)
for i in range(1,num_subs):
    fil = parent_directory+"/fbirn_unsmoothed/despikedData/dwa_" + str(corrected_names[i])+".nii"
    print(i)
    #if the correct image is not in the folder, throw out
    if os.path.isfile(fil):
        img = nib.load(fil)
        img_data = img.get_data()
        img_shape = img_data.shape
        #if the image is missing voxels or time steps, throw out
        if img_shape[3] ==x.shape[0]:
            img_masked = apply_mask(img,mask)
            print("masked image")
            print(img_masked.shape)

            x = np.concatenate((x,img_masked),axis=1)
'''

print(x.shape)
print(comps.shape)
model = smf.OLS(x,comps.T)
lm = model.fit()
print(lm.params.shape)
subs = [lm.params]
#subs = np.split(lm.params,5,axis=1)
ave = 0
for s in subs:
    print(s.shape)
    ave += s/5
print(ave.shape)

for i in range(50):
    umask = unmask(ave[i,:],mask)
    nib.save(umask,"comp_"+str(i)+".nii")