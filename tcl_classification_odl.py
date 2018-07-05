""" Classification
    Main script for the simulation described in Hyvarinen and Morioka, NIPS 2016.

    Perform time-contrastive learning from artificial data.
    Source signals are generated based on segment-wise-modulated Laplace distribution (q = |.|).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import shutil
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA

from generate_artificial_data import generate_artificial_data, apply_MLP_to_source
#from preprocessing import pca
from tcl_train import train
import nibabel as nib
from nilearn.masking import apply_mask
import scipy.io as sio

# Parameters ==================================================
# =============================================================

# Data generation ---------------------------------------------
random_seed = 0 # random seed
num_comp = 20 # number of components (dimension)
num_subs = 1
num_segmentdata = 603 # number of data-points in each segment
num_segment = 100 # number of segnents
num_layer = 5 # number of layers of mixing-MLP

# MLP ---------------------------------------------------------
list_hidden_nodes = [300, 350, 250, 125, 20]
# list of the number of nodes of each hidden layer of feature-MLP
# [layer1, layer2, ..., layer(num_layer)]

# Training ----------------------------------------------------
initial_learning_rate = 0.0001 # initial learning rate
momentum = 0.9 # momentum parameter of SGD
max_steps = int(6e5) # number of iterations (mini-batches)
decay_steps = int(5e5) # decay steps (tf.train.exponential_decay)
decay_factor = 0.1 # decay factor (tf.train.exponential_decay)
batch_size = 50 # mini-batch size
moving_average_decay = 0.9999 # moving average decay of variables to be saved
checkpoint_steps = 1e5 # interval to save checkpoint

# for MLR initialization
max_steps_init = int(7e3) # number of iterations (mini-batches) for initializing only MLR
decay_steps_init = int(6e3) # decay steps for initializing only MLR

# Other -------------------------------------------------------
# # Note: save folder must be under ./storage
train_dir = './storage_tst_3/temp5' # save directory
saveparmpath = os.path.join(train_dir, 'parm.pkl') # file name to save parameters


# =============================================================
# =============================================================

# Prepare save folder -----------------------------------------
if train_dir.find("./storage_tst_3/") > -1:
    if os.path.exists(train_dir):
        print("delete savefolder: {0:s}...".format(train_dir))
        shutil.rmtree(train_dir)  # Remove folder
    print("make savefolder: {0:s}...".format(train_dir))
    os.makedirs(train_dir)  # Make folder
else:
    assert False, "savefolder looks wrong"

#our raw nii files
#parent_directory = '/export/mialab/users/tderamus/tools/TCLbeta/data/'


#schizophrenic or not labels
#sub_labels = np.loadtxt(parent_directory+"/fbirn_unsmoothed/labels.txt")


#Not all files in the data set are "correct"
#f = open(parent_directory+'/fbirn_unsmoothed/corrected_names.csv', 'r')
#file = f.readlines()
#f.close()
#corrected_names = [line.rstrip('\n') for line in file]

'''
DEAR THOMAS: A brief symposium on data shape
    This code is littered with transposes because no one can agree on proper shapes.
We read in the data as time x components, and then, as per usual pythonic code, we run PCA
on the dataset of time x components. However, TCL requires the data be in component x time
format.  So we have to transpose in order for that to happen.

Also, for brevity, I read in some subjects (num_of_subs) without regard to class (SZ vs Healthy) and
treat each subject as a segment.  So, I think there is a better way to do this, but that's where you 
come in :)
'''

mask = nib.load("fbirnp3_restMask.nii")
#mask = nib.load("/export/mialab/users/nlewis/ica_tf/data/fbirn_unsmoothed/fbirnp3_restMask.nii")

'''
Numpy-ness:  Due to numpy arrays, we can't concatenate to an empty array.  So, we initialize the array
to be the first subject in our list.  You'll notice the for loop is indexed at 1, instead of 0
'''
raw_labels = []
raw_labels+=[0]*162

raw = []
'''
fil = "/export/mialab/users/nlewis/TCL/old/ica_tf/data/fbirn_unsmoothed/despikedData/dwa_000303008407_0002.nii"
img = nib.load(fil)
img_data = apply_mask(img,mask)
pca = PCA(n_components=num_comp)
s = pca.fit_transform(img_data)
x=s
'''
fil = '/export/mialab/users/nlewis/TCL/old/ica_tf/tcl_mri/fbirn_unsmoothed/despikedData/dwa_000306518979_0002.nii'
fil = 'dwa_000306518979_0002.nii'
img = nib.load(fil)
img_data = apply_mask(img,mask)
pca = PCA(n_components=num_comp)
s = pca.fit_transform(img_data)
img_shape = img_data.shape
x=img_data[:,3:]
x = x/np.max(x)
raw.append(x)
print(x.shape)

'''
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
            pca = PCA(n_components=num_comp)
            s = pca.fit_transform(img_masked)
            print(s.shape)
            x = np.concatenate((x,s),axis=1)
            np.savetxt("individual_"+str(i)+".csv",s.T,delimiter=',')
            raw_labels += [i]*162
            raw.append(img_masked)



'''
source = x
np.savetxt("total_input.csv",source,delimiter=',')
print("source shape")
print(source.shape)
labels = []
for i in range(num_segment):
    labels += [i]*num_segmentdata
print(len(labels))
print(source.shape)
print(num_segment)
print(max(labels))
print(labels[-1])
print(np.sum(np.isnan(source)))
print(np.max(source))
print(np.min(source))
# Preprocessing -----------------------------------------------
labels = np.array(labels)

model_parm = {'random_seed':random_seed,
              'num_comp':num_comp,
              'num_segment':num_segment,
              'num_segmentdata':num_segmentdata,
              'num_layer':num_layer,
              'list_hidden_nodes':list_hidden_nodes,
              'moving_average_decay':moving_average_decay,
              'num_subs':num_subs,
              'pca_parm':[]}
print("training")
# Train model (only MLR) --------------------------------------

train(source,
      labels,
      num_class = num_segment,
      list_hidden_nodes = list_hidden_nodes,
      initial_learning_rate = initial_learning_rate,
      momentum = momentum,
      max_steps = max_steps_init, # For init
      decay_steps = decay_steps_init, # For init
      decay_factor = decay_factor,
      batch_size = batch_size,
      train_dir = train_dir,
      checkpoint_steps = checkpoint_steps,
      moving_average_decay = moving_average_decay,
      MLP_trainable = False, # For init
      save_file='model_init.ckpt', # For init
      random_seed = random_seed)

init_model_path = os.path.join(train_dir, 'model_init.ckpt')

# Train model -------------------------------------------------
train(source,
      labels,
      num_class = num_segment,
      list_hidden_nodes = list_hidden_nodes,
      initial_learning_rate = initial_learning_rate,
      momentum = momentum,
      max_steps = max_steps,
      decay_steps = decay_steps,
      decay_factor = decay_factor,
      batch_size = batch_size,
      train_dir = train_dir,
      checkpoint_steps = checkpoint_steps,
      moving_average_decay = moving_average_decay,
      load_file=init_model_path,
      random_seed = random_seed)


# Save parameters necessary for evaluation --------------------
model_parm = {'random_seed':random_seed,
              'num_comp':num_comp,
              'num_segment':num_segment,
              'num_segmentdata':num_segmentdata,
              'num_layer':num_layer,
              'list_hidden_nodes':list_hidden_nodes,
              'moving_average_decay':moving_average_decay,
              'num_subs':num_subs,
              'pca_parm':[]}



print("Save parameters...")
with open(saveparmpath, 'wb') as f:
    pickle.dump(model_parm, f, pickle.HIGHEST_PROTOCOL)
print("done.")

