""" Evaluation
    Main script for evaluating the model trained by tcl_classification.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pickle
import tensorflow as tf
import nibabel as nib
from nilearn.masking import apply_mask

from generate_artificial_data import generate_artificial_data
from preprocessing import pca
from subfunc.showdata import *
import tcl
import tf_eval
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

FLAGS = tf.app.flags.FLAGS

# Parameters ==================================================
# =============================================================

# Data generation ---------------------------------------------
random_seed = 0 # random seed
num_subs = 2

# Training ----------------------------------------------------
initial_learning_rate = 0.01 # initial learning rate
momentum = 0.9 # momentum parameter of SGD
max_steps = int(7e5) # number of iterations (mini-batches)
decay_steps = int(5e5) # decay steps (tf.train.exponential_decay)
decay_factor = 0.1 # decay factor (tf.train.exponential_decay)
batch_size = 10 # mini-batch size
moving_average_decay = 0.9999 # moving average decay of variables to be saved
checkpoint_steps = 1e5 # interval to save checkpoint

# for MLR initialization
max_steps_init = int(7e4) # number of iterations (mini-batches) for initializing only MLR
decay_steps_init = int(5e4) # decay steps for initializing only MLR


# eval_dir = './storage/L5_Ns128'
eval_dir = './storage_tst_3/temp5'
parmpath = os.path.join(eval_dir, 'parm.pkl')

apply_fastICA = True
nonlinearity_to_source = 'abs' # Assume that sources are generated from laplacian distribution

# =============================================================
# =============================================================

# Load trained file -------------------------------------------
ckpt = tf.train.get_checkpoint_state(eval_dir)
modelpath = ckpt.model_checkpoint_path

# Load parameter file
with open(parmpath, 'rb') as f:
    model_parm = pickle.load(f)

num_comp = model_parm['num_comp']
num_segment = model_parm['num_segment']
num_segmentdata = model_parm['num_segmentdata']
num_layer = model_parm['num_layer']
list_hidden_nodes = model_parm['list_hidden_nodes']
moving_average_decay = model_parm['moving_average_decay']
random_seed = model_parm['random_seed']
pca_parm = model_parm['pca_parm']
num_subs = model_parm['num_subs']


#our raw nii files
#parent_directory = '/export/mialab/users/nlewis/'


#schizophrenic or not labels
#sub_labels = np.loadtxt(parent_directory+"/fbirn_unsmoothed/labels.txt")


#Not all files in the data set are "correct"
#f = open(parent_directory+'/fbirn_unsmoothed/corrected_names.csv', 'r')
#file = f.readlines()
#f.close()
#corrected_names = [line.rstrip('\n') for line in file]


mask = nib.load("/export/mialab/users/nlewis/HCP_ZFU/Fixed_MNI152_T1_3mm_brain_mask.nii.gz")

raw_labels = []
raw_labels+=[0]*162

raw = []

fil = '/export/mialab/users/nlewis/HCP_ZFU/101410.nii'
img = nib.load(fil)
img_data = apply_mask(img,mask)
pca = PCA(n_components=num_comp)
s = pca.fit_transform(img_data)
img_shape = img_data.shape
x=s[5:,:]
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
            print(img_masked.shape)
            pca = PCA(n_components=num_comp)
            s = pca.fit_transform(img_masked)
            print(s.shape)
            x = np.concatenate((x,s),axis=1)
            raw_labels += [i]*162
            raw.append(img_masked)

'''

labels = []
for i in range(num_segment):
    labels += [i]*num_segmentdata
labels = np.array(labels)
source = x
print("source shape")
print(source.shape)

# Generate sensor signal --------------------------------------
#sensor, source, label = generate_artificial_data(num_comp=num_comp,
#                                                 num_segment=num_segment,
#                                                 num_segmentdata=num_segmentdata,
#                                                 num_layer=num_layer,
#                                                 random_seed=random_seed)

# Preprocessing -----------------------------------------------
sensor = source.T
pca = PCA(n_components=50)
pca.fit(source)
comp_compare = pca.transform(source).T

# Evaluate model ----------------------------------------------
with tf.Graph().as_default() as g:

    data_holder = tf.placeholder(tf.float64, shape=[None, sensor.shape[0]], name='data')
    label_holder = tf.placeholder(tf.int32, shape=[None], name='label')

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, feats = tcl.inference(data_holder, list_hidden_nodes, num_class=num_segment)

    # Calculate predictions.
    top_value, preds = tf.nn.top_k(logits, k=1, name='preds')

    # Restore the moving averaged version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)

        tensor_val = tf_eval.get_tensor(sensor, [preds, feats], sess, data_holder, batch=256)
        pred_val = tensor_val[0].reshape(-1)
        feat_val = tensor_val[1]


# Calculate accuracy ------------------------------------------
accuracy, confmat = tf_eval.calc_accuracy(pred_val, labels)
print(type(feat_val))
print(feat_val.shape)
print("ICA-ing")
np.savetxt("components.txt",feat_val)

# Apply fastICA -----------------------------------------------
if apply_fastICA:
    ica = FastICA(random_state=random_seed,max_iter=4000)
    feat_val = ica.fit_transform(feat_val)

print("finished ICA")
print(feat_val.shape)
# Evaluate ----------------------------------------------------
if nonlinearity_to_source == 'abs':
    xseval = np.abs(comp_compare) # Original source
else:
    raise ValueError
feateval = feat_val.T # Estimated feature
#
print(feateval.shape)
showtimedata(feateval)
np.savetxt("comps.txt",feateval,delimiter=',')
corrmat, sort_idx, _ = tf_eval.correlation(feateval, xseval, 'Pearson')
showmat(corrmat)
meanabscorr = np.mean(np.abs(np.diag(corrmat)))

print(feateval.shape)
# Display results ---------------------------------------------
print("Result...")
print("    accuracy(train) : {0:7.4f} [%]".format(accuracy*100))
print("    correlation     : {0:7.4f}".format(meanabscorr))

print("done.")


