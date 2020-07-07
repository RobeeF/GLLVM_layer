# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:25:11 2020

@author: rfuchs
"""

import os 

os.chdir('C:/Users/rfuchs/Documents/GitHub/GLLVM_layer')

from copy import deepcopy

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder

import pandas as pd

from gower import gower_matrix
from sklearn.metrics import silhouette_score


from glmlvm import glmlvm
from init_params import dim_reduce_init
from utils import gen_categ_as_bin_dataset, \
        compute_nj, misc

import autograd.numpy as np
from autograd.numpy.random import uniform

###############################################################################
###############         Heart    vizualisation          #######################
###############################################################################

#===========================================#
# Importing data
#===========================================#
os.chdir('C:/Users/rfuchs/Documents/These/Stats/mixed_dgmm/datasets')

heart = pd.read_csv('heart_statlog/heart.csv', sep = ' ', header = None)
y = heart.iloc[:,:-1]
labels = heart.iloc[:,-1]
labels = np.where(labels == 1, 0, labels)
labels = np.where(labels == 2, 1, labels)

y = y.infer_objects()
numobs = len(y)

# Too many zeros for this "continuous variable". Add a little noise to avoid 
# the correlation matrix for each group to blow up
uniform_draws = uniform(0, 1E-12, numobs)
y.iloc[:, 9] = np.where(y[9] == 0, uniform_draws, y[9])

n_clusters = len(np.unique(labels))
p = y.shape[1]

#===========================================#
# Formating the data
#===========================================#
var_distrib = np.array(['continuous', 'bernoulli', 'categorical', 'continuous',\
                        'continuous', 'bernoulli', 'categorical', 'continuous',\
                        'bernoulli', 'continuous', 'ordinal', 'ordinal',\
                        'categorical']) # Last one is ordinal for me (but real
                        # real in the data description)
    
# Ordinal data already encoded
 
y_categ_non_enc = deepcopy(y)
vd_categ_non_enc = deepcopy(var_distrib)

# Encode categorical datas
y, var_distrib = gen_categ_as_bin_dataset(y, var_distrib)

# Encode binary data
le = LabelEncoder()
for col_idx, colname in enumerate(y.columns):
    if var_distrib[col_idx] == 'bernoulli': 
        y[colname] = le.fit_transform(y[colname])
    
enc = OneHotEncoder(sparse = False, drop = 'first')
labels_oh = enc.fit_transform(np.array(labels).reshape(-1,1)).flatten()

nj, nj_bin, nj_ord = compute_nj(y, var_distrib)
y_np = y.values
nb_cont = np.sum(var_distrib == 'continuous')

p_new = y.shape[1]


# Feature category (cf)
cf_non_enc = np.logical_or(vd_categ_non_enc == 'categorical', vd_categ_non_enc == 'bernoulli')

# Non encoded version of the dataset:
y_nenc_typed = y_categ_non_enc.astype(np.object)
y_np_nenc = y_nenc_typed.values

# Defining distances over the non encoded features
dm = gower_matrix(y_nenc_typed, cat_features = cf_non_enc) 

#===========================================#
# Running the algorithm
#===========================================# 

r = 1
numobs = len(y)
k = n_clusters
M = r * 4

seed = 1
init_seed = 2
    
eps = 1E-05
it = 50
maxstep = 100

dtype = {y.columns[j]: np.float64 if (var_distrib[j] != 'bernoulli') & \
        (var_distrib[j] != 'categorical') else np.str for j in range(p_new)}

y = y.astype(dtype, copy=True)


prince_init = dim_reduce_init(y, k, r, nj, var_distrib, dim_red_method = 'prince', seed = None)
m, pred = misc(labels_oh, prince_init['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))
print(silhouette_score(dm, pred, metric = 'precomputed'))

out = glmlvm(y_np, r, k, prince_init, var_distrib, nj, M, it, eps, maxstep, None)
m, pred = misc(labels_oh, out['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))
print(silhouette_score(dm, pred, metric = 'precomputed'))