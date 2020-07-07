# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:37:43 2020

@author: rfuchs
"""

import os 
os.chdir('C:/Users/rfuchs/Documents/GitHub/GLLVM_layer')

import warnings 
warnings.filterwarnings("ignore") # Attention..!!!!!!!!!!!!!!!!!!!!!!!!!

import autograd.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import prince

from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder

from copy import deepcopy
from glmlvm import glmlvm
from init_params import init_params, dim_reduce_init
from utils import misc, gen_categ_as_bin_dataset, \
        ordinal_encoding, plot_gmm_init, compute_nj, cluster_purity
        
warnings.filterwarnings("error") # Attention..!!!!!!!!!!!!!!!!!!!!!!!!


###############################################################################################
#################            Breast cancer vizualisation          #############################
###############################################################################################

#===========================================#
# Importing and droping NaNs
#===========================================#
os.chdir('C:/Users/rfuchs/Documents/These/Stats/mixed_dgmm/datasets')

br = pd.read_csv('breast_cancer/breast.csv', sep = ',', header = None)
y = br.iloc[:,1:]
labels = br.iloc[:,0]

y = y.infer_objects()

# Droping missing values
labels = labels[y.iloc[:,4] != '?']
y = y[y.iloc[:,4] != '?']

labels = labels[y.iloc[:,7] != '?']
y = y[y.iloc[:,7] != '?']
y = y.reset_index(drop = True)

enc = OneHotEncoder(sparse = False, drop = 'first')
labels_oh = enc.fit_transform(np.array(labels).reshape(-1,1)).flatten()


#===========================================#
# Formating the data
#===========================================#
var_distrib = np.array(['ordinal', 'ordinal', 'ordinal', 'ordinal', \
                        'bernoulli', 'ordinal', 'categorical',
                        'categorical', 'bernoulli'])
    
ord_idx = np.where(var_distrib == 'ordinal')[0]

all_labels = [np.unique(y.iloc[:,idx]) for idx in ord_idx]
all_labels[1] = ['premeno', 'lt40', 'ge40']

all_codes = [list(range(len(lab))) for lab in all_labels]    

# Encode ordinal data
for i, idx in enumerate(ord_idx):
    y.iloc[:,idx] = ordinal_encoding(y.iloc[:,idx], all_labels[i], all_codes[i])

# Encode categorical datas
y, var_distrib = gen_categ_as_bin_dataset(y, var_distrib)

# Encode binary data
le = LabelEncoder()
for colname in y.columns:
    if y[colname].dtype != np.int64:
        y[colname] = le.fit_transform(y[colname])
 
nj, nj_bin, nj_ord = compute_nj(y, var_distrib)
y_np = y.values

#===========================================#
# Running the algorithm
#===========================================# 

# Launching the algorithm
r = 1
numobs = len(y)
M = r * 4
k = 2

seed = 1
init_seed = 2
    
eps = 1E-05
it = 30
maxstep = 100

# Prince init
prince_init = dim_reduce_init(y, k, r, nj, var_distrib, dim_red_method = 'prince', seed = None)
out = glmlvm(y_np, r, k, prince_init, var_distrib, nj, M, it, eps, maxstep, seed)
m, pred = misc(labels_oh, out['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))

# Random init
random_init = init_params(r, nj_bin, nj_ord, k, None)
out = glmlvm(y_np, r, k, random_init, var_distrib, nj, M, it, eps, maxstep, seed)
m, pred = misc(labels_oh, out['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))


#=======================================================================
# Performance measure : Finding the best specification for init and DDGMM
#=======================================================================

res_folder = 'C:/Users/rfuchs/Documents/These/Experiences/mixed_algos/breast'

# GLMLVM
# Best one: r = 1 for prince and for random
numobs = len(y)
k = 2
    
eps = 1E-05
it = 30
maxstep = 100

nb_trials= 30
glmlvm_res = pd.DataFrame(columns = ['it_id', 'r', 'micro', 'macro', 'purity'])


for r in range(1, 6):
    print(r)
    M = r * 4
    for i in range(nb_trials):
        # Prince init
        #prince_init = dim_reduce_init(y, k, r, nj, var_distrib, dim_red_method = 'prince', seed = None)
        random_init = init_params(r, nj_bin, nj_ord, k, None)

        try:
            out = glmlvm(y_np, r, k, random_init, var_distrib, nj, M, it, eps, maxstep, seed = None)
            m, pred = misc(labels_oh, out['classes'], True) 
            cm = confusion_matrix(labels_oh, pred)
            purity = cluster_purity(cm)
                
            micro = precision_score(labels_oh, pred, average = 'micro')
            macro = precision_score(labels_oh, pred, average = 'macro')
            #print(micro)
            #print(macro)
        
            glmlvm_res = glmlvm_res.append({'it_id': i + 1, 'r': str(r), 'micro': micro, 'macro': macro, \
                                            'purity': purity}, ignore_index=True)
        except:
            glmlvm_res = glmlvm_res.append({'it_id': i + 1, 'r': str(r), 'micro': np.nan, 'macro': np.nan, \
                                            'purity': np.nan}, ignore_index=True)
       
glmlvm_res.groupby('r').mean()
glmlvm_res.groupby('r').std()

#glmlvm_res.to_csv(res_folder + '/glmlvm_res.csv')
glmlvm_res.to_csv(res_folder + '/glmlvm_res_random.csv')
