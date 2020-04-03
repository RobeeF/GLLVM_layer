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

from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder

from copy import deepcopy
from glmlvm import glmlvm
from init_params import init_params, dim_reduce_init
from utils import misc, gen_categ_as_bin_dataset, \
        ordinal_encoding, plot_gmm_init, compute_nj
        
warnings.filterwarnings("error") # Attention..!!!!!!!!!!!!!!!!!!!!!!!!


###############################################################################################
#################            Breast cancer vizualisation          #############################
###############################################################################################

#===========================================#
# Importing and droping NaNs
#===========================================#
os.chdir('C:/Users/rfuchs/Documents/These/Stats/binary_dgmm/datasets')

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
        
#===========================================#
# Exploratory analysis
#===========================================#  

mca = prince.MCA(n_components=2, n_iter=3, copy=True,check_input=True, engine='auto', random_state=42)
mca = mca.fit(y)
y_mca = mca.row_coordinates(y)


df = pd.DataFrame(deepcopy(y_mca))
df.columns = ["mca0", 'mca1']
df['classes'] = labels

flatui = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="mca0", y="mca1",
    hue="classes",
    palette=sns.color_palette(sns.xkcd_palette(flatui), len(np.unique(labels))),
    data=df,
    legend="full",
    alpha=0.3
)

mca.eigenvalues_
mca.explained_inertia_ # A little better than for Portuguese banks

# Visualisation of MCA + GMM 
np.random.seed(2)

gmm = GaussianMixture(n_components=2, covariance_type='full').fit(y_mca)
init_preds = gmm.predict(y_mca)
plot_gmm_init(y_mca.values, init_preds, gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')

enc = OneHotEncoder(sparse = False, drop = 'first')
labels_oh = enc.fit_transform(np.array(labels).reshape(-1,1)).flatten()

misc(labels_oh, init_preds)
confusion_matrix(labels_oh, init_preds)

#===========================================#
# Running the algorithm
#===========================================# 

nj, nj_bin, nj_ord = compute_nj(y, var_distrib)
y_np = y.values

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
out = glmlvm(y_np, r, k, it, prince_init, eps, maxstep, var_distrib, nj, M, seed)
m, pred = misc(labels_oh, out['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))

# Random init
random_init = init_params(r, nj_bin, nj_ord, k, None)
out = glmlvm(y_np, r, k, it, random_init, eps, maxstep, var_distrib, nj, M, seed)
m, pred = misc(labels_oh, out['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))
