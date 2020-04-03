# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:33:34 2020

@author: rfuchs
"""

import os 
os.chdir('C:/Users/rfuchs/Documents/GitHub/GLLVM_layer')

import warnings 
warnings.filterwarnings("ignore") # Attention..!!!!!!!!!!!!!!!!!!!!!!!!!

import prince
import pandas as pd
import seaborn as sns
import autograd.numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture

from copy import deepcopy

from glmlvm import glmlvm
from init_params import init_params, dim_reduce_init
from utils import misc, gen_categ_as_bin_dataset, \
        ordinal_encoding, plot_gmm_init, compute_nj

warnings.filterwarnings("error") # Attention..!!!!!!!!!!!!!!!!!!!!!!!!

###############################################################################################
##############        Clustering on the Mushrooms dataset (UCI)          ######################
###############################################################################################

#===========================================#
# Importing and droping NaNs
#===========================================#
os.chdir('C:/Users/rfuchs/Documents/These/Stats/binary_dgmm/datasets')

# Importing and selecting data
mush = pd.read_csv('mushrooms/agaricus-lepiota.csv', sep = ',', header = None)
mush = mush.infer_objects()

y = mush.iloc[:,1:]

le = LabelEncoder()
labels = mush.iloc[:,0]
labels_oh = le.fit_transform(labels)

#Delete missing data
missing_idx = y.iloc[:, 10] != '?'
y = y[missing_idx]

labels = labels[missing_idx]
labels_oh = labels_oh[missing_idx]
k = len(np.unique(labels_oh))


#===========================================#
# Formating the data
#===========================================#

var_distrib = np.array(['categorical', 'categorical', 'categorical', 'bernoulli', 'categorical',\
                        'categorical', 'categorical', 'bernoulli', 'categorical', 'categorical',\
                        'categorical', 'categorical', 'categorical', 'categorical', 'categorical', \
                        'categorical', 'categorical', 'ordinal', 'categorical', 'categorical', \
                        'categorical', 'categorical'])

ord_idx = np.where(var_distrib == 'ordinal')[0]

# Extract labels for each y_j and then perform dirty manual reordering
all_labels = [np.unique(y.iloc[:,idx]) for idx in ord_idx]
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
print('Explained inertia is', mca.explained_inertia_) # Far better than the other ones

# Visualisation of MCA + GMM 
np.random.seed(2)

gmm = GaussianMixture(n_components = k, covariance_type='full').fit(y_mca)
init_preds = gmm.predict(y_mca)
plot_gmm_init(y_mca.values, init_preds, gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')


m, init_preds = misc(labels_oh, init_preds, True) 
print(m)
print(confusion_matrix(labels_oh, init_preds))

#===========================================#
# Running the algorithm
#===========================================# 

nj, nj_bin, nj_ord = compute_nj(y, var_distrib)
y_np = y.values

# Launching the algorithm
r = 4
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

