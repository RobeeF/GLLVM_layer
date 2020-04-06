# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:32:07 2020

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
from utils import misc, ordinal_encoding, plot_gmm_init,\
    compute_nj, performance_testing

warnings.filterwarnings("error") # Attention..!!!!!!!!!!!!!!!!!!!!!!!!


###############################################################################################
#################        Clustering on the car dataset (UCI)          #########################
###############################################################################################

#===========================================#
# Importing and droping NaNs
#===========================================#
os.chdir('C:/Users/rfuchs/Documents/These/Stats/binary_dgmm/datasets')

# Importing and selecting data
car = pd.read_csv('car/car.csv', sep = ',', header = None)
car = car.infer_objects()

y = car.iloc[:,:-1]

le = LabelEncoder()
labels = car.iloc[:,-1]
labels_oh = le.fit_transform(labels)
k = len(np.unique(labels_oh))

#===========================================#
# Formating the data
#===========================================#

var_distrib = np.array(['ordinal'] * y.shape[1])

ord_idx = np.where(var_distrib == 'ordinal')[0]

# Extract labels for each y_j and then perform dirty manual reordering
all_labels = [np.unique(y.iloc[:,idx]) for idx in ord_idx]
all_labels[0] = ['low', 'med', 'high', 'vhigh']
all_labels[1] = ['low', 'med', 'high', 'vhigh']
all_labels[4] = ['small', 'med', 'big']
all_labels[5] = ['low', 'med', 'high']

all_codes = [list(range(len(lab))) for lab in all_labels]    

# Encode ordinal data
for i, idx in enumerate(ord_idx):
    y.iloc[:,idx] = ordinal_encoding(y.iloc[:,idx], all_labels[i], all_codes[i])

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
plt.show()

mca.eigenvalues_
mca.explained_inertia_ # Very bad


# Visualisation of MCA + GMM 
np.random.seed(2)

gmm = GaussianMixture(n_components = k, covariance_type='full').fit(y_mca)
init_preds = gmm.predict(y_mca)
plot_gmm_init(y_mca.values, init_preds, gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')

misc(labels_oh, init_preds)
confusion_matrix(labels_oh, init_preds)

#===========================================#
# Running the algorithm
#===========================================# 

nj, nj_bin, nj_ord = compute_nj(y, var_distrib)
y_np = y.values

# Launching the algorithm
r = 2
numobs = len(y)
M = r * 4

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

#=============================================
# Performance measure
#=============================================

#res_prince = performance_testing(y, labels_oh, k, 'prince', var_distrib, nj, r_max = 7, seed = None)
res_prince = pd.read_csv('car/prince_30runs.csv')

# Analysis of Prince init
res_prince[['r', 'micro', 'macro']].boxplot(by = 'r', figsize = (20, 10))
res_prince[['r','run_time']].boxplot(by = 'r', figsize = (20, 10))
res_prince[['r','nb_iterations']].boxplot(by = 'r', figsize = (20, 10))

# Plot the percentage of launches that failed
res_prince = res_prince.set_index('r')
NaNs_per_r = res_prince.isna().astype(int).groupby('r').mean()
plt.plot('r', 'macro', data=NaNs_per_r.reset_index()) # No fails either

res_prince.to_csv('car/prince_30runs.csv')


# Analysis of random init
#res_random = performance_testing(y, labels_oh, k, 'random', var_distrib, nj, r_max = 7, seed = None)
res_random = pd.read_csv('car/random_30runs.csv')

res_random[['r', 'micro', 'macro']].boxplot(by = 'r', figsize = (20, 10))
res_random[['r','run_time']].boxplot(by = 'r', figsize = (20, 10))
res_random[['r','nb_iterations']].boxplot(by = 'r', figsize = (20, 10))


# Plot the percentage of launches that failed
res_random = res_random.set_index('r')
NaNs_per_r = res_random.isna().astype(int).groupby('r').mean()
plt.plot('r', 'macro', data=NaNs_per_r.reset_index()) # No fails either

res_random.to_csv('car/random_30runs.csv')
