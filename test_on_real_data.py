# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:37:43 2020

@author: rfuchs
"""

import os 
os.chdir('C:/Users/rfuchs/Documents/GitHub/GLLVM_layer')

import warnings 
warnings.filterwarnings("ignore") # Attention..!!!!!!!!!!!!!!!!!!!!!!!!!

import matplotlib.pyplot as plt
from sklearn import manifold
import umap
import seaborn as sns

import pandas as pd
from sklearn.preprocessing import LabelEncoder 

from init_params import init_params, init_cv
from misc import misc
from gllvm_block import gllvm

import autograd.numpy as np

warnings.filterwarnings("error") # Attention..!!!!!!!!!!!!!!!!!!!!!!!!


###############################################################################################
################# Portuguese bank account subsample vizualisation #############################
###############################################################################################
os.chdir('C:/Users/rfuchs/Documents/These/Stats/binary_dgmm/datasets')

# Importing and selecting data
bank = pd.read_csv('bank/bank.csv', sep = ';')
bank = bank[['age', 'education', 'default', 'housing', 'loan', 'campaign', 'previous']]
bank = bank.infer_objects()

# Encoding the labels
## Manually encode education
edu = bank['education'].values
edu = np.where(edu == 'unknown', 0, edu)
edu = np.where(edu == 'primary', 1, edu)
edu = np.where(edu == 'secondary', 2, edu)
edu = np.where(edu == 'tertiary', 3, edu)
bank['education'] = edu

## Automatically encode the other
le = LabelEncoder()
for colname in bank.columns:
    if bank[colname].dtype != np.int64:
        bank.loc[:,colname] = le.fit_transform(bank[colname])

# Defining the parameters of the algorithm
r = 2
numobs = len(bank)
M = 100
k = 3

seed = 1
init_seed = 2
    
var_distrib = np.array(["binomial","ordinal","bernoulli","bernoulli", 'bernoulli', 'binomial', 'binomial'])

nj = []
nj_bin = []
nj_ord = []
for i in range(len(bank.columns)):
    if np.logical_or(var_distrib[i] == 'bernoulli',var_distrib[i] == 'binomial'): 
        max_nj = np.max(bank.iloc[:,i], axis = 0)
        nj.append(max_nj)
        nj_bin.append(max_nj)
    else:
        card_nj = len(np.unique(bank.iloc[:,i], axis = 0))
        nj.append(card_nj)
        nj_ord.append(card_nj)

nj = np.array(nj)
nj_bin = np.array(nj_bin)
nj_ord = np.array(nj_ord)


init = init_params(r, nj_bin, nj_ord, k, init_seed)

eps = 1E-05
it = 30
maxstep = 100

# Launching the algorithm
nb_trials = 30
miscs = np.zeros(nb_trials)
y = bank.values


#random_init = init_params(r, nj_bin, nj_ord, k, init_seed)
random_init = init_params(r, nj_bin, nj_ord, k, None)

# Random init
#random_init = init_cv(y, var_distrib, r, nj_bin, nj_ord, k, seed)
out = gllvm(y, numobs, r, k, it, random_init, eps, maxstep, var_distrib, nj, M, seed)

# UMAP init
umap_init = dim_reduce_init(y, k, r, var_distrib, seed = None)
out = gllvm(y, numobs, r, k, it, umap_init, eps, maxstep, var_distrib, nj, M, seed)

# t-SNE init
tsne_init = dim_reduce_init(y, k, r, var_distrib, umap = False, seed = None)
out = gllvm(y, numobs, r, k, it, tsne_init, eps, maxstep, var_distrib, nj, M, seed)


# Representing the learned clustering with tsne   
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
y_tsne = tsne.fit_transform(y)

df = pd.DataFrame(y_tsne, columns = ["tsne-2d-one", 'tsne-2d-two'])
df['pred_classes'] = out['classes']

flatui = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="pred_classes",
    palette=sns.color_palette(sns.xkcd_palette(flatui), len(np.unique(out['classes']))),
    data=df,
    legend="full",
    alpha=0.3
)

# Representing the learned clustering with UMAP   
reducer = umap.UMAP()
embedding = reducer.fit_transform(y)
df_umap = pd.DataFrame(embedding, columns = ["umap-2d-one", 'umap-2d-two'])
df_umap['pred_classes'] = out['classes']

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="umap-2d-one", y="umap-2d-two",
    hue="pred_classes",
    palette=sns.color_palette(sns.xkcd_palette(flatui), len(np.unique(out['classes']))),
    data=df_umap,
    legend="full",
    alpha=0.3
)



