# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:45:45 2020

@author: Utilisateur
"""

import autograd.numpy as np
import os

os.chdir('C:/Users/rfuchs/Documents/GitHub/GLLVM_layer')

from init_params import init_params, init_cv
from misc import misc
from gllvm_block import gllvm
from datagen_functions import gen_mvdata, gen_mv_z, gen_z, gen_data

numobs = 100
seed = None

    
##############################################################################
# Debugging part
##############################################################################

r = 4
p1 = 3
numobs = 2000
M = 200
k = 3

seed = 1
init_seed = 2

nj= np.array([1,1,10,5])
nj_bin = np.array([1,1,10])
nj_ord = np.array([5])
init = init_params(r, nj_bin, nj_ord, k, init_seed)
var_distrib = np.array(["bernoulli","bernoulli","binomial","ordinal"])


eps = 1E-05
it = 30
maxstep = 100

z, labels = gen_mv_z(numobs, r, init, seed)
y = gen_mvdata(z, init, seed = None)

nb_trials = 30
miscs = np.zeros(nb_trials)

for i in range(nb_trials):
    
    #random_init = init_params(r, nj_bin, nj_ord, k, init_seed)
    random_init = init_cv(y, var_distrib, r, nj_bin, nj_ord, k, seed)
    out = gllvm(y, numobs, r, k, it, random_init, eps, maxstep, var_distrib, nj, M, seed)
    
    miscs[i] = misc(labels, out['classes'])
    print(miscs[i])
    
    
miscsM10 = miscs

miscs.mean()
miscsM10.mean()

out['mu']
init['mu']

out['sigma']
init['sigma']

out["lambda_bin"]
init["lambda_bin"]

out["lambda_ord"]
init["lambda_ord"]

