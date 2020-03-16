# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:55:44 2020

@author: Utilisateur
"""

import autograd.numpy as np
from autograd.numpy.random import uniform
#import numpy as np
#from numpy.random import uniform
from gllvm_block import gllvm

def init_params(r, nj_bin, nj_ord, k, init_seed):
    ''' Generate random initialisations for the parameters
    Consider no regressors here'''
    
    # Seed for init
    np.random.seed = init_seed
    init = {}
    max_nj_ord = np.max(nj_ord)
    
    p1 = len(nj_bin)
    p2 = len(nj_ord)
    
    if p1 > 0:
        init['lambda_bin'] = uniform(low = -3, high = 3, size = (p1, r + 1))
  
    if p2 > 0:
        lambda0_ord = np.zeros(shape = (p2, max_nj_ord - 1)) # Why not max_nj_ord - 1
        for j in range(p2):
            lambda0_ord[j, :nj_ord[j] - 1] = np.sort(uniform(low = -2, high = 2, size = (1, nj_ord[j] - 1))) #.reshape(1, szo, r)
  
        Lambda_ord = uniform(low = -3, high = 3, size = (p2, r))
        init['lambda_ord'] = np.hstack([lambda0_ord, Lambda_ord])
  
    if (r > 1): 
        init['lambda_bin'] = np.tril(init['lambda_bin'], k = 1)

    init['w'] = np.full(k, 1/k).reshape(k,1) # Check for transpose ? 
    
    # Maybe -5, 0, 5 was better ?
    mu_init = np.repeat([[-1], [0], [1]], axis = 1, repeats = r)
    init['mu'] = (uniform(low = -5, high = 5, size = (1,1)) * mu_init)
  
    # Will have to define diag matrix
    init['sigma'] = np.zeros(shape = (k, r, r))
    for i in range(k):
        init['sigma'][i,: , :] = 0.50 * np.eye(r)# Too low ?
  
    return(init)

def init_cv(y,  k, var_distrib, r, nj, seed = None):
    ''' Test 20 different inits for a few iterations and returns the best one'''
    
    numobs = y.shape[0]
    p = y.shape[1]
    p2 = sum(np.array(var_distrib) == "ordinal")
    p1 = p - p2
    o = nj[var_distrib == "ordinal"][0] # Dirty hack to remove when several ordinal variables
    szo = o # Dirty hack to remove when several ordinal variables
  
    nb_init_tested = 5
    M = 300
    best_lik = -1000000
    best_init = {}
    nb_it = 3
    maxstep = 0
    eps = 1E-5

    for i in range(nb_init_tested):
        init = init_params(r, p, p1, p2, o, szo, k, init_seed = None)
        out_pilot_mc = gllvm(y, numobs, r, k, p, p1, p2, nb_it, o, szo, init, eps, maxstep, 
                             var_distrib, nj, M, seed)

        lik = out_pilot_mc['likelihood'][-1]
    
        if (best_lik < lik):
            best_lik = lik
            best_init = init
    
    return(best_init)

#from factor_analyzer import FactorAnalyzer
#help(FactorAnalyzer)
#FactorAnalyzer(y) 
