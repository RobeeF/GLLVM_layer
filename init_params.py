# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:55:44 2020

@author: Utilisateur
"""

import autograd.numpy as np
from autograd.numpy.random import uniform

from gllvm_block import gllvm
from autograd.numpy import newaxis as n_axis
from autograd.numpy import transpose as t

from autograd.numpy.linalg import cholesky, pinv

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

    init['w'] = np.full(k, 1/k) 
    
    mu_init = np.repeat([[-1], [0], [1]], axis = 1, repeats = r)
    init['mu'] = (uniform(low = -5, high = 5, size = (1,1)) * mu_init)
    init['mu'] = init['mu'][..., np.newaxis]
  
    # Will have to define diag matrix
    init['sigma'] = np.zeros(shape = (k, r, r))
    for i in range(k):
        init['sigma'][i,: , :] = 0.050 * np.eye(r)
        
    # Enforcing identifiability constraints

    muTmu = init['mu'] @ t(init['mu'], (0,2,1))  
     
    E_zzT = (init['w'][..., n_axis, n_axis] * (init['sigma'] + muTmu)).sum(0, keepdims = True)
    Ezz_T = (init['w'][...,n_axis, n_axis] * init['mu']).sum(0, keepdims = True)
    
    # A vérifier
    var_z = E_zzT - Ezz_T @ t(Ezz_T, (0,2,1)) # Koenig-Huyghens Formula for Variance Computation
    sigma_z = cholesky(var_z)
     
    init['sigma'] = pinv(sigma_z) @ init['sigma'] @ t(pinv(sigma_z), (0, 2, 1))
    init['mu'] = pinv(sigma_z) @ init['mu']
    init['mu']  = init['mu']  - Ezz_T
   
    return(init)


def init_cv(y, var_distrib, r, nj_bin, nj_ord, k, seed):
    ''' Test 20 different inits for a few iterations and returns the best one'''
    
    numobs = y.shape[0]
  
    nb_init_tested = 10
    M = 20
    best_lik = -1000000
    best_init = {}
    nb_it = 2
    maxstep = 100
    eps = 1E-5
    nj = np.concatenate([nj_bin, nj_ord])

    for i in range(nb_init_tested):
        init = init_params(r, nj_bin, nj_ord, k, None)
        try:
            out = gllvm(y, numobs, r, k, nb_it, init, eps, maxstep, var_distrib, nj, M, seed)
        except:
            continue

        lik = out['likelihood'][-1]
    
        if (best_lik < lik):
            best_lik = lik
            best_init = init
    
    return(best_init)

#from factor_analyzer import FactorAnalyzer
#help(FactorAnalyzer)
#FactorAnalyzer(y) 
