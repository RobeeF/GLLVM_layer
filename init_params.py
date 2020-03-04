# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:55:44 2020

@author: Utilisateur
"""

import numpy as np
from numpy.random import uniform
from gllvm_alg_mc import gllvm_alg_mc_pilot


nj = np.array([1,1,10,5])
modello = np.array(["bernoulli","bernoulli","binomial","ordinal"])
o = nj[modello == "ordinal"]
szo = max(o)
init_seed = None

def init_params(r, p, p1, p2, o, szo, k, init_seed):
    ''' Generate random initialisations for the parameters
    Consider no regressors here'''
    
    # Seed for init
    np.random.seed = init_seed
    init = {}
    
    if p1 > 0:
        init['alpha'] = uniform(low = -3, high = 3, size = (p1, r + 1))
  
    if p2 > 0:
        init['thr'] = np.zeros(shape = (p2, szo)) # Problem here thr should be 4 dimensional
        for j in range(p2):
            init['thr'][j, range(o)] = np.sort(uniform(low = -2, high = 2, size = (p2, szo)))#.reshape(1, szo, r)
  
    if p2 > 0:
        init['alphaor'] = uniform(low = -3, high = 3, size = (p2, r))
  
    if (r > 1):        
        init['alpha.tot'] = np.vstack([init['alpha'][:, 1:], init['alphaor']])
        for i in range(1,r): 
            init['alpha.tot'][:i, i] = 0
            if (p1 > 0): 
                init['alpha'][:, 1:] = init['alpha.tot'][:p1, :]
            if (p2 > 0): 
                init['alphaor'] = init['alpha.tot'][p1:(p + 1), :]
        
  
    init['w'] = np.full(k, 1/k).reshape(k,1) # Check for transpose ? 
    
    # Maybe -5, 0, 5 was better ?
    mu_init = np.repeat([[-1], [0], [1]], axis = 1, repeats = r)
    init['mu'] = (uniform(low = -5, high = 5, size = (1,1)) * mu_init)
  
    # Will have to define diag matrix
    init['sigma'] = np.zeros(shape = (k, r, r))
    for i in range(k):
        init['sigma'][i,: , :] = 0.50 * np.eye(r)# Too low ?
  
    return(init)


def init_cv(out, y,  k, var_distrib, r, nj, seed = None):
    ''' Test 20 different inits for a few iterations and returns the best one'''
    
    numobs = y.shape[0]
    p = y.shape[1]
    p2 = sum(np.array(modello) == "ordinal")
    p1 = p - p2
    o = nj[modello == "ordinal"]
    var_distrib = np.array(["bernoulli","bernoulli","binomial","ordinal"])

    if type(o) != list: # Dirty formating
        o = [o]
    szo = max(o)
  
    nb_init_tested = 20
    M = 300
    best_lik = -1000000
    best_init = {}
    nb_it = 3
    maxstep = 0
    eps = 1E-5

    for i in range(nb_init_tested):
        init = init_params(r, p, p1, p2, o, szo, init_seed = None)
        out_pilot_mc = gllvm_alg_mc_pilot(y, numobs, r, k, p, p1, p2, nb_it, o, szo, init, eps, maxstep, 
                             var_distrib, nj, M, seed)
        lik = out_pilot_mc['lik'][nb_it,0]
    
        if (best_lik < lik):
            best_lik = lik
            best_init = init
    
    return(best_init)

#from factor_analyzer import FactorAnalyzer
#help(FactorAnalyzer)
#FactorAnalyzer(y) 
