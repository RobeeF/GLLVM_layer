# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:55:44 2020

@author: Utilisateur
"""

import numpy as np
from numpy.random import multivariate_normal, uniform, binomial

nj = np.array([1,1,10,5])
modello = np.array(["bernoulli","bernoulli","binomial","ordinal"])
o = nj[modello == "ordinal"]
szo = max(o)
init_seed = None

def init_params(r, p, p1, p2, o, szo, x, init_seed):
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
            init['thr'][j, range(o[j])] = np.sort(uniform(low = -2, high = 2, size = (o[j])))
  
    if p2 > 0:
        init['alphaor'] = uniform(low = -3, high = 3, size = (p2, r))
  
    if (r > 1):
        raise RuntimeError('r > 1: Not implemented')
        '''
        init['alpha.tot'] = np.hstack([init['alpha'][:, -1], init['alphaor']])
        for i in 2:r: 
            init$alpha.tot[1:(i - 1), i] = 0
            if (p1 > 0) 
                init$alpha[, -1] = init$alpha.tot[1:p1, ]
            if (p2 > 0) 
                init$alphaor = init$alpha.tot[(p1 + 1):p, ]
        '''
  
    init['w'] = np.full(k, 1/k).reshape(k,1) # Check for transpose ?  
    init['mu'] = (uniform(low = -5, high = 5) * np.array([-1, 0, 1])).reshape(k,r)
  
    # Will have to define diag matrix
    init['sigma'] = np.zeros(shape = (k, r, r))
    for i in range(k):
        init['sigma'][i,: , :] = 0.50 * np.diag([r])# Too law
  
    return(init)



def init_cv(out, y,  k, modello, r, nj, w_veri, mu_veri, sigma_veri, thr_veri, alpha_veri, alphaor_veri = None):
    # Remove veri quantities very soon
  
    alphaor = None
    alpha = None
    numobs = y.shape[0]
    p = y.shape[1]
    p2 = sum(np.array(modello) == "ordinal")
    p1 = p - p2
    o = nj[modello == "ordinal"]
  
    if type(o) != list: # Dirty formating
        o = [o]
    szo = max(o)
    m = 0
  
    nb_init_tested = 20
    M = 30
    best_lik = -1000000
    best_init = {}
    nb_it = 3
    for i in range(nb_init_tested):
        init = init_params(r, p, p1, p2, o, szo, x, init_seed = None)
        out_pilot_mc = gllvm_fm_pilot(y, modello, k, r, x = None, it = nb_it, eps = 1e-05, \
                                  init = init, seed = None, pp = gauher(8), maxstep = 10, \
                                  nj = nj, ps_y = None, use_gauher = use_gauher, M = M, init_seed = None) 
        lik = out_pilot_mc['lik'][nb_it,0]
    
        perf.mc = perf_code(true_out= out, est_out = out_pilot_mc, w_veri, mu_veri, sigma_veri, thr_veri, alpha_veri, alphaor_veri)
        if (best_lik < lik):
            best_lik = lik
            best_init = init
    
    return(best_init)

