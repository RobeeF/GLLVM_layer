# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:52:28 2020

@author: Utilisateur
"""

import autograd.numpy as np

from autograd.numpy import newaxis as n_axis
from autograd.numpy import transpose as t

from autograd.numpy.random import multivariate_normal
from autograd.numpy.linalg import cholesky, pinv


from scipy.linalg import block_diag
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import multivariate_normal as mvnorm
 
from lik_functions import compute_py_zM_bin, compute_py_zM_ord
from lik_gradients import ord_autograd, bin_autograd

from copy import deepcopy
from scipy.optimize import minimize
from lik_functions import binom_lik_opt, ord_lik_opt
from scipy.optimize import LinearConstraint

import warnings
warnings.filterwarnings("error")


def gllvm(y, numobs, r, k, it, init, eps, maxstep, var_distrib, nj, M, seed): 
    ''' Add maxstep '''

    prev_lik = - 100000
    tol = 0.01
    
    # Initialize the parameters
    mu = deepcopy(init['mu'])
    sigma = deepcopy(init['sigma'])
    lambda_bin = deepcopy(init['lambda_bin'])
    lambda_ord = deepcopy(init['lambda_ord'])
    w = deepcopy(init['w'])
    
    likelihood = []
    hh = 0
    ratio = 1000
    np.random.seed = seed
    
    #M2 = 10**r
    likelihood = []
    
    # Dispatch variables between categories
    y_bin = y[:, np.logical_or(var_distrib == 'bernoulli',var_distrib == 'binomial')]
    nj_bin = nj[np.logical_or(var_distrib == 'bernoulli',var_distrib == 'binomial')]
    nb_bin = len(nj_bin)
    
    y_ord = y[:, var_distrib == 'ordinal']    
    nj_ord = nj[var_distrib == 'ordinal']
    nb_ord = len(nj_ord)
    max_nj_ord = max(nj_ord)
                     
    while ((hh < it) & (ratio > eps)):
        hh = hh + 1
        
        # Simulate pseudo-observations
        zM = multivariate_normal(size = (M, 1), mean = mu.flatten(order = 'F'), cov = block_diag(*sigma)) 
        zM = t(zM.reshape(M, k, r, order = 'F'), (0, 2, 1))
        
        ###### Compute the p(y| zM) for all variable categories ########
        
        # First the Binomial data
        py_zM_bin = compute_py_zM_bin(lambda_bin, y_bin, zM, k, nj_bin) # shape = (M,k,numobs)
                
        # Then the categorical variables        
        # A regler !!! Pour des variables ordinales ayant des tailles différentes ont fait comment ?
        enc = OneHotEncoder(categories='auto')
        y_oh = []
        for j in range(len(nj_ord)):
            y_oh.append(enc.fit_transform(y_ord[:,j][..., n_axis]).toarray())
        y_oh = np.stack(y_oh)
                
        py_zM_ord = compute_py_zM_ord(lambda_ord, y_oh, zM, k, nj_ord) 
        py_zM = py_zM_bin + py_zM_ord 
        
        py_zM = np.exp(py_zM)
        
        #####################################################################################
        ############################ E step #################################################
        #####################################################################################
        
        # Resample zM conditionally on y 
        qM = py_zM / np.sum(py_zM, axis = 0, keepdims = True)
        new_zM = np.zeros((M,numobs, r, k))
        
        new_zM = np.zeros((M, numobs, r, k))
        for i in range(k):
            qM_cum = qM[:,:, i].T.cumsum(axis=1)
            u = np.random.rand(numobs, 1, M)
            
            choices = u < qM_cum[..., np.newaxis]
            idx = choices.argmax(1)
            
            new_zM[:,:,:,i] = np.take(zM[:,:, i], idx.T, axis=0)
        
        del(u)
        
        pz_s = np.zeros((M, 1, k))
                
        for i in range(k): # Have to retake the function for DGMM to parallelize or use apply along axis
            pz_s[:,:, i] = mvnorm.pdf(zM[:,:,i], mean = mu[i].flatten(), cov = sigma[i])[..., n_axis]
                
        # Compute (17) p(y | s_i = 1)
        pz_s_norm = pz_s / np.sum(pz_s, axis = 0, keepdims = True) 
        py_s = (pz_s_norm * py_zM).sum(axis = 0)
        
        # Compute (16) p(z |y, s) 
        p_z_ys = py_s * py_zM / py_s[n_axis]
        
        # Free some memory
        del(py_zM)
        del(pz_s_norm)
        del(pz_s)
        del(qM)
        del(y_oh)
        
        # Compute unormalized (18)
        ps_y = w[n_axis] * py_s
        ps_y = ps_y / np.sum(ps_y, axis = 1, keepdims = True)        
        p_y = py_s @ w
        
        # Compute E_{y,s}(z) and E_{y,s}(zTz)
        E_z_sy = t(np.mean(new_zM, axis = 0), (0, 2, 1)) # Remove transpose later on
        zTz = (t(new_zM[...,n_axis], (0, 1, 3, 2, 4)) @ \
                   t(new_zM[...,n_axis], (0, 1, 3, 4, 2)))
        E_zz_sy = np.mean(zTz, axis = 0)
        
        # Compute E_y(z) and E_y(zTz)
        Ez_y = (ps_y[...,n_axis] * E_z_sy).sum(1)
                
        del(new_zM)
            
        # Normalizing p(z|y,s)
        p_z_ys = p_z_ys / np.sum(p_z_ys, axis = 0, keepdims = True)
                
        # Computing Gaussian Parameters
        w = np.mean(ps_y, axis = 0)
        den = ps_y.sum(0, keepdims = True).T[..., n_axis]
        den = np.where(den < 1E-14, 1E-14, den)
        
        mu = (ps_y[...,n_axis] * E_z_sy).sum(0)[..., np.newaxis] / den

        muTmu = mu @ t(mu, (0,2,1))  
        sigma = np.sum(ps_y[..., n_axis, n_axis] * (E_zz_sy - \
                    muTmu[n_axis]), axis = 0) / den
         
        # Enforcing identifiability constraints
        E_zzT = (w[..., n_axis, n_axis] * (sigma + muTmu)).sum(0, keepdims = True)
        Ezz_T = (w[...,n_axis, n_axis] * mu).sum(0, keepdims = True)

        var_z = E_zzT - Ezz_T @ t(Ezz_T, (0,2,1)) # Koenig-Huyghens Formula for Variance Computation
        sigma_z = cholesky(var_z)
        
        sigma = pinv(sigma_z) @ sigma @ t(pinv(sigma_z), (0, 2, 1))
        mu = pinv(sigma_z) @ mu
        mu  = mu  - Ezz_T
        
        del(E_z_sy)
        del(E_zz_sy)
         
        
        ###########################################################################
        ############################ M step #######################################
        ###########################################################################
        
        # We optimize each column separately as it is faster than all column jointly 
        # (and more relevant with the independence hypothesis)
        
        for j in range(nb_bin):
            # Add initial guess and lim iterations
            opt = minimize(binom_lik_opt, lambda_bin[j,:], args = (y_bin[:,j], zM, k, ps_y, p_z_ys, nj_bin[j]), 
                           tol = tol, method='BFGS', jac = bin_autograd, options = {'maxiter': maxstep})
                
            if not(opt.success):
                raise RuntimeError('Binomial optimization failed')
                    
            lambda_bin[j, :] = deepcopy(opt.x)  

        for j in range(nb_ord):
            enc = OneHotEncoder(categories='auto')
            y_oh = enc.fit_transform(y_ord[:,j][..., n_axis]).toarray()                
            
            nb_constraints = nj_ord[j] - 2
            np_params = lambda_ord[j].shape[0]
            lcs = np.full(nb_constraints, -1)
            lcs = np.diag(lcs, 1)
            np.fill_diagonal(lcs, 1)
        
            lcs = np.hstack([lcs[:nb_constraints, :], np.zeros([nb_constraints, np_params - (nb_constraints + 1)])])
            
            linear_constraint = LinearConstraint(lcs, np.full(nb_constraints, -np.inf), \
                                                 np.full(nb_constraints, 0), keep_feasible = True)
            
            warnings.filterwarnings("default")
        
            opt = minimize(ord_lik_opt, lambda_ord[j] , args = (y_oh, zM, k, nj_ord[j], ps_y, p_z_ys), 
                               tol = tol, method='trust-constr',  jac = ord_autograd, \
                               constraints = linear_constraint, hess = '2-point', options = {'maxiter': maxstep})
            
            if not(opt.success):
                raise RuntimeError('Categorical optimization failed')
                print(opt)
            lambda_ord[j] = deepcopy(opt.x) 
            
                        
        # Last identifiability part
        lambda_bin = np.tril(lambda_bin, k = 1)
        lambda_bin[:,1:] = lambda_bin[:,1:] @ sigma_z[0] 
        lambda_ord[:, max_nj_ord - 1 :] = lambda_ord[:, max_nj_ord - 1 :] @ sigma_z[0]
        
        new_lik = np.sum(np.log(p_y))
        likelihood.append(new_lik)
        ratio = (new_lik - prev_lik)/abs(prev_lik)
        
        if (hh < 3): 
            ratio = 2 * eps
        prev_lik = new_lik
        print(hh)
        print(likelihood)
        
    classes = np.argmax(ps_y, axis = 1) 

    out = dict(lambda_bin = lambda_bin, lambda_ord = lambda_ord, \
                w = w, mu = mu, sigma = sigma, likelihood = likelihood, \
                classes = classes)
    return(out)
