# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:52:28 2020

@author: Utilisateur
"""

import autograd.numpy as np

from autograd import grad
from autograd.numpy import newaxis as n_axis
from autograd.numpy import expand_dims as exp_dim
from autograd.numpy.random import multivariate_normal
from autograd.numpy.linalg import cholesky, pinv


#import numpy as np 
#from numpy import newaxis as n_axis
#from numpy import expand_dims as exp_dim
from scipy.linalg import block_diag
#from numpy.random import multivariate_normal
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import multivariate_normal as mvnorm
#from numpy.linalg import cholesky, pinv
 
from lik_functions import compute_py_zM_bin, compute_py_zM_ord
from lik_gradients import binom_gr_lik_opt, ord_gr_lik, ord_autograd

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
    #fake_ord = np.random.sample()
    
    # Simulate pseudo-observations
    zM = multivariate_normal(size = (M, 1), mean = mu.flatten(order = 'F'), cov = block_diag(*sigma)) 
    zM = np.transpose(zM.reshape(M, k, r, order = 'F'), (0, 2, 1))
    
    while ((hh < it) & (ratio > eps)):
        hh = hh + 1
        ###### Compute the p(y| zM) for all variable categories ########
        # First the Binomial data
        py_zM_bin = compute_py_zM_bin(lambda_bin, y_bin, zM, k, nj_bin) # shape = (M,k,numobs)
        
        # Then the categorical variables
        # One hot encoding, is this better than a loop ? List comprehension ?
        
        # A regler !!! Pour des variables ordinales ayant des tailles différentes ont fait comment ?
        enc = OneHotEncoder(categories='auto')
        y_oh = []
        for j in range(len(nj_ord)):
            y_oh.append(enc.fit_transform(y_ord[:,j][..., n_axis]).toarray())
        y_oh = np.stack(y_oh)
        
        py_zM_ord = compute_py_zM_ord(lambda_ord, y_oh, zM, k, nj_ord)
        py_zM_new = py_zM_bin + py_zM_ord #/ 2
        
        py_zM_new = np.exp(py_zM_new)
        py_zM_new = np.where(py_zM_new <= 1E-20, 1E-20, py_zM_new)
        
        #####################################################################################
        ############################ E step #################################################
        #####################################################################################
        
        # Resample zM conditionally on y 
        qM_new = py_zM_new / np.sum(py_zM_new, axis = 0, keepdims = True)
        new_zM = np.zeros((M,numobs, r, k))
        
        new_zM = np.zeros((M, numobs, r, k))
        for i in range(k):
            qM_cum = qM_new[:,:, i].T.cumsum(axis=1)
            u = np.random.rand(numobs, 1, M)
            
            choices = u < qM_cum[..., np.newaxis]
            idx = choices.argmax(1)
            
            new_zM[:,:,:,i] = np.take(zM[:,:, i], idx.T, axis=0)
        
        del(u)
        
        pz_s_new = np.zeros((M, 1, k))
                
        for i in range(k): # Have to retake the function for DGMM to parallelize or use apply along axis
            pz_s_new[:,:, i] = mvnorm.pdf(zM[:,:,i], mean = mu[i], cov = sigma[i])[..., n_axis]
                
        # Compute (17) p(y | s_i = 1)
        pz_s_norm_new = pz_s_new / np.sum(pz_s_new, axis = 0, keepdims = True) 
        py_s_new = (pz_s_norm_new * py_zM_new).sum(axis = 0)
        
        # Compute (16) p(z |y, s) 
        p_z_ys_new = py_s_new * py_zM_new / py_s_new[n_axis]
        
        # Free some memory
        del(py_zM_new)
        del(pz_s_norm_new)
        del(pz_s_new)
        del(qM_new)
        del(y_oh)
        
        # Compute unormalized (18)
        ps_y_new = w.T * py_s_new
        ps_y_new = ps_y_new / np.sum(ps_y_new, axis = 1, keepdims = True)        
        p_y_new = py_s_new @ w
        
        # Compute E_{y,s}(z) and E_{y,s}(zTz)
        E_z_sy_new = np.transpose(np.mean(new_zM, axis = 0), (0, 2, 1)) # Remove transpose later on
        zTz_new = (np.transpose(new_zM[...,n_axis], (0, 1, 3, 2, 4)) @ \
                   np.transpose(new_zM[...,n_axis], (0, 1, 3, 4, 2)))
        E_zz_sy_new = np.mean(zTz_new, axis = 0)
        
        # Compute E_{y}(z) and E_{y}(zTz)
        #Ez_y_new = (ps_y_new[...,n_axis] * E_z_sy_new).sum(1)
        
        del(new_zM)
            
        # Normalizing p(z|y,s)
        p_z_ys_new = p_z_ys_new / np.sum(p_z_ys_new, axis = 0, keepdims = True)
                
        # Enforce identifiability conditions
        w_new = np.mean(ps_y_new, axis = 0)
        den_new = ps_y_new.sum(0, keepdims = True).T
        den_new = np.where(den_new < 1E-14, 1E-14, den_new)
        
        mu_new = (ps_y_new[...,n_axis] * E_z_sy_new).sum(0) / den_new
        muTmu = mu_new[..., n_axis] @ exp_dim(mu_new, 1)
        sigma_new = np.sum(ps_y_new[..., n_axis, n_axis] * (E_zz_sy_new - \
                    muTmu[n_axis]), axis = 0) / den_new[..., n_axis]
        
        mu_var = (w_new[..., n_axis, n_axis] * (sigma_new + muTmu)).sum(0)
        w_mu = (w_new[..., n_axis] * mu_new).sum(0, keepdims = True)
        
        var_z_new = mu_var - w_mu.T @ w_mu
        sigma_z = cholesky(var_z_new)
        
        sigma_new = pinv(sigma_z).T[n_axis] @ sigma_new @ pinv(sigma_z)[n_axis]
        mu_new = mu_new @ pinv(sigma_z)
        
        mu_tot_new = w_new.T @ mu_new
        mu_new = mu_new - mu_tot_new
        
        del(E_z_sy_new)
        del(E_zz_sy_new)
        
        ###########################################################################
        ############################ M step #######################################
        ###########################################################################
        
        # We optimize each column separately as it is faster than all column jointly 
        # (and more relevant with the independence hypothesis)
        
        for j in range(nb_bin):
            # Add initial guess and lim iterations
            opt = minimize(binom_lik_opt, lambda_bin[j,:], args = (y_bin[:,j], zM, k, ps_y_new, p_z_ys_new, nj_bin[j]), 
                           tol = tol, method='BFGS', jac = binom_gr_lik_opt)
                
            if not(opt.success):
                    print('Binomial optimization failed')
                    
            lambda_bin[j, :] = deepcopy(opt.x)  

        col_nb = 0        
        for j in range(nb_ord):
            enc = OneHotEncoder(categories='auto')
            y_oh = enc.fit_transform(y_ord[:,j][..., n_axis]).toarray()                
            
            nb_constraints = nj_ord[col_nb] - 2
            np_params = lambda_ord[col_nb].shape[0]
            lcs = np.full(nb_constraints, -1)
            lcs = np.diag(lcs, 1)
            np.fill_diagonal(lcs, 1)
        
            lcs = np.hstack([lcs[:nb_constraints, :], np.zeros([nb_constraints, np_params - (nb_constraints + 1)])])
            
            linear_constraint = LinearConstraint(lcs, np.full(nb_constraints, -np.inf), \
                                                 np.full(nb_constraints, 0), keep_feasible = True)
            
            warnings.filterwarnings("default")
        
            opt = minimize(ord_lik_opt, lambda_ord[col_nb] , args = (y_oh, zM, k, nj_ord[col_nb], ps_y_new, p_z_ys_new), 
                               tol = tol, method='trust-constr',  jac = ord_autograd, \
                               constraints = linear_constraint, hess = '2-point')
            
            if opt.status != 2:
                print('Categorical optimization failed')
            lambda_ord[col_nb] = deepcopy(opt.x) 
            
            col_nb += 1
            
        # Last identifiability part
        lambda_bin = np.tril(lambda_bin, k = 1)
        lambda_bin[:,1:] = lambda_bin[:,1:] @ sigma_z.T
        lambda_ord[:, max_nj_ord - 1 :] = lambda_ord[:, max_nj_ord - 1 :] @ sigma_z.T
        
        new_lik = np.sum(np.log(p_y_new))
        likelihood.append(new_lik)
        ratio = (new_lik - prev_lik)/abs(prev_lik)
        
        if (hh < 10): 
            ratio = 2 * eps
        prev_lik = new_lik
        print(hh)
        print(likelihood)
        
    classes = np.argmax(ps_y_new, axis = 1) 

    out = dict(lambda_bin = lambda_bin, lambda_ord = lambda_ord, \
                w = w, mu = mu, sigma = sigma, likelihood = likelihood, \
                classes = classes)
    return(out)

