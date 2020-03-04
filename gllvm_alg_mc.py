# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 08:55:21 2020

@author: Utilisateur
"""

import numpy as np
from numpy.random import multivariate_normal
from numpy.linalg import cholesky, pinv
from scipy.special import binom
from scipy.stats import norm
from copy import deepcopy
from scipy.optimize import minimize
from lik_functions import binom_lik_opt, categ_lik_opt
from lik_gradients import binom_gr_lik_opt
from sklearn.preprocessing import OneHotEncoder


def gllvm_alg_mc_pilot(y, numobs, r, k, p, p1, p2, it, o, szo, init, eps, lik, maxstep, 
                             var_distrib, nj, ps_y, M, seed): 
    ''' Add maxstep !!! 
    o1 inutile '''
    tol = 0.01
    
    # Initialize the parameters
    mu = init['mu']
    sigma = init['sigma']
    alpha = init['alpha']
    alphaor = init['alphaor']
    thr = init['thr']
    w = init['w']
    
    likelihood = []
    hh = 0
    ratio = 1000
    np.random.seed = seed
      
    py_s = np.zeros(shape =(numobs, k))
    ps_y = np.zeros(shape =(numobs, k))
      
    E_z_sy = np.zeros(shape =(numobs, k, r))
    E_zz_sy = np.zeros(shape =(numobs, k, r, r))
      
    zM = np.zeros(shape =(M, r, k))
    
    p_z_ys = np.zeros(shape =(M, numobs, k))
            
    # Generate the gaussians at the beginining
    # Initiate the points zM from the prior f(z_1 | s_i  = 1, Theta)
    if r == 1:
        zM = multivariate_normal(size = (M, 1), mean = mu.flatten(), cov = np.diag(cholesky(sigma).flatten())) 

    else:
        raise RuntimeError('Not implemented for the moment')
        #rmvnorm(M, mean = mu[i], sigma = sigma[i]^(0_5))
    
    # Debugging use existing zM
    #zM = np.genfromtxt('../asta_Rcode_compMCGH/zM.csv', delimiter = ',', skip_header = True)[:,1:]
    #zM = zM.reshape(M, 1, k)
    
    #new_zM_tot = np.genfromtxt('../asta_Rcode_compMCGH/new_zM.csv', delimiter = ',', skip_header = True)[:,1:]             
    #new_zM_tot = new_zM_tot.reshape(M, numobs, r, k)
    
    while ((hh < it) & (ratio > eps)):
        hh = hh + 1
        for i in range(k):
            py_zM = np.zeros(shape =(M, numobs))
            co = -1 # Delete this ugly co later on
            
            for j in range(p):
                
                if (var_distrib[j] == "binomial" or var_distrib[j] == "bernoulli"):
                    zi_star = np.hstack((np.ones((zM[:,:,i].shape[0], 1)), zM[:,:,i]))
                    eta = np.repeat(zi_star @ alpha[j,:][...,np.newaxis], axis = 1, repeats = numobs) # Might do this at the end ? 
                    pi_greco = np.exp(eta)/(1 + np.exp(eta))
                  
                    yg = np.repeat(y[:, j][np.newaxis], axis = 0, repeats = M)  
                    py_zM = py_zM + np.log(binom(nj[j], yg)) + np.log(np.power(pi_greco,yg)) +\
                                    np.log(np.power(1 - pi_greco, nj[j] - yg))
                    
                    if np.isnan(py_zM).any():
                        raise RuntimeError('Nan in py_zM')
                        #py_zM = np.nan_to_num(py_zM, copy=True, nan=0.0)
                    
                
                if var_distrib[j] == "ordinal": # Problem here : zeros in log to correct
                    co = co + 1
                    if r == 1: 
                        alphaor = alphaor.reshape(p2, r)
                        thr = np.reshape(thr, (r, szo))
                    
                    gamma_prev_s = None
                    for s in range(nj[j]): 
                        # Min Hack to remove
                        gamma_s = np.repeat(thr[co,min(s, nj[j] - 2)]-zM[:,:,i] @ alphaor[co,:][...,np.newaxis], axis = 1, repeats = numobs)
                        
                        if s == 0:
                            pi_greco_ord = np.exp(gamma_s)/(1 + np.exp(gamma_s))
                            
                        if s > 0 & s < nj[j] - 1:
                            pi_greco_ord = np.exp(gamma_s)/(1 + np.exp(gamma_s)) - np.exp(gamma_prev_s)/(1 + np.exp(gamma_prev_s))
                        
                        if s == nj[j] - 1:
                            pi_greco_ord = 1 - np.exp(gamma_prev_s)/(1 + np.exp(gamma_prev_s))
                            
                        # [!] s + 1 beacause the first class is numeroted 1 and not 0     
                        yg_s = np.repeat((y[:, j] == s + 1)[np.newaxis], axis = 0, repeats = M)
                        
                        py_zM = py_zM + np.log(np.power(pi_greco_ord,yg_s))
                        
                        if np.isnan(py_zM).any():
                            raise RuntimeError('Nan in py_zM')     
                        
                        gamma_prev_s = deepcopy(gamma_s)
                
            py_zM = np.exp(py_zM)
            py_zM[py_zM == 0] = 1e-50
            qM = py_zM / np.sum(py_zM, axis = 0, keepdims = True)
        
            # Resampling according to qM (replace later with an apply)
            new_zM = np.zeros((M,numobs))
            for obs in range(numobs):
                new_zM[:,obs] = np.random.choice(zM[:,0,i], M, p = qM[:,obs], replace = True)

            # For debuging to remove
            # Check for r > 1
            #new_zM = new_zM_tot[:,:,i]
            #print(new_zM.sum())
            new_zM = new_zM.reshape(M, numobs, r)

            # Determing p(zM | s, theta)_  
            if r==1:
                pz_s = norm.pdf(zM[:,:,i], loc = mu[i], scale = cholesky(sigma[i]))
            else:
                raise RuntimeError('pz_s not implemented for r>1 for the moment ')   
                #pz_s = matrix(dmvnorm(zM[,,i], mean = mu[i,], sigma[i,]^0_5), M) # To check for multivariate, add cholesky
              
            pz_s_norm = pz_s / np.sum(pz_s, axis = 0, keepdims = True) # To check
            
            # Compute (17): p(y | s= 1)_  Keep the prod accross dimensions ?
            py_s[:, i] = np.sum(np.repeat(np.prod(pz_s_norm, axis = 1, keepdims = True),\
                axis = 1, repeats = numobs) * py_zM, axis = 0) 
        
            # Compute (16) p(z |y, s) 
            p_z_ys[:, :, i] = (np.repeat(np.prod(pz_s, axis = 1, keepdims = True),\
                axis = 1, repeats = numobs) * py_zM) / py_s[:,i]
            
            # Compute unormalized (18)
            ps_y[:, i] = w[i] * py_s[:,i]  
              
            if r == 1:
                E_z_sy[:, i, :] = np.mean(new_zM, axis = 0, keepdims = True)
                E_zz_sy[:,i,:,:] = np.mean(new_zM * new_zM , axis = 0)[...,np.newaxis] 
            else:
                raise RuntimeError('E_z_sy not implemented for r>1 for the moment ')   

        # Normalize ps_y in order to obtain (18)
        ps_y = ps_y / np.sum(ps_y, axis = 1, keepdims = True)        
        p_y = py_s @ w
        
        Ez_y = np.zeros((numobs, r))
        for i in range(k):
            Ez_y = Ez_y + ps_y[:, i].reshape(numobs, r) * E_z_sy[:, i, :]
        
        # Normalizing p(z|y,s)
        for i in range(k): # Check axis for summation...
            p_z_ys[:,:,i] = p_z_ys[:,:,i] / np.sum(p_z_ys[:,:,i], axis = 0, keepdims = True)  
                
        # Begining of the M-step 
        w = np.mean(ps_y, axis = 0)
        
        # Temp so dirty...
        temp1 = np.zeros((r, r))
        temp2 = np.zeros((r, r))
        temp3 = np.zeros(r)
        for i in range(k):
            den = sum(ps_y[:, i]) 
            den = den if den > 0 else 1e-14
            
            mu[i,:] = np.sum(ps_y[:, i].reshape(numobs, r) * E_z_sy[:, i, :], axis = 0)/den
            sigma[i, :, :] = np.sum(ps_y[:, i].reshape(numobs, r, r) * (E_zz_sy[:, i, :, :] - \
                    np.repeat((mu[i, :] @ mu[i, :].T).reshape(1, r, r), axis = 0, repeats = numobs)), axis = (0,1))/den
            
            temp1 = temp1 + w[i] * sigma[i, :, :]
            temp2 = temp2 + w[i] * (mu[i, :] @ mu[i, :].T)
            temp3 = temp3 + w[i] * mu[i, :]
        
        
        # Compute alpha    
        co = -1
        #start = proc_time()
        for j in range(p):
            if var_distrib[j] == "bernoulli" or var_distrib[j] == "binomial":
                # Add initial guess and lim iterations
                opt = minimize(binom_lik_opt, alpha[j,:], args = (y[:,j], zM, k, ps_y, p_z_ys, nj[j]), 
                   tol = tol, method='BFGS', jac = binom_gr_lik_opt)
                alpha[j, :] = opt.x  
                      
            if var_distrib[j] == "ordinal":
                co = co + 1
                theta = np.concatenate([thr[co, :nj[j] - 1], alphaor[co,:]])
                o1 = nj[j]
                
                enc = OneHotEncoder()
                y_oh = enc.fit_transform(y[:,j][..., np.newaxis]).toarray() 
                opt = minimize(categ_lik_opt, theta, args = (y_oh, zM, k, o1, ps_y, p_z_ys), 
                   tol = tol, method='Nelder-Mead') 
                theta = opt.x 
                
                thr[co, :(o1 - 1)] = theta[:(o1 - 1)]
                alphaor[co, :] = theta[o1 - 1:(len(theta) + 1)]
            
                
        # Identifiability
        var_z = temp1 + temp2 - temp3 @ temp3.T
        
        A = cholesky(var_z)
        for i in range(k):
            sigma[i,: , :] = pinv(A).T @ sigma[i, :, :] @ pinv(A)
            mu[i, :] = pinv(A).T @ mu[i, :]
        
        mu_tot = w.T @ mu
        mu = mu - mu_tot
        
        if r > 1:
            alpha_tot = np.hstack([alpha[:, -1], alphaor])
        else: # To finish
            alpha_tot = np.concatenate([alpha[:, -1], alphaor[0]])[...,np.newaxis]
        alpha_tot = alpha_tot @ A.T
        
        if r > 1: 
            for i in range(1,r):
                alpha_tot[:i - 1, i] = 0
        if p1 > 0: 
            alpha[:, -1] = alpha_tot[:p1, ].flatten()
        if p2 > 0: 
            alphaor = alpha_tot[p1:p, ]
        
        temp = np.sum(np.log(p_y))
        likelihood.append(temp)
        ratio = (temp - lik)/abs(lik)
        if (hh < 10): 
            ratio = 2 * eps
        lik = temp
        print(hh)
      
        
    # To recheck 
    classes = np.argmax(ps_y, axis = 1) + 1 # +1 to match the group labels that start at 1
    
    out = dict(alpha = alpha, alphaor = alphaor, thr = thr, \
                w = w, mu = mu, sigma = sigma, likelihood = likelihood, \
                ps_y = ps_y, py_s = py_s, Ez_y = Ez_y, py = p_y, classes = classes)
    return(out)
