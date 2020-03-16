# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 08:55:21 2020

@author: Utilisateur
"""

import numpy as np
from numpy.random import multivariate_normal
from numpy.linalg import cholesky, pinv
from scipy.special import binom
from scipy.stats import norm, multivariate_normal as mvnorm
from copy import deepcopy
from scipy.optimize import minimize
from lik_functions import binom_lik_opt, ord_lik_opt
from lik_gradients import binom_gr_lik_opt, categ_gr_lik
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import LinearConstraint
from resample_zM import sample_MC_points
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("error")

def gllvm_alg_mc_pilot(y, numobs, r, k, p, p1, p2, it, o, szo, init, eps, maxstep, 
                             var_distrib, nj, M, seed): 
    ''' Add maxstep !!! 
    o1 inutile '''
    lik = - 100000
    tol = 0.01
    
    # Initialize the parameters
    mu = deepcopy(init['mu'])
    sigma = deepcopy(init['sigma'])
    alpha = deepcopy(init['alpha'])
    alphaor = deepcopy(init['alphaor'])
    thr = deepcopy(init['thr'])
    w = deepcopy(init['w'])
    
    likelihood = []
    hh = 0
    ratio = 1000
    np.random.seed = seed
      
    py_s = np.zeros(shape =(numobs, k))
    ps_y = np.zeros(shape =(numobs, k))
      
    E_z_sy = np.zeros(shape =(numobs, k, r))
    E_zz_sy = np.zeros(shape =(numobs, k, r, r))
      
    M2 = 10**r
    #zM = np.zeros(shape =(M, r, k))    
    #p_z_ys = np.zeros(shape =(M, numobs, k)) 
     
    while ((hh < it) & (ratio > eps)):
        hh = hh + 1
        warnings.filterwarnings("error")
        zM = np.zeros(shape =(M, r, k))    
        p_z_ys = np.zeros(shape =(M, numobs, k))

        # Generate the gaussians at the beginining
        # Initiate the points zM from the prior f(z_1 | s_i  = 1, Theta)
        if r == 1:
            zM = multivariate_normal(size = (M, r), mean = mu.flatten(), cov = np.diag(sigma.flatten())) 
    
        else:
            for i in range(k): # To change with np.block_diag when ready
                zM[:,:,i] = multivariate_normal(size = M, mean = mu[i,:].flatten(), cov = sigma[i], check_valid = 'raise') 

        py_zM_tot =  np.zeros(shape = (M, numobs, k)) 
        for i in range(k):
            py_zM = np.zeros(shape = (M, numobs)) # Checker la place dans le code R
            co = -1 # Delete this ugly co later on
            
            for j in range(p):
                
                if (var_distrib[j] == "binomial" or var_distrib[j] == "bernoulli"):
                    zi_star = np.hstack((np.ones((zM[:,:,i].shape[0], 1)), zM[:,:,i]))
                    eta = np.repeat(zi_star @ alpha[j,:][...,np.newaxis], axis = 1, repeats = numobs) 
                    pi_greco = 1/(1 + np.exp(-eta))
                    
                    yg = np.repeat(y[:, j][np.newaxis], axis = 0, repeats = M)  
                    
                    try:
                        np.log(binom(nj[j], yg)) + np.log(np.power(pi_greco,yg)) +\
                                    np.log(np.power(1 - pi_greco, nj[j] - yg))

                    except RuntimeWarning:
                        print(alpha[j,:])
                        print(zi_star.mean())
                        print(np.exp(eta).mean())
                        print(np.exp(eta).max())

                        print(eta.mean())
                        print(pi_greco.mean())
                        raise RuntimeError('zero in py greco...')
                        
                    py_zM = py_zM + np.log(binom(nj[j], yg)) + np.log(np.power(pi_greco,yg)) +\
                                    np.log(np.power(1 - pi_greco, nj[j] - yg))
                    
                    if np.isnan(py_zM).any():
                        raise RuntimeError('Nan in py_zM')
                    
                
                if var_distrib[j] == "ordinal": # Problem here : zeros in log to correct
                    co = co + 1
                    alphaor = alphaor.reshape(p2, r)
                    thr = np.reshape(thr, (1, szo))
                        
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
                        #yg_s = np.repeat((y[:, j] == s + 1)[np.newaxis], axis = 0, repeats = M)
                        yg_s = np.repeat((y[:, j] == s)[np.newaxis], axis = 0, repeats = M)
                        
                        py_zM = py_zM + np.log(np.power(pi_greco_ord,yg_s))
                        
                        if np.isnan(py_zM).any():
                            raise RuntimeError('Nan in py_zM')     
                        
                        gamma_prev_s = deepcopy(gamma_s)
                
            py_zM_tot[:,:,i] = py_zM
            py_zM = np.exp(py_zM)
            py_zM[py_zM == 0] = 1e-50
            qM = py_zM / np.sum(py_zM, axis = 0, keepdims = True)
        
            # Resampling according to qM (replace later with an apply)
            new_zM = np.zeros((M,numobs, r))

            # New version to test for r = 1
            indices = range(0,M)
            for obs in range(numobs):
                drawn_ids = np.random.choice(indices, M, p = qM[:,obs], replace = True)
                new_zM[:,obs,:] = zM[drawn_ids,:,i]

            # Determing p(zM | s, theta)_. Should erase the first one after test 
            if r==1:
                pz_s = norm.pdf(zM[:,:,i], loc = mu[i], scale = cholesky(sigma[i]))
            else:
                #pz_s = mvnorm.pdf(zM[:,:,i], mean = mu[i], cov = cholesky(sigma[i]))
                pz_s = mvnorm.pdf(zM[:,:,i], mean = mu[i], cov = sigma[i])
                pz_s = pz_s[...,np.newaxis]
                        
            pz_s_norm = pz_s / np.sum(pz_s, axis = 0, keepdims = True) 
            
            # Compute (17): p(y | s= 1)_  Keep the prod accross dimensions ?
            py_s[:, i] = np.sum(np.repeat(pz_s_norm,\
                axis = 1, repeats = numobs) * py_zM, axis = 0) 
                              
            # Compute (16) p(z |y, s) 
            p_z_ys[:, :, i] = (np.repeat(np.prod(pz_s, axis = 1, keepdims = True),\
                axis = 1, repeats = numobs) * py_zM) / py_s[:,i]
            
            # Compute unormalized (18)
            ps_y[:, i] = w[i] * py_s[:,i]  
            
            E_z_sy[:, i, :] = np.mean(new_zM, axis = 0, keepdims = True)
 
            zTz = new_zM[...,np.newaxis] @ np.transpose(new_zM[...,np.newaxis], (0, 1 , 3, 2))
            E_zz_sy[:,i,:,:] = np.atleast_3d(np.mean(zTz, axis = 0))
              
               
        # Normalize ps_y in order to obtain (18)
        ps_y = ps_y / np.sum(ps_y, axis = 1, keepdims = True)        
        p_y = py_s @ w
        
        Ez_y = np.zeros((numobs, r))
        for i in range(k):
            Ez_y = Ez_y + ps_y[:, i][...,np.newaxis] * E_z_sy[:, i, :]
                
        # Normalizing p(z|y,s)
        p_z_ys = p_z_ys / np.sum(p_z_ys, axis = 0, keepdims = True)

        ####### New
        pz_s = np.zeros((M, 1, k))
        if r == 1:
            for i in range(k):
                pz_s[:,:, i] = norm.pdf(zM[:,:,i], loc = mu[i], scale = cholesky(sigma[i]))
        else:
            for i in range(k):
                pz_s[:,:, i] = mvnorm.pdf(zM[:,:,i], mean = mu[i], cov = sigma[i])[...,np.newaxis]
        
        true_lik = binom_lik_opt(alpha[0,:], y[:,0], zM, k, ps_y, p_z_ys, nj[0])
        print('Before', true_lik)
        
        error = []
        for i in range(1, 20):
            M2 = 20 * i 
            p_z_ys2, zM2 =  sample_MC_points(zM, p_z_ys, M2) 
            p_z_ys2 = p_z_ys2 / np.sum(p_z_ys2, axis = 0, keepdims = True)
            
            approx_lik = binom_lik_opt(alpha[0,:], y[:,0], zM2, k, ps_y, p_z_ys2, nj[0])
            #print('After', approx_lik)
            error.append(np.abs(true_lik - approx_lik)/true_lik)
         
        print(approx_lik)
        plt.plot(np.array(range(1,20)) * 20, error)
        plt.show()
        M2 = 50
        ##########
        # Begining of the M-step 
        w = np.mean(ps_y, axis = 0)
        
        # Temp so dirty...
        temp1 = np.zeros((r, r))
        temp2 = np.zeros((r, r))
        temp3 = np.zeros(r)
        
        i = 1
        for i in range(k):
            den = sum(ps_y[:, i]) 
            den = den if den > 0 else 1e-14
            
            mu[i] = np.sum(ps_y[:, i][...,np.newaxis] * E_z_sy[:, i, :], axis = 0)/den
            sigma[i] = np.sum(ps_y[:, i][...,np.newaxis, np.newaxis] * (E_zz_sy[:, i, :, :] - \
                    (mu[i][np.newaxis].T @ mu[i][np.newaxis])[np.newaxis]), axis = 0)/den
            
            temp1 = temp1 + w[i] * sigma[i]
            temp2 = temp2 + w[i] * (mu[i][np.newaxis].T @ mu[i][np.newaxis])
            temp3 = temp3 + w[i] * mu[i]
        
        # Compute alpha    
        co = -1
        #start = proc_time()
        for j in range(p):
            if var_distrib[j] == "bernoulli" or var_distrib[j] == "binomial":
                # Add initial guess and lim iterations
                opt = minimize(binom_lik_opt, alpha[j,:], args = (y[:,j], zM, k, ps_y, p_z_ys, nj[j]), 
                   tol = tol, method='BFGS', jac = binom_gr_lik_opt)
                
                #print('Binomial')
                #print(opt.success)
                #print(opt.x)
                if not(opt.success):
                    print('Binomial optimization failed')
                    
                alpha[j, :] = deepcopy(opt.x)  
                      
            if var_distrib[j] == "ordinal":
                co = co + 1
                theta = np.concatenate([thr[co, :nj[j] - 1], alphaor[co,:]])
                o1 = nj[j]
                
                enc = OneHotEncoder(categories='auto')
                y_oh = enc.fit_transform(y[:,j][..., np.newaxis]).toarray()                
                
                nb_constraints = thr.shape[1] - 2
                np_params = theta.shape[0]
                lcs = np.full(nb_constraints, -1)
                lcs = np.diag(lcs, 1)
                np.fill_diagonal(lcs, 1)

                lcs = np.hstack([lcs[:nb_constraints, :], np.zeros([nb_constraints, np_params - (nb_constraints + 1)])])
                
                linear_constraint = LinearConstraint(lcs, np.full(nb_constraints, -np.inf), \
                                                     np.full(nb_constraints, 0), keep_feasible = True)
                
                warnings.filterwarnings("default")

                opt = minimize(ord_lik_opt, theta, args = (y_oh, zM, k, o1, ps_y, p_z_ys), 
                                   tol = tol, method='trust-constr',  jac = categ_gr_lik, \
                                   constraints = linear_constraint, hess = '2-point')
                
                #print('Ordinal')
                #print(opt.status)
                #print(opt.x)
                if opt.status != 2:
                    print('Categorical optimization failed')
                theta = deepcopy(opt.x) 
                thr[co, :(o1 - 1)] = deepcopy(theta[:(o1 - 1)])
                alphaor[co, :] = deepcopy(theta[o1 - 1:(len(theta) + 1)])
            
                
        # Identifiability
        var_z = temp1 + temp2 - temp3[np.newaxis].T @ temp3[np.newaxis]
        
        A = cholesky(var_z)
        for i in range(k):
            sigma[i] = pinv(A).T @ sigma[i] @ pinv(A)
            mu[i] = pinv(A).T @ mu[i]
        
        mu_tot = w.T @ mu
        mu = mu - mu_tot
        
        if r > 1:
            alpha_tot = np.vstack([alpha[:, 1:], alphaor])
        else: # To finish
            alpha_tot = np.concatenate([alpha[:, -1], alphaor[0]])[...,np.newaxis]
        alpha_tot = alpha_tot @ A.T
        
        if r > 1: 
            for i in range(1,r):
                alpha_tot[:i, i] = 0
        if p1 > 0: 
            alpha[:, 1:] = alpha_tot[:p1, ]
        if p2 > 0: 
            alphaor = alpha_tot[p1:p, ]
        
        temp = np.sum(np.log(p_y))
        likelihood.append(temp)
        ratio = (temp - lik)/abs(lik)
        if (hh < 10): 
            ratio = 2 * eps
        lik = temp
        print(hh)
        print(likelihood)
      
        
    # To recheck 
    classes = np.argmax(ps_y, axis = 1) #+ 1 # +1 to match the group labels that start at 1
    
    out = dict(alpha = alpha, alphaor = alphaor, thr = thr, \
                w = w, mu = mu, sigma = sigma, likelihood = likelihood, \
                ps_y = ps_y, py_s = py_s, Ez_y = Ez_y, py = p_y, classes = classes)
    return(out)
