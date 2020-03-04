# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:33:27 2020

@author: Utilisateur
"""

import numpy as np
from scipy.special import binom

def binom_lik(alpha, y, zM, k, ps_y, p_z_ys, nj): 
    M = zM.shape[0]
    numobs = len(y)
    temp = np.zeros(numobs)
    yg = np.repeat(y[np.newaxis], axis = 0, repeats = M)
    coeff_binom = binom(nj, yg)

    for i in range(k):
        zi_star = np.hstack((np.ones((zM[:,:,i].shape[0], 1)), zM[:,:,i]))
        eta = zi_star @ alpha[:,np.newaxis]
        
        den = nj * np.log(1 + np.exp(eta))
        den = np.repeat(den, axis = 1, repeats = numobs)
        
        num = eta @ y[np.newaxis]  # To check

        log_p_y_z = num - den + np.log(coeff_binom)
        temp = temp + ps_y[:, i] * np.sum(p_z_ys[:, :, i] * log_p_y_z, axis = 0)
      
    return -np.sum(temp)

def binom_lik_opt(alpha, y, zM, k, ps_y, p_z_ys, nj): # Passer plus de chose en argument
    M = zM.shape[0]
    r = zM.shape[1]
    numobs = len(y)
    yg = np.repeat(y[np.newaxis], axis = 0, repeats = M)
    coeff_binom = binom(nj, yg).reshape(M, 1, numobs)
    
    eta = np.transpose(zM, (0, 2, 1)) @ alpha[1:].reshape(r, 1, 1)
    eta = eta + alpha[0].reshape(1, 1, 1)# Add the constant
    
    den = nj * np.log(1 + np.exp(eta))
    num = eta @ y[np.newaxis, np.newaxis]  # To check
    log_p_y_z = num - den + np.log(coeff_binom)
    
    return -np.sum(ps_y * np.sum(p_z_ys * np.transpose(log_p_y_z, (0,2,1)), axis = 0))


######################################################################
# Ordinal likelihood functions
######################################################################

def categ_lik(theta, y, zM, k, o1, ps_y, p_z_ys):
    #r = zM.shape[1]
    M = zM.shape[0]
    numobs = len(y)
    thro = theta[:(o1 - 1)]
    alphao = theta[o1 - 1:(len(theta) + 1)]
    temp = np.zeros(numobs)
    num = np.zeros((o1 - 1, M, numobs))
    den = np.zeros((o1 - 1, M, numobs))
    log_p_y_z = np.zeros((M, numobs))

    for i in range(k):
        for s in range(o1):
            if (s < o1 - 1): 
                exp_eta = np.repeat(np.exp(thro[s] - zM[:,:, i] @ alphao[...,np.newaxis]), axis = 1, \
                                    repeats =  numobs)
                num[s, :, :] = exp_eta
                den[s, :, :] = (1 + exp_eta)
                
            yg_s = np.repeat((y == s + 1)[np.newaxis], axis = 0, repeats = M)  
         
            if (s == 0): # Pourquoi pas de passage au log ici ?
                log_p_y_z = yg_s * np.log((num[s,: ,:]/den[s, :, :]))
            if (s > 0 and s < o1 - 1): 
                log_p_y_z = yg_s * np.log((num[s, :, :]/den[s, :, :] - num[s - 1, :, :]/den[s - 1, :, :]))
            
            if (s == o1 - 1): 
                log_p_y_z = yg_s * np.log((1 - num[s - 1, :, :]/den[s - 1, :, :]))
    
            temp = temp + ps_y[:, i] * np.sum(p_z_ys[:, :, i] * log_p_y_z, axis = 0)
            
    return - np.sum(temp)


def categ_lik_opt(theta, y_oh, zM, k, o1, ps_y, p_z_ys):
    r = zM.shape[1]
    M = zM.shape[0]
    thro = theta[:(o1 - 1)]
    alphao = theta[o1 - 1:(len(theta) + 1)]
 
    broad_thro = thro.reshape(o1 - 1, 1, 1, 1)
    eta = broad_thro - np.transpose(zM, (0, 2, 1)) @ alphao.reshape(r, 1, 1, 1)

    gamma = np.exp(eta) / (1 + np.exp(eta))
    gamma_prev = np.concatenate([np.zeros((1,M, k, r)), gamma])
    gamma_next = np.concatenate([gamma, np.ones((1,M, k, r))])
    pi = gamma_next - gamma_prev
    
    yg = np.expand_dims(y_oh.T, 1)[..., np.newaxis, np.newaxis] 

    log_p_y_z = yg * np.log(np.expand_dims(pi, axis=2)) # Put the newaxis elsewhere
    
    return -np.sum(ps_y * np.sum(np.expand_dims(p_z_ys[np.newaxis], axis = 4) * log_p_y_z, (0,1,4)))

   
#########################################################################################
# Test section 
#########################################################################################
'''
j = 2
binom_lik(alpha[j,:], y[:,j], zM, k, ps_y, p_z_ys, nj[j])
binom_lik_opt(alpha[j,:], y[:,j], zM, k, ps_y, p_z_ys, nj[j])

j = 3
theta = np.concatenate([thr[co, :nj[j] - 1], alphaor[co,:]])
o1 = nj[j]

categ_lik(theta, y[:,j], zM, k, o1, ps_y, p_z_ys)

# Optimizer test
from scipy.optimize import minimize

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(binom_lik, alpha[j,:], args = (y[:,j], zM, k, ps_y, p_z_ys, nj[j]))
res.x 
               #, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
help(minimize)

# For ordinal
res = minimize(categ_lik, theta, args = (y[:,j], zM, k, o1, ps_y, p_z_ys))
res.x 

# For opt ordinal 
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y_oh = enc.fit_transform(y[:,j][..., np.newaxis]).toarray() 
categ_lik_opt(theta, y_oh, zM, k, o1, ps_y, p_z_ys)

'''
