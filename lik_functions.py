# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:33:27 2020

@author: Utilisateur
"""

import numpy as np
from scipy.special import binom
import warnings
warnings.filterwarnings('default')
 
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
    
    eta = np.transpose(zM, (0, 2, 1)) @ alpha[1:].reshape(1, r, 1)
    eta = eta + alpha[0].reshape(1, 1, 1)# Add the constant
    
    den = nj * np.log(1 + np.exp(eta))
    num = eta @ y[np.newaxis, np.newaxis]  # To check
    log_p_y_z = num - den + np.log(coeff_binom)
    
    return -np.sum(ps_y * np.sum(p_z_ys * np.transpose(log_p_y_z, (0, 2, 1)), axis = 0))


def compute_py_zM_bin(lambda_bin, y_bin, zM, k, nj_bin): # Précalculer y_oh
    M = zM.shape[0]
        
    yg = np.repeat(y_bin[np.newaxis], axis = 0, repeats = M)
    
    coeff_binom = binom(nj_bin[np.newaxis, np.newaxis], yg)
    coeff_binom = np.transpose(coeff_binom, (0, 2, 1))
    coeff_binom = np.expand_dims(coeff_binom, 2)
    
    eta = lambda_bin[:,1:][np.newaxis] @ zM # shape = (M, nb_bin, k)
    eta = eta + lambda_bin[:,0][np.newaxis, ..., np.newaxis] # Add the constant
    
    den = nj_bin[np.newaxis, ..., np.newaxis] * np.log(1 + np.exp(eta))
    num = np.expand_dims(y_bin.T[np.newaxis], 2) * eta[...,np.newaxis]
    log_p_y_z = num - den[..., np.newaxis] + np.log(coeff_binom)

    return np.transpose(log_p_y_z.sum(1), (0, 2, 1))    
    
def binom_lik_block(lambda_bin, y_bin, zM, k, ps_y, p_z_ys, nj_bin):
    # Deflatten lambda_bin
    nb_bin = y_bin.shape[1]
    r = zM.shape[1]
    lambda_bin = lambda_bin.reshape(nb_bin, r + 1)
    
    pyzM = compute_py_zM_bin(lambda_bin, y_bin, zM, k, nj_bin)
    return -np.sum(ps_y * np.sum(p_z_ys * pyzM, axis = 0))

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


def ord_lik_opt(theta, y_oh, zM, k, o1, ps_y, p_z_ys):
    r = zM.shape[1]
    M = zM.shape[0]
    thro = theta[:(o1 - 1)]
    alphao = theta[o1 - 1:(len(theta) + 1)]
 
    broad_thro = thro.reshape(o1 - 1, 1, 1, 1)
    eta = broad_thro - (np.transpose(zM, (0, 2, 1)) @ alphao.reshape(1, r, 1))[np.newaxis]
    
    gamma = 1 / (1 + np.exp(-eta))
    gamma_prev = np.concatenate([np.zeros((1,M, k, 1)), gamma])
    gamma_next = np.concatenate([gamma, np.ones((1,M, k, 1))])
    pi = gamma_next - gamma_prev
    
    yg = np.expand_dims(y_oh.T, 1)[..., np.newaxis, np.newaxis] 
    
    log_p_y_z = yg * np.log(np.expand_dims(pi, axis=2)) 

    return -np.sum(ps_y * np.sum(np.expand_dims(p_z_ys[np.newaxis], axis = 4) * log_p_y_z, (0,1,4)))



def compute_py_zM_ord(lambda_ord, y_oh, zM, k, nj_ord):
    # Assume that they all have the same number of classes for the moment
    # A mask will soon be needed
    
    # zM shape = (M, r, k)
    r = zM.shape[1]
    M = zM.shape[0]
    
    max_nj_ord = max(nj_ord)
    nb_ord = len(nj_ord)
    
    lambda0 = lambda_ord[:, :(max_nj_ord - 1)] # Shape (nb_ord, max_nj_ord - 1)
    Lambda = lambda_ord[:,(max_nj_ord - 1) :(max_nj_ord + r)] # Shape (nb_ord, r + 1)
     
    broad_thro = lambda0.reshape(nb_ord, max_nj_ord - 1, 1)[np.newaxis] 
    eta = broad_thro - np.expand_dims(Lambda[np.newaxis] @ zM, 2) 

    gamma = 1 / (1 + np.exp(-eta))
    gamma_prev = np.concatenate([np.zeros((M, nb_ord, 1, k)), gamma], 2)
    gamma_next = np.concatenate([gamma, np.ones((M, nb_ord, 1, k))], 2)
    pi = gamma_next - gamma_prev
    
    yg = y_oh[np.newaxis,..., np.newaxis] 
    log_p_y_z = yg * np.log(np.expand_dims(pi, axis=2)) # Using sparse format could accelerate things up
   
    return log_p_y_z.sum((1, 3))


def ord_lik_block(lambda_ord, y_oh, zM, k, ps_y, p_z_ys, nj_ord):
    # Deflatten lambda_ord
    r = zM.shape[1]
    nb_ord = y_oh.shape[0]
    max_nj_ord = y_oh.shape[2]
    lambda_ord = lambda_ord.reshape(nb_ord, max_nj_ord + r - 1)

    pyzM = compute_py_zM_ord(lambda_ord, y_oh, zM, k, nj_ord)
    return -np.sum(ps_y * np.sum(p_z_ys * pyzM, axis = 0))

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
