# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:43:11 2020

@author: Utilisateur
"""

import numpy as np

def binom_gr_lik_opt(alpha, y, zM, k, ps_y, p_z_ys, nj):    
    r = zM.shape[1]
    
    zMT = np.transpose(zM, (0, 2, 1))
    eta = zMT @ alpha[1:].reshape(1, r, 1)
    eta = eta + alpha[0].reshape(1, 1, 1)# Add the constant
        
    pi = 1/(1 + np.exp(-eta))
    
    theta_prime = np.full(r + 1, nj) * np.expand_dims(pi, axis = 1) # to check when r > 1
    
    So = np.expand_dims(y[..., np.newaxis, np.newaxis], axis = 0) - theta_prime
    So[:,:,:,1:] = So[:,:,:,1:] * np.expand_dims(zMT, 1)
    
    return - np.sum(ps_y[..., np.newaxis] * \
                    np.sum(p_z_ys[...,np.newaxis] *  So, axis = 0),\
                    axis = (0,1))
    
                
def binom_hess(alpha, y, zM, k, ps_y, p_z_ys, nj):
    r = zM.shape[1]
    
    eta = np.transpose(zM, (0, 2, 1)) @ alpha[1:].reshape(r, 1, 1)
    eta = eta + alpha[0].reshape(1, 1, 1)# Add the constant
        
    flatH = nj * np.exp(eta)/(1 + np.exp(eta))**2
    flatH = np.ones((r + 1)**2) * np.expand_dims(flatH, axis = 1)
    
    flatH[:,:,:,1] = flatH[:,:,:,1] * zM # S0''(lo, l1)
    flatH[:,:,:,2] = flatH[:,:,:,2] * zM # S0''(l1, l0)
    flatH[:,:,:,3] = flatH[:,:,:,3] * zM**2 # S0''(l1, l1)
    
    flatH = - np.sum(ps_y[..., np.newaxis] * \
                    np.sum(p_z_ys[...,np.newaxis] *  flatH, axis = 0), axis = (0,1))
    
    return flatH.reshape(r + 1, r + 1)
   
###########################################################################
# Categorical gradient
###########################################################################

def categ_gr_lik(theta, y_oh, zM, k, o1, ps_y, p_z_ys):
    ''' Compute the likelihood gradients of the categorical data
    '''
    r = zM.shape[1]
    M = zM.shape[0]
    numobs = len(y_oh)
    thro = theta[:(o1 - 1)]
    alphao = theta[o1 - 1:(len(theta) + 1)]
 
    broad_thro = thro.reshape(o1 - 1, 1, 1, 1)
    zT = np.transpose(zM, (0, 2, 1))
    
    eta = broad_thro - (zT @ alphao.reshape(1, r, 1))[np.newaxis]
    
    gamma = 1 / (1 + np.exp(-eta))
    gamma_prev = np.concatenate([np.zeros((1,M, k, 1)), gamma])
    gamma_next = np.concatenate([gamma, np.ones((1,M, k, 1))])
    
    # Taken from Moustaki. 
    der_theta_thr = ((1 - gamma) * gamma_next[1:]) / (gamma_next[1:] - gamma) # Ok
    der_prev_theta_thr = - (1 - gamma[1:]) *  gamma[1:] / (gamma[1:] - gamma_prev[1:len(thro)])
    
    der_b_thr = (1 - gamma) * gamma / (gamma_next[1:] - gamma)
    der_prev_b_thr = - (1 - gamma[1:]) * gamma_prev[1:len(thro)] / (gamma[1:] - gamma_prev[1:len(thro)]) 
    
    der_theta_alphao = - zT[np.newaxis] * gamma_next[1:] 
    der_b_alphao = - zT[np.newaxis] * gamma 
    
    # Format the derivatives
    der_prev_theta_thr = np.concatenate([np.zeros((1,M, k, 1)), der_prev_theta_thr])
    der_prev_b_thr = np.concatenate([np.zeros((1,M, k, 1)), der_prev_b_thr])

    yg = np.expand_dims(y_oh.T, 1)[..., np.newaxis, np.newaxis] 

    # Attention aux signes
    der_log_pyz_thr = np.concatenate([np.zeros((1,1, numobs, 1, 1)), yg])[:len(thro)] * \
                        np.expand_dims(der_prev_theta_thr, axis = 2) \
                  + yg[:len(thro)] * np.expand_dims(- der_prev_b_thr + der_theta_thr, axis = 2) \
                          - yg[1:] * np.expand_dims(der_b_thr, axis = 2)
                          
    thr_grad = - np.sum(ps_y[np.newaxis, ..., np.newaxis] * \
            np.sum(np.expand_dims(p_z_ys[np.newaxis], axis = 4) * der_log_pyz_thr, 1), axis = (1, 2))
    
    der_log_pyz_alphao = yg[:len(thro)] * np.expand_dims(der_theta_alphao, axis = 2) \
                    - yg[1:] * np.expand_dims(der_b_alphao, axis = 2)
                    
    alphao_grad = - np.sum(ps_y[np.newaxis, ..., np.newaxis] * \
            np.sum(np.expand_dims(p_z_ys[np.newaxis], axis = 4) * der_log_pyz_alphao, 1), axis = (0, 1, 2))                    
        
    return np.concatenate([thr_grad.flatten(), alphao_grad.flatten()]) 
    
    #A la mano 
    '''
    g = 2
    (1 - gamma[g]) * gamma[g+1] / (gamma[g+1] - gamma[g]) # Ok pour 1
    
    - (1 - gamma[g]) * gamma[g] / (gamma[g] - gamma[g - 1]) # Ok pour 2
    der_prev_theta_thr[g - 1]
    
    # Compute of the 4 gradients by hand
    gammag = np.expand_dims(gamma, axis = 2)
    derf_t2 = - yg[0] * ((1 - gammag[1]) * gammag[1]) / (gammag[1] - gammag[0]) \
    + y_oh[1] * ((1 - gammag[1]) * gammag[0]) / (gammag[1] - gammag[0]) \
    + y_oh[1] * ((1 - gammag[1]) * gammag[2]) / (gammag[2] - gammag[1]) \
    - y_oh[2] * ((1 - gammag[1]) * gammag[1]) / (gammag[2] - gammag[1])
        
    print(derf_t2[:,:,:,1].sum())
    '''