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
 
def binom_gr_lik_block(lambda_bin, y_bin, zM, k, ps_y, p_z_ys, nj_bin):    
    # Deflatten lambda_bin
    r = zM.shape[1]
    nb_bin = y_bin.shape[1]
    lambda_bin = lambda_bin.reshape(nb_bin, r + 1)
            
    eta = lambda_bin[:,1:][np.newaxis] @ zM # shape = (M, nb_bin, k)
    eta = eta + lambda_bin[:,0][np.newaxis, ..., np.newaxis] # Add the constant
    
    pi = 1/(1 + np.exp(-eta))    
    theta_prime = np.repeat(nj_bin[np.newaxis, ..., np.newaxis, np.newaxis], r + 1, axis = -1) *\
                        pi[..., np.newaxis] # to check when r > 1. 

    # Computing So, shape = (M, nb_bin, k, r + 1)
    tzM_aug = np.expand_dims(np.expand_dims(np.transpose(zM, (0, 2, 1)), 1), 2)    
    So = y_bin[np.newaxis, ..., np.newaxis, np.newaxis] - np.expand_dims(theta_prime, 1) 
    So[:,:,:,:,1:] = So[:,:,:,:,1:] * tzM_aug
    
    return - np.sum(np.expand_dims(ps_y[..., np.newaxis], 1) * \
                    np.sum(np.expand_dims(p_z_ys[..., np.newaxis], 2) *  So, axis = 0),\
                    axis = (0,2)).flatten() 
    
                
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

def ord_gr_lik(theta, y_oh, zM, k, o1, ps_y, p_z_ys):
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


def ord_gr_lik_block(lambda_ord, y_oh, zM, k, ps_y, p_z_ys, nj_ord):
    ''' Compute the likelihood gradients of the categorical data
    '''
    
    r = zM.shape[1]
    M = zM.shape[0]
    numobs = y_oh.shape[1]
    
    max_nj_ord = max(nj_ord)
    nb_ord = len(nj_ord)
    
     # Deflatten lambda_ord
    lambda_ord = lambda_ord.reshape(nb_ord, max_nj_ord + r - 1)
    
    lambda0 = lambda_ord[:, :(max_nj_ord - 1)]
    Lambda = lambda_ord[:,(max_nj_ord - 1) :(max_nj_ord + r)]
         
    broad_thro = lambda0.reshape(nb_ord, max_nj_ord - 1, 1)[np.newaxis] 
    eta = broad_thro - np.expand_dims(Lambda[np.newaxis] @ zM, 2) 
    
    # gamma shape (M, nb_ord, max_nj_ord - 1, k)
    gamma = 1 / (1 + np.exp(-eta))
    gamma_prev = np.concatenate([np.zeros((M, nb_ord, 1, k)), gamma], 2)
    gamma_next = np.concatenate([gamma, np.ones((M, nb_ord, 1, k))], 2)
    
    zT = np.transpose(zM[np.newaxis], (1, 2, 0, 3))
    
    # Taken from Moustaki. 
    der_theta_lambda0 = ((1 - gamma) * gamma_next[:, :, 1:]) / (gamma_next[:, :, 1:] - gamma) # Ok
    der_prev_theta_lambda0 = - (1 - gamma[:, :, 1:]) *  gamma[:, :, 1:] / (gamma[:, :, 1:] - gamma_prev[:, :, 1:(max_nj_ord - 1)])
    
    der_b_lambda0 = (1 - gamma) * gamma / (gamma_next[:, :, 1:] - gamma)
    der_prev_b_lambda0 = - (1 - gamma[:, :, 1:]) * gamma_prev[:, :, 1:(max_nj_ord - 1)] / (gamma[:, :, 1:] - gamma_prev[:, :, 1:(max_nj_ord - 1)]) 
    
    # Shape (M, nb_ord, r, max_nj_ord - 1, k)
    der_theta_Lambda = - np.expand_dims(zT, 1) * np.expand_dims(gamma_next[:, :, 1:], 2)
    der_b_Lambda = - np.expand_dims(zT, 1) * np.expand_dims(gamma, 2) 
        
    # Format the derivatives
    der_prev_theta_lambda0 = np.concatenate([np.zeros((M, nb_ord, 1, k)), der_prev_theta_lambda0], axis = 2)
    der_prev_b_lambda0 = np.concatenate([np.zeros((M, nb_ord, 1, k)), der_prev_b_lambda0], axis = 2)

    yg = y_oh[np.newaxis, ..., np.newaxis] 
    
    # Compute lambda0 gradient
    der_log_pyz_lambda0 = np.concatenate([np.zeros((1, nb_ord, numobs, 1, 1)), yg], axis = 3)[:, :, :, :(max_nj_ord - 1)] * \
                        np.expand_dims(der_prev_theta_lambda0, axis = 2) \
                  + yg[:, :, :, :(max_nj_ord - 1)] * np.expand_dims(- der_prev_b_lambda0 + der_theta_lambda0, axis = 2) \
                          - yg[:, :, :, 1:] * np.expand_dims(der_b_lambda0, axis = 2)
         
    lambda0_grad = - np.sum(np.expand_dims(ps_y[np.newaxis], 2) * \
            np.sum(np.expand_dims(np.expand_dims(p_z_ys, axis = 1), axis = 3) * der_log_pyz_lambda0, 0), axis = (1, 3))
    
    # Compute Lambda gradient
    der_log_pyz_Lambda = np.expand_dims(yg[:, :, :, :(max_nj_ord - 1)], 2) * np.expand_dims(der_theta_Lambda, axis = 3) \
                    - np.expand_dims(yg[:, :, :, 1:], 2) * np.expand_dims(der_b_Lambda, axis = 3)
        
    broad_p_z_y_s = np.expand_dims(np.expand_dims(np.expand_dims(p_z_ys, 1), 2), 4)
    Lambda_grad = - np.sum(np.expand_dims(ps_y[np.newaxis, np.newaxis], 3) * \
            np.sum(broad_p_z_y_s * der_log_pyz_Lambda, 0), axis = (2, 3, 4))                    
        
    return np.hstack([lambda0_grad, Lambda_grad]).flatten()