# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:33:27 2020

@author: Utilisateur
"""

import autograd.numpy as np
from autograd.numpy import newaxis as n_axis
from scipy.special import binom
import warnings
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('default')


def binom_loglik_j(lambda_bin_j, y_bin_j, zM, k, ps_y, p_z_ys, nj_bin_j): # Passer plus de chose en argument
    ''' Compute the expected log-likelihood for each binomial variable y_j
    lambda_bin_j ( (r + 1) 1darray): Coefficients of the binomial distributions in the GLLVM layer
    y_bin_j (numobs 1darray): The subset containing only the binary/count variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    ps_y (numobs x k ndarray): p(s_i = k1 | y_i) for all k1 in [1,k] and i in [1,numobs]
    p_z_ys (M x numobs x k ndarray): p(z_i | y_i, s_i = k) for all m in [1,M], k1 in [1,k] and i in [1,numobs]
    nj_bin (int): The number of possible values/maximum values of the jth binary/count variable
    --------------------------------------------------------------
    returns (float): E_{zM, s | y, theta}(y_bin_j | zM, s1 = k1)
    '''    
    M = zM.shape[0]
    r = zM.shape[1]
    numobs = len(y_bin_j)
    
    yg = np.repeat(y_bin_j[np.newaxis], axis = 0, repeats = M)
    coeff_binom = binom(nj_bin_j, yg).reshape(M, 1, numobs)
    
    eta = np.transpose(zM, (0, 2, 1)) @ lambda_bin_j[1:].reshape(1, r, 1)
    eta = eta + lambda_bin_j[0].reshape(1, 1, 1) # Add the constant
    
    den = nj_bin_j * np.log(1 + np.exp(eta))
    num = eta @ y_bin_j[np.newaxis, np.newaxis]  # To check
    log_p_y_z = num - den + np.log(coeff_binom)
    
    return -np.sum(ps_y * np.sum(p_z_ys * np.transpose(log_p_y_z, (0, 2, 1)), axis = 0))


def log_py_zM_bin(lambda_bin, y_bin, zM, k, nj_bin): 
    ''' Compute sum_j log p(y_j | zM, s1 = k1) of all the binomial data at once
    lambda_bin (nb_bin x (r + 1) ndarray): Coefficients of the binomial distributions in the GLLVM layer
    y_bin (numobs x nb_bin ndarray): The subset containing only the binary/count variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    nj_bin (nb_bin x 1d-array): The number of possible values/maximum values of binary/count variables respectively
    --------------------------------------------------------------
    returns (ndarray): The sum_j p(y_j | zM, s1 = k1)
    '''
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
    
def binom_loglik(lambda_bin, y_bin, zM, k, ps_y, p_z_ys, nj_bin):
    ''' Compute expected log-likelihood of all the binomial variables at once
    lambda_bin (nb_bin x (r + 1) ndarray): Coefficients of the binomial distributions in the GLLVM layer
    y_bin (numobs x nb_bin ndarray): The subset containing only the binary/count variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    ps_y (numobs x k ndarray): p(s_i = k1 | y_i) for all k1 in [1,k] and i in [1,n]
    p_z_ys (M x numobs x k ndarray): p(z_i | y_i, s_i = k) for all m in [1,M], k1 in [1,k] and i in [1,numobs]
    nj_bin (nb_bin x 1d-array): The number of possible values/maximum values of binary/count variables respectively
    --------------------------------------------------------------
    returns (float): E_{zM, s | y, theta}(y_bin | zM, s1 = k1)
    '''
    # Deflatten lambda_bin
    nb_bin = y_bin.shape[1]
    r = zM.shape[1]
    lambda_bin = lambda_bin.reshape(nb_bin, r + 1)
    
    pyzM = log_py_zM_bin(lambda_bin, y_bin, zM, k, nj_bin)
    return -np.sum(ps_y * np.sum(p_z_ys * pyzM, axis = 0))

######################################################################
# Ordinal likelihood functions
######################################################################

def log_py_zM_ord_j(lambda_ord_j, y_oh_j, zM, k, nj_ord_j): # Prendre uniquement les coeff non 0 avec nj_ord
    ''' Compute log p(y_j | zM, s1 = k1) of each ordinal variable 
    lambda_ord_j ( (nj_ord_j + r - 1) 1darray): Coefficients of the ordinal distributions in the GLLVM layer
    y_oh_j (numobs 1darray): The jth ordinal variable in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    nj_ord_j (int): The number of possible values values of the jth ordinal variable
    --------------------------------------------------------------
    returns (ndarray): The p(y_j | zM, s1 = k1) for the jth ordinal variable
    '''    
    r = zM.shape[1]
    M = zM.shape[0]
    lambda0 = lambda_ord_j[:(nj_ord_j - 1)]
    Lambda = lambda_ord_j[(nj_ord_j - 1):(len(lambda_ord_j) + 1)]
 
    broad_lambda0 = lambda0.reshape((nj_ord_j - 1, 1, 1, 1))
    eta = broad_lambda0 - (np.transpose(zM, (0, 2, 1)) @ Lambda.reshape((1, r, 1)))[np.newaxis]
    
    gamma = 1 / (1 + np.exp(-eta))
    gamma_prev = np.concatenate([np.zeros((1,M, k, 1)), gamma])
    gamma_next = np.concatenate([gamma, np.ones((1,M, k, 1))])
    pi = gamma_next - gamma_prev
    
    yg = np.expand_dims(y_oh_j.T, 1)[..., np.newaxis, np.newaxis] 
    
    log_p_y_z = yg * np.log(np.expand_dims(pi, axis=2)) 
    
    return log_p_y_z.sum((0))

def log_py_zM_ord(lambda_ord, y_ord, zM, k, nj_ord): 
    ''' Compute sum_j log p(y_j | zM, s1 = k1) of all the ordinal data with a for loop
    lambda_ord ( nb_ord x (nj_ord_j + r - 1) 1darray): Coefficients of the ordinal distributions in the GLLVM layer
    y_ord (numobs x nb_bin ndarray): The subset containing only the binary/count variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    nj_ord (nb_ord x 1d-array): The number of possible values values of ordinal variables
    --------------------------------------------------------------
    returns (ndarray): The sum_j p(y_j | zM, s1 = k1) for ordinal variables
    '''
    
    nb_ord = y_ord.shape[1]
    enc = OneHotEncoder(categories='auto')
    r = zM.shape[1]

    log_pyzM = 0
    for j in range(nb_ord):
        lambda_ord_j = np.concatenate([lambda_ord[j, :(nj_ord[j] - 1)], lambda_ord[j, -r:]])
        y_oh_j = enc.fit_transform(y_ord[:,j][..., n_axis]).toarray()
        log_pyzM += log_py_zM_ord_j(lambda_ord_j, y_oh_j, zM, k, nj_ord[j])
        
    return log_pyzM
        

def ord_loglik_j(lambda_ord_j, y_oh_j, zM, k, ps_y, p_z_ys, nj_ord_j):
    ''' Compute the expected log-likelihood for each ordinal variable y_j
    lambda_ord_j ( (nj_ord_j + r - 1) 1darray): Coefficients of the ordinal distributions in the GLLVM layer
    y_oh_j (numobs 1darray): The subset containing only the ordinal variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    ps_y (numobs x k ndarray): p(s_i = k1 | y_i) for all k1 in [1,k] and i in [1,numobs]
    p_z_ys (M x numobs x k ndarray): p(z_i | y_i, s_i = k) for all m in [1,M], k1 in [1,k] and i in [1,numobs]
    nj_ord_j (int): The number of possible values of the jth ordinal variable
    --------------------------------------------------------------
    returns (float): E_{zM, s | y, theta}(y_ord_j | zM, s1 = k1)
    ''' 
    log_pyzM_j = log_py_zM_ord_j(lambda_ord_j, y_oh_j, zM, k, nj_ord_j)
    return -np.sum(ps_y * np.sum(np.expand_dims(p_z_ys, axis = 3) * log_pyzM_j, (0,3)))


def log_py_zM_ord_block(lambda_ord, y_ord, zM, k, nj_ord):
    ''' Compute sum_j log p(y_j | zM, s1 = k1) of all the ordinal data at once
    lambda_ord ( nb_ord x (nj_ord_j + r - 1) 1darray): Coefficients of the ordinal distributions in the GLLVM layer
    y_ord (numobs x nb_ord ndarray): The subset containing only the ordinal variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    nj_ord (nb_ord x 1d-array): The number of possible values values of ordinal variables
    --------------------------------------------------------------
    returns (ndarray): The sum_j p(y_j | zM, s1 = k1) for ordinal variables
    '''
    r = zM.shape[1]
    M = zM.shape[0]
    numobs = len(y_ord)    
    
    max_nj_ord = max(nj_ord)
    nb_ord = len(nj_ord)
        
    enc = OneHotEncoder(categories='auto')
    y_oh = np.zeros((nb_ord, numobs, max_nj_ord))
    for j in range(nb_ord):
        y_oh[j, :, : nj_ord[j]] = enc.fit_transform(y_ord[:,j][..., n_axis]).toarray()
            
    lambda0 = lambda_ord[:, :(max_nj_ord - 1)] # Shape (nb_ord, max_nj_ord - 1)
    Lambda = lambda_ord[:,(max_nj_ord - 1) :(max_nj_ord + r)] # Shape (nb_ord, r + 1)
     
    broad_lambda0 = lambda0.reshape(nb_ord, max_nj_ord - 1, 1)[np.newaxis] 
    eta = broad_lambda0 - np.expand_dims(Lambda[np.newaxis] @ zM, 2) 
        
    gamma = 1 / (1 + np.exp(-eta)) # shape = (M, nb_ord, k, r)
    for j in range(nb_ord):
        gamma[:, j, (nj_ord[j]-1):] = 1          
        
    gamma_prev = np.concatenate([np.zeros((M, nb_ord, 1, k)), gamma], 2)
    gamma_next = np.concatenate([gamma, np.ones((M, nb_ord, 1, k))], 2)
    pi = gamma_next - gamma_prev  
    pi = np.where(pi <= 0, 1E-14, pi)# <0 values suppressed just after by yg mask. Not very elegant..
      
    yg = y_oh[np.newaxis,..., np.newaxis] 
    p_y_z = np.expand_dims(pi, axis=2) ** yg
    log_p_y_z = np.where(p_y_z > 0, np.log(p_y_z), p_y_z)
    
    return log_p_y_z.sum((1, 3))

def ord_loglik(lambda_ord, y_oh, zM, k, ps_y, p_z_ys, nj_ord):
    ''' Compute expected log-likelihood of all the ordinal variables at once
    lambda_ord ( nb_ord x (nj_ord_j + r - 1) 1darray): Coefficients of the ordinal distributions in the GLLVM layer
    y_oh (numobs x nb_ord ndarray): The subset containing only the ordinal variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    ps_y (numobs x k ndarray): p(s_i = k1 | y_i) for all k1 in [1,k] and i in [1,n]
    p_z_ys (M x numobs x k ndarray): p(z_i | y_i, s_i = k) for all m in [1,M], k1 in [1,k] and i in [1,numobs]
    nj_ord (nb_ord x 1d-array): The number of possible values values of ordinal variables
    --------------------------------------------------------------
    returns (float): E_{zM, s | y, theta}(y_ord | zM, s1 = k1)
    '''
    r = zM.shape[1]
    nb_ord = y_oh.shape[0]
    max_nj_ord = y_oh.shape[1]
    lambda_ord = lambda_ord.reshape(nb_ord, max_nj_ord + r - 1)

    log_pyzM = log_py_zM_ord_block(lambda_ord, y_oh, zM, k, nj_ord)
    return -np.sum(ps_y * np.sum(p_z_ys * log_pyzM, axis = 0))

#########################################################################################
# Test section 
#########################################################################################
'''
lambda_ord_j = lambda_ord[0]

enc = OneHotEncoder(categories='auto')
y_oh = enc.fit_transform(y_ord[:,0][..., n_axis]).toarray()


np.abs(log_py_zM_ord(lambda_ord, y_ord, zM, k, nj_ord)[:,:,:,0] - \
log_py_zM_ord_block(lambda_ord, y_oh, zM, k, nj_ord)).sum()


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
