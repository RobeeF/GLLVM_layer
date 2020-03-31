# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:55:44 2020

@author: Utilisateur
"""

from sklearn.cluster import KMeans
from factor_analyzer import FactorAnalyzer
#from gllvm_block import gllvm

import autograd.numpy as np
from autograd.numpy.random import uniform

from autograd.numpy import newaxis as n_axis
from autograd.numpy import transpose as t

from autograd.numpy.linalg import cholesky, pinv

from sklearn import manifold



def init_params(r, nj_bin, nj_ord, k, init_seed):
    ''' Generate random initialisations for the parameters
    Consider no regressors here'''
    
    # Seed for init
    np.random.seed = init_seed
    init = {}
    
    
    # Gaussian mixture params
    init['w'] = np.full(k, 1/k) 
    
    mu_init = np.repeat(np.linspace(-1.0, 1.0, num = k)[..., n_axis], axis = 1, repeats =r)
    init['mu'] = (uniform(low = -5, high = 5, size = (1,1)) * mu_init)
    init['mu'] = init['mu'][..., np.newaxis]
  
    init['sigma'] = np.zeros(shape = (k, r, r))
    for i in range(k):
        init['sigma'][i] = 0.050 * np.eye(r)
        
    # Enforcing identifiability constraints
    muTmu = init['mu'] @ t(init['mu'], (0,2,1))  
     
    E_zzT = (init['w'][..., n_axis, n_axis] * (init['sigma'] + muTmu)).sum(0, keepdims = True)
    Ezz_T = (init['w'][...,n_axis, n_axis] * init['mu']).sum(0, keepdims = True)
    
    var_z = E_zzT - Ezz_T @ t(Ezz_T, (0,2,1)) # Koenig-Huyghens Formula for Variance Computation
    sigma_z = cholesky(var_z)
     
    init['sigma'] = pinv(sigma_z) @ init['sigma'] @ t(pinv(sigma_z), (0, 2, 1))
    init['mu'] = pinv(sigma_z) @ init['mu']
    init['mu']  = init['mu']  - Ezz_T

    # GLLVM params    
    p1 = len(nj_bin)
    p2 = len(nj_ord)
    
    if p1 > 0:
        init['lambda_bin'] = uniform(low = -3, high = 3, size = (p1, r + 1))
        init['lambda_bin'][:,1:] = init['lambda_bin'][:,1:] @ sigma_z[0] 
        
        if (r > 1): 
            init['lambda_bin'] = np.tril(init['lambda_bin'], k = 1)

    else:
        init['lambda_bin'] = np.array([]) #np.full((p1, r + 1), np.nan)
  
    if p2 > 0:
        max_nj_ord = np.max(nj_ord)

        lambda0_ord = np.zeros(shape = (p2, max_nj_ord - 1)) # Why not max_nj_ord - 1
        for j in range(p2):
            lambda0_ord[j, :nj_ord[j] - 1] = np.sort(uniform(low = -2, high = 2, size = (1, nj_ord[j] - 1))) #.reshape(1, szo, r)
  
        Lambda_ord = uniform(low = -3, high = 3, size = (p2, r))
        init['lambda_ord'] = np.hstack([lambda0_ord, Lambda_ord])
        init['lambda_ord'][:, max_nj_ord - 1:] = init['lambda_ord'][:, max_nj_ord -1  :] @ sigma_z[0]
        
    else:
        init['lambda_ord'] = np.array([])#np.full((p2, 1), np.nan)
    
    return(init)


def init_cv(y, var_distrib, r, nj_bin, nj_ord, k, seed):
    ''' Test 20 different inits for a few iterations and returns the best one'''
    
    numobs = y.shape[0]
  
    nb_init_tested = 10
    M = 20
    best_lik = -1000000
    best_init = {}
    nb_it = 2
    maxstep = 100
    eps = 1E-5
    nj = np.concatenate([nj_bin, nj_ord])

    for i in range(nb_init_tested):
        init = init_params(r, nj_bin, nj_ord, k, None)
        try:
            out = gllvm(y, numobs, r, k, nb_it, init, eps, maxstep, var_distrib, nj, M, seed)
        except:
            continue

        lik = out['likelihood'][-1]
    
        if (best_lik < lik):
            best_lik = lik
            best_init = init
    
    return(best_init)

####################################################################################
# Try k-means + factanal init
####################################################################################



def factanal_init(y, k1, r1):
    ''' Initiate the paremeters of the Gaussian Mixture.
    Later will be converted for MFAs '''
        
    p = y.shape[1]
    init = {}
    
    pred_labels = KMeans(n_clusters=k1).fit_predict(y)
    
    labels_indices, count = np.unique(pred_labels, return_counts = True)
    w = count / np.sum(count)
    init['w'] = w

    psi = np.zeros((k1, r1, r1))    
    mu = np.zeros((k1, r1, 1))
    
    for idx in labels_indices:
    
        fa = FactorAnalyzer(rotation = None, method = 'ml', n_factors = 1)
        stima = fa.fit(y[pred_labels == idx])
    
        psi[idx] = np.diag(stima.get_uniquenesses())
        mu[idx] = np.mean(y[pred_labels == idx])[..., np.newaxis]
        
        
    # Ensure identifiability
        
        
    return init

####################################################################################
# Try "Umap/t-sne/prince + kmeans + estim mu and sigma in the projected space" init
####################################################################################

def bin_to_bern(Nj, yk_binom, zMk_binom):
    n_yk = len(yk_binom) # parameter k of the binomial
    
    # Generate Nj Bernoullis from each binomial and get a (numobsxNj, 1) table
    u = uniform(size =(n_yk,Nj))
    p = (yk_binom/Nj)[..., n_axis]
    yk_bern = (u > p).astype(int).flatten('A')#[..., n_axis] 
        
    return yk_bern, np.repeat(zMk_binom, Nj, 0)

# umap init
import os
os.chdir('C:/Users/rfuchs/Documents/GitHub/GLLVM_layer')

from sklearn.linear_model import LogisticRegression
import umap
from bevel.linear_ordinal_regression import  OrderedLogit # Dirty local hard copy of the Github bevel package
import prince
import pandas as pd

def dim_reduce_init(y, k, r, nj, var_distrib, dim_red_method = 'prince', seed = None):
    ''' Perform dimension reduction into a continuous r dimensional space and determine the init coefficients in that space'''
    
    if dim_red_method == 'umap':
        reducer = umap.UMAP(n_components = r, random_state = seed)
        z_emb = reducer.fit_transform(y)
        
    elif dim_red_method == 'tsne':
        tsne = manifold.TSNE(n_components = r, init='pca', random_state = seed)
        z_emb = tsne.fit_transform(y)
    elif dim_red_method == 'prince':
        
        if type(y) != pd.core.frame.DataFrame:
            raise TypeError('y should be a dataframe for prince')
            
        famd = prince.FAMD(n_components=2, n_iter=3, copy=True, \
                           check_input=True, engine='auto', random_state = seed)
        famd = famd.fit(y)
        z_emb = famd.row_coordinates(y).values.astype(float)
        
        y = y.values.astype(int)
    else:
        raise ValueError('Only tnse, umap and prince initialisation is available not ', dim_red_method)
        
    y_bin = y[:, np.logical_or(var_distrib == 'bernoulli',var_distrib == 'binomial')]
    nj_bin = nj[np.logical_or(var_distrib == 'bernoulli',var_distrib == 'binomial')]
    nb_bin = len(nj_bin)
    
    y_ord = y[:, var_distrib == 'ordinal']    
    nj_ord = nj[var_distrib == 'ordinal']
    nb_ord = len(nj_ord)
       
    pred_labels = KMeans(n_clusters = k).fit_predict(z_emb)
    
    init = {}
    
    labels_indices, count = np.unique(pred_labels, return_counts = True)
    w = count / np.sum(count)
    init['w'] = w
    
    init['mu'] = np.zeros((k, r, 1))
    init['sigma'] = np.zeros((k, r, r))    
    
    for k, label in enumerate(labels_indices):
        init['mu'][k] = z_emb[pred_labels == label].mean(axis=0, keepdims = True).T
        init['sigma'][k] = np.cov(z_emb[pred_labels == label].T)
    
    # Enforcing identifiability constraints
    muTmu = init['mu'] @ t(init['mu'], (0,2,1))  
     
    E_zzT = (init['w'][..., n_axis, n_axis] * (init['sigma'] + muTmu)).sum(0, keepdims = True)
    Ezz_T = (init['w'][...,n_axis, n_axis] * init['mu']).sum(0, keepdims = True)
    
    var_z = E_zzT - Ezz_T @ t(Ezz_T, (0,2,1)) # Koenig-Huyghens Formula for Variance Computation
    sigma_z = cholesky(var_z)
     
    init['sigma'] = pinv(sigma_z) @ init['sigma'] @ t(pinv(sigma_z), (0, 2, 1))
    init['mu'] = pinv(sigma_z) @ init['mu']
    init['mu']  = init['mu']  - Ezz_T        
         
    # Determining the coefficients of the GLLVM layer
    ## Determining lambda_bin coefficients.
    
    lambda_bin = np.zeros((nb_bin, r + 1))
    
    for j in range(nb_bin): 
        Nj = np.max(y_bin[:,j]) # The support of the jth binomial is [1, Nj]
        for k, label in enumerate(labels_indices):
            
            if Nj ==  1:  # If the variable is Bernoulli not binomial
                yk = y_bin[pred_labels == labels_indices[k],j]
                zMk = z_emb[pred_labels == label]
            else: # If not, need to convert Binomial output to Bernoulli output
                yk, zMk = bin_to_bern(Nj, y_bin[pred_labels == labels_indices[k],j], z_emb[pred_labels == label])
            
            lr = LogisticRegression()
            lr.fit(zMk, yk)
            lambda_bin[j] = np.concatenate([lr.intercept_, lr.coef_[0]])
    
    
    ## Determining lambda_ord coefficients
    max_nj_ord = np.max(nj_ord)
    lambda_ord = np.zeros((nb_ord, max_nj_ord - 1 + r))
    
    for j in range(nb_ord):
        Nj = len(np.unique(y_ord[:,j], axis = 0))  # The support of the jth ordinal is [1, Nj]
        for k, label in enumerate(labels_indices):
            yk = y_ord[pred_labels == labels_indices[k],j]
            zMk = z_emb[pred_labels == label]
    
            ol = OrderedLogit()
            ol.fit(zMk, yk)
            lambda_ord[j,:Nj - 1 + r] = np.concatenate([ol.alpha_, ol.beta_]) 
    
    # Identifiability part on the GLLVM
    lambda_bin = np.tril(lambda_bin, k = 1)
    lambda_bin[:,1:] = lambda_bin[:,1:] @ sigma_z[0] 
    lambda_ord[:, max_nj_ord - 1:] = lambda_ord[:, max_nj_ord -1  :] @ sigma_z[0]
            
    init['lambda_bin'] = lambda_bin
    init['lambda_ord'] = lambda_ord
    
    return init


