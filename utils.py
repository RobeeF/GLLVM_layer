# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:26:07 2020

@author: Utilisateur
"""


import sys

from time import time
from scipy import linalg
from copy import deepcopy
from itertools import permutations
from sklearn.metrics import precision_score
from sklearn.preprocessing import OneHotEncoder
from init_params import init_params, dim_reduce_init

#from glmlvm import glmlvm

import itertools
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import autograd.numpy as np
from autograd.numpy.linalg import multi_dot, eigh
from autograd.numpy.linalg import cholesky, LinAlgError


def sample_MC_points(zM, p_z_ys, nb_points):
    ''' Resample nb_points from zM with the highest p_z_ys probability
    
    zM (M x r x k ndarray) : The M Monte Carlo copies of z for each path k
    p_z_ys (M x k x 1 ndarray): The probability density of each point
    nb_points (int): The number of points to resample from the original M points
    --------------------------------------------------------------------------------
    returns (tuple: (nb_points x k x 1), (nb_points x r x k) ndarrays): The resampled p(z | y, s) and zM
    '''
    M = p_z_ys.shape[0]
    numobs = p_z_ys.shape[1]
    k = p_z_ys.shape[2]
    r = zM.shape[1]

    assert nb_points > 0    
    assert nb_points < M
    
    # Compute the fraction of points to keep
    rs_frac = nb_points / M
    
    # Compute the <nb_points> points that have the highest probability through the observations
    sum_p_z_ys = p_z_ys.sum(axis = 1, keepdims = True)
    
    # Masking the the points with less probabilities over all observations
    imask = sum_p_z_ys <= np.quantile(sum_p_z_ys, [1 - rs_frac], axis = 0)
    
    msp_z_ys = np.ma.masked_where(np.repeat(imask, axis = 1, repeats = numobs),\
                                  p_z_ys, copy=True)
    
    mzM = np.ma.masked_where(np.repeat(imask, axis = 1, repeats = r),\
                             zM, copy=True)

    # Need to transpose then detranspose due to compressed ordering conventions
    msp_z_ys = np.transpose(msp_z_ys, (1, 2, 0)).compressed()
    msp_z_ys = msp_z_ys.reshape(numobs, k, int(M * rs_frac))

    mzM = np.transpose(mzM, (1, 2, 0)).compressed()
    mzM = mzM.reshape(r, k, int(M * rs_frac))
    
    return np.transpose(msp_z_ys, (2, 0, 1)), np.transpose(mzM, (2, 0, 1))


def misc(true, pred, return_relabeled = False):
    ''' Compute a label invariant misclassification error and can return the relabeled predictions
    
    true (numobs 1darray): array with the true labels
    pred (numobs 1darray): array with the predicted labels
    return_relabeled (Bool): Whether or not to return the relabeled predictions
    --------------------------------------------------------
    returns (float): The misclassification error rate  
    '''
    best_misc = 0
    true_classes = np.unique(true).astype(int)
    nb_classes = len(true_classes)
    
    best_labeled_pred = pred

    best_misc = 1
    
    # Compute of the possible labelling
    all_possible_labels = [list(l) for l in list(permutations(true_classes))]
    
    # And compute the misc for each labelling
    for l in all_possible_labels:
        shift = max(true_classes) + 1
        shift_pred = pred + max(true_classes) + 1
        
        for i in range(nb_classes):
            shift_pred = np.where(shift_pred == i + shift, l[i], shift_pred)
        
        current_misc = np.mean(true != shift_pred)
        if current_misc < best_misc:
            best_misc = deepcopy(current_misc)
            best_labeled_pred = deepcopy(shift_pred)
      
    if return_relabeled:
        return best_misc, best_labeled_pred
    else:
        return best_misc
        
def cluster_purity(cm):
    ''' Compute the cluster purity index mentioned in Chen and He (2016)'''
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm) 

def gen_categ_as_bin_dataset(y, var_distrib):
    ''' Convert the categorical variables in the dataset to binary variables
    
    y (numobs x p ndarray): The observations containing categorical variables
    var_distrib (p 1darray): An array containing the types of the variables in y 
    ----------------------------------------------------------------------------
    returns ((numobs, p_new) ndarray): The new dataset where categorical variables 
    have been converted to binary variables
    '''
    new_y = deepcopy(y)
    new_y = new_y.reset_index(drop = True)
    new_var_distrib = deepcopy(var_distrib[var_distrib != 'categorical'])

    categ_idx = np.where(var_distrib == 'categorical')[0]
    oh = OneHotEncoder(drop = 'first')
        
    for idx in categ_idx:
        name = y.iloc[:, idx].name
        categ_var = pd.DataFrame(oh.fit_transform(pd.DataFrame(y.iloc[:, idx])).toarray())
        nj_var = len(categ_var.columns)
        categ_var.columns = [str(name) + '_' + str(categ_var.columns[i]) for i in range(nj_var)]
        
        # Delete old categorical variable & insert new binary variables in the dataframe
        del(new_y[name])
        new_y = new_y.join(categ_var.astype(int))
        new_var_distrib = np.concatenate([new_var_distrib, ['bernoulli'] * nj_var])
        
    return new_y, new_var_distrib

def ordinal_encoding(sequence, ord_labels, codes):
    ''' Perform label encoding, replacing ord_labels with codes
    
    sequence (numobs 1darray): The sequence to encode
    ord_labels (nj_ord_j 1darray): The labels existing in sequences 
    codes (nj_ord_j 1darray): The codes used to replace ord_labels 
    -----------------------------------------------------------------
    returns (numobs 1darray): The encoded sequence
    '''
    new_sequence = deepcopy(sequence.values)
    for i, lab in enumerate(ord_labels):
        new_sequence = np.where(new_sequence == lab, codes[i], new_sequence)

    return new_sequence
    

def plot_gmm_init(X, Y_, means, covariances, index, title):
    ''' Plot the GMM fitted in the continuous representation of the original data X.
    Code from sklearn website.
    
    X (numobs x 2 nd-array): The 2D continuous representation of the original data
    Y_ (numobs 1darray): The GMM predicted labels 
    means (k x r ndarray): The means of the Gaussian components identified
    covariances (k x r x r): The covariances of the Gaussian components identified
    index (int): Set to zero is ok
    title (str): The title displayed over the graph
    ------------------------------------------------------------------------------
    returns (void): pyplot figure
    '''
    
    color_iter = itertools.cycle(['navy', 'darkorange', 'purple', 'gold', 'red'])

    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

def compute_nj(y, var_distrib):
    ''' Compute nj for each variable y_j
    
    y (numobs x p ndarray): The original data
    var_distrib (p 1darray): The type of the variables in the data
    -------------------------------------------------------------------
    returns (tuple (p 1d array, nb_bin 1d array, nb_ord 1d array)): The number 
    of categories of all the variables, for count/bin variables only and for 
    ordinal variables only
    '''
    
    nj = []
    nj_bin = []
    nj_ord = []
    for i in range(len(y.columns)):
        if np.logical_or(var_distrib[i] == 'bernoulli',var_distrib[i] == 'binomial'): 
            max_nj = np.max(y.iloc[:,i], axis = 0)
            nj.append(max_nj)
            nj_bin.append(max_nj)
        else:
            card_nj = len(np.unique(y.iloc[:,i]))
            nj.append(card_nj)
            nj_ord.append(card_nj)
    
    nj = np.array(nj)
    nj_bin = np.array(nj_bin)
    nj_ord = np.array(nj_ord)

    return nj, nj_bin, nj_ord


def performance_testing(y, labels, k, init_method, var_distrib, nj, r_max = 5, seed = None):
    ''' Utility used to mesure performance of the algorithm on a given dataset.
    For time measures initialisation times are here neglected
    
    y (numobs x p ndarray): The original data
    labels (numobs 1darray): The one-hot encoded labels 
    var_distrib (p 1darray): The type of the variables in the data
    -------------------------------------------------------------------    
    returns (DataFrame): Performance measures 
    '''
    
    results = pd.DataFrame(columns = ['it_id', 'r', 'run_time', 'nb_iterations', 'micro', 'macro'])
    nb_trials = 30
    r_list = range(1, r_max + 1)
    
    if seed != None:
        seed_list = [seed + i for i in range(nb_trials)]
    else:
        seed_list = [None for i in range(nb_trials)]

    if type(y) == pd.core.frame.DataFrame:
        y_pd = deepcopy(y)
        y = y.values
        
    _, nj_bin, nj_ord = compute_nj(y_pd, var_distrib)

    for r in r_list:
        print(r)            
        M = r * 5
        for i in range(nb_trials):
            start = time()
            
            if init_method == 'prince':
                init = dim_reduce_init(y_pd, k, r, nj, var_distrib, dim_red_method = 'prince',\
                                       seed = seed_list[i])
            else:
                init = init_params(r, nj_bin, nj_ord, k, init_seed = seed_list[i])

            try:
                out = glmlvm(y, r, k, init, var_distrib, nj, M, seed = seed_list[i])
                end = time()
                
                # Compute precision
                m, pred = misc(labels, out['classes'], True) 
                macro = precision_score(labels, pred, average = 'macro')
                micro = precision_score(labels, pred, average = 'micro') # same thing as 1 - m ...

                nb_iterations = len(out['likelihood']) - 1
                
                results = results.append({'it_id': i + 1, 'r': r , 'run_time': end -start, \
                            'nb_iterations': nb_iterations, 'micro': micro, 'macro': macro}, ignore_index=True)
            except:
                results = results.append({'it_id': i + 1, 'r': r , 'run_time': np.nan, \
                            'nb_iterations': np.nan, 'micro': np.nan, 'macro': np.nan}, ignore_index=True)
    return results    

#=============================================================================
# Numeric stability
#=============================================================================

def log_1plusexp(eta_):
    ''' Numerically stable version np.log(1 + np.exp(eta)) '''

    eta_original = deepcopy(eta_)
    eta_ = np.where(eta_ >= np.log(sys.float_info.max), np.log(sys.float_info.max) - 1, eta_) 
    return np.where(eta_ >= 50, eta_original, np.log1p(np.exp(eta_)))
        
def expit(eta_):
    ''' Numerically stable version of 1/(1 + exp(eta)) '''
    
    max_value_handled = np.log(np.sqrt(sys.float_info.max) - 1)
    eta_ = np.where(eta_ <= - max_value_handled + 3, - max_value_handled + 3, eta_) 
    eta_ = np.where(eta_ >= max_value_handled - 3, np.log(sys.float_info.max) - 3, eta_) 

    return np.where(eta_ <= -50, np.exp(eta_), 1/(1 + np.exp(-eta_)))


def make_symm(X):
    ''' Ensures that a matric is symmetric by setting the over-diagonal 
    coefficients as the transposed under-diagonal coefficients.
    In our case, it keeps matrices robustly symmetric to rounding errors. 
    X (2d-array): A matrix 
    ----------------------------------------------------------------------
    returns (2d-array): The "symmetrized" matrix
    '''
    return np.tril(X, k = -1) + np.tril(X).T

def make_positive_definite(m, tol = None):
    ''' Computes a matrix close to the original matrix m that is positive definite.
    This function is just a transcript of R' make.positive.definite function.
    m (2d array): A matrix that is not necessary psd.
    tol (int): A tolerence level controlling how "different" the psd matrice
                can be from the original matrix
    ---------------------------------------------------------------
    returns (2d array): A psd matrix
    '''
    d = m.shape[0]
    if (m.shape[1] != d): 
        raise RuntimeError("Input matrix is not square!")
    eigvalues, eigvect = eigh(m)
    
    # Sort the eigen values
    idx = eigvalues.argsort()[::-1]   
    eigvalues = eigvalues[idx]
    eigvect = eigvect[:,idx]
            
    if (tol == None): 
        tol = d * np.max(np.abs(eigvalues)) * sys.float_info.epsilon
    delta = 2 * tol
    tau = np.maximum(0, delta - eigvalues)
    dm = multi_dot([eigvect, np.diag(tau), eigvect.T])
    return(m + dm)

def ensure_psd(mtx_list):
    ''' Checks the positive-definiteness (psd) of a list of matrix. 
    If a matrix is not psd it is replaced by a "similar" positive-definite matrix.
    mtx_list (list of 2d-array/3d-arrays): The list of matrices to check
    ---------------------------------------------------------------------
    returns (list of 2d-array/3d-arrays): A list of matrices that are all psd.
    '''
    
    L = len(mtx_list)
    for l in range(L):
        for idx, X in enumerate(mtx_list[l]):
            try:
                cholesky(X)
            except LinAlgError:
                mtx_list[l][idx] = make_positive_definite(make_symm(X), tol = 10E-5)
    return mtx_list
                
