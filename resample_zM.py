# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:26:07 2020

@author: Utilisateur
"""

import numpy as np

def sample_MC_points(zM, p_z_ys, nb_points):
    ''' Resample nb_points from zM with the highest p_z_ys probability'''
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