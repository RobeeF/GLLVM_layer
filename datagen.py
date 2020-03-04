# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:45:45 2020

@author: Utilisateur
"""

import numpy as np
from numpy.random import multivariate_normal, uniform, binomial
import os

os.chdir('C:/Users/Utilisateur/Documents_importants/scolarité/These/Stats/binary_dgmm/asta/asta_Python')
numobs = 100
seed = None

def gen_z(numobs, seed):
    np.random.seed = seed

    true_w = [0.3, 0.4, 0.3]
    true_mu = [-1.26,0,1.26]
    true_cov = np.diag([0.055,0.0361,0.055])
    
    all_z = multivariate_normal(size = numobs, mean = true_mu, cov = true_cov)
    z = np.zeros(numobs)
    cases = uniform(0, 1, numobs)    
    labels = np.zeros(numobs)
    
    ranges = [0] + true_w + [1]
    
    for i in range(numobs):
        for dim in range(3):
            if (cases[i]>np.sum(ranges[:dim + 1])) and (cases[i]<np.sum(ranges[:dim + 2])):
                z[i] = all_z[i, dim]
                labels[i] = dim
    return z, labels

z, labels = gen_z(500, 1)

def gen_data(z, seed = None):
    np.random.seed = seed

    numobs = len(z)  
    p1=2
    p2=1
    p3=1
    p=p1+p2+p3
    szo=5
    y = np.zeros(shape = (numobs, p))
    
    ######## gen binary #################
    alpha_zero = np.repeat([[0.50], [0.60]], axis = 1, repeats = numobs)
    #alphabin = np.repeat([[2], [2.50]], axis = 1, repeats = numobs)
    alphabin = np.array([[2], [2.50]])
    
    # A checker donne que des zeros...
    pred = alpha_zero + alphabin @ z[np.newaxis]
    probbin = np.exp(pred) / (1 + np.exp(pred))
    probbin = probbin.T
    casuali1 = uniform(size = (numobs, p1))
    x = np.zeros((numobs,p1))
    x = (casuali1 < probbin).astype(int)
    y[:,0:p1]=x
  
  
    ######## gen count #################
  
    alpha_zcount = 0.70
    alphacount = [2.00]
  
    pred = alpha_zcount + alphacount @ z[np.newaxis]  
    probcount = np.exp(pred) / (1 + np.exp(pred))
  
    for l in range(numobs):
        y[l,p1:(p1+p2)] = binomial(n = 10, p = probcount[l], size = 1) # p1 + p2 ?
  
    ########gen ordinal#############
    probor = np.zeros(shape = (numobs,szo,p3))
    thr = [-1.00, -0.30, 0.40,  0.90]
  
    alphaor = [[1.50]]
  
  
    for k in range(szo): # Truc chelou avec thr
        if (k<szo - 1):
            pred = thr[k] - alphaor @ z[np.newaxis]
            probor[:,k,:] = (np.exp(pred)/(1+ np.exp(pred))).T
        else:
            probor[:,k,:] = 1
            
    for l in range(numobs):                    
        for i in range(p3):
            x = uniform(0, 1, 1)[0]    
            for k in range(szo):
                if (k==0):
                    if (x<=probor[l,k,i]): 
                        y[l,p - 1] = k
                else:
                    if (probor[l,k-1,i]<x)&(x<=probor[l,k,i]):
                        y[l,p - 1] = k 
  
    return(y)

# Debugging part
y = np.genfromtxt('../asta_Rcode_compMCGH/y.csv', delimiter = ',', skip_header = True)[:,1:]

numobs = 500
M = 1000

#z, labels = gen_z(numobs, seed)
#y = gen_data(z, 1)
#print(labels)

p1=3
p2=1
p=p1+p2
szo=5
o = 5
k = 3
r = 1
var_distrib = np.array(["bernoulli","bernoulli","binomial","ordinal"])
nj= np.array([1,1,10,5])
lik = - np.inf
eps = 1E-05
it = 3
maxstep = 10

## Veri values
w = np.array([0.3,0.4,0.3])
mu = np.array([-1.26,0,1.26]).reshape(k,r)
sigma = np.array([0.055,0.0361,0.055]).reshape(k, r, r)

thr = np.array([[-1.00, -0.30, 0.40,  0.90, 0.00]])
alpha = np.array([[0.50, 2.00], [0.60, 2.50], [0.70, 2.00]]) 
alphaor = np.array([[1.50]])


init = {}
init['mu'] = mu 
init['sigma'] = sigma
init['alpha'] = alpha
init['alphaor'] = alphaor
init['thr'] = thr
init['w'] = w

out = gllvm_alg_mc_pilot(y, numobs, r, k, p, p1, p2, it, o, szo, init, eps, lik, maxstep, 
                             var_distrib, nj, ps_y, M, seed)