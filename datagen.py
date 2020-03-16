# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:45:45 2020

@author: Utilisateur
"""

#import numpy as np
import autograd.numpy as np
from numpy.random import multivariate_normal, uniform, binomial
import matplotlib.pyplot as plt
import os

os.chdir('C:/Users/Utilisateur/Documents/GitHub/GLLVM_layer_clean')
from init_params import init_params, init_cv
from misc import misc
from gllvm_block import gllvm

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


####################################################################
# Multivariate data generation
####################################################################
def gen_mv_z(numobs, r, init, seed):
    np.random.seed = seed

    k  = 3

    all_z = np.zeros((numobs,r,k))
    for i in range(k): # To change with np.block_diag when ready
        all_z[:,:,i] = multivariate_normal(size = numobs, mean = init['mu'][i], cov = init['sigma'][i]) 

    z = np.zeros((numobs, r))
    cases = uniform(0, 1, numobs)    
    labels = np.zeros(numobs)
    
    ranges = [0] + init['w'].flatten().tolist() + [1]
    
    for i in range(numobs):
        for dim in range(3):
            if (cases[i]>np.sum(ranges[:dim + 1])) and (cases[i]<np.sum(ranges[:dim + 2])):
                z[i] = all_z[i, : ,dim]
                labels[i] = dim
    return z, labels  

def gen_mvdata(z, init, seed = None):
    np.random.seed = seed

    numobs = len(z)  
    p1 = 2
    p2 = 1
    p3 = 1
    p = p1+p2+p3
    max_nj_ord = 5
    y = np.zeros(shape = (numobs, p))
    
    ######## gen binary and count #################
    pred = init['lambda_bin'] @ np.vstack([np.ones((1, numobs)), z.T]) 
    probbin = np.exp(pred) / (1 + np.exp(pred))
    probbin = probbin.T
    casuali1 = uniform(size = (numobs, p1))
    x = np.zeros((numobs,p1))
    x = (casuali1 < probbin[:,:p1]).astype(int)
    y[:,0:p1]=x

    # For count
    for l in range(numobs):
        y[l,p1:(p1+p2)] = binomial(n = 10, p = probbin[l,p1:(p1 + p2)], size = 1) # p1 + p2 ?

    ########gen ordinal#############
    probor = np.zeros(shape = (numobs, max_nj_ord, p3))
      
    # Erreur... A reprendre
    lambda0 = init['lambda_ord'][:, :(max_nj_ord - 1)] # Shape (nb_ord, max_nj_ord)
    Lambda = init['lambda_ord'][:,(max_nj_ord - 1) :(max_nj_ord + r)] # Shape (nb_ord, r)
    
    for k in range(max_nj_ord): # Truc chelou avec thr
        if (k < max_nj_ord - 1):
            pred = lambda0[:,k] - Lambda @ z.T
            probor[:,k,:] = (np.exp(pred)/(1+ np.exp(pred))).T
        else:
            probor[:,k,:] = 1
            
    for l in range(numobs):                    
        for i in range(p3):
            x = uniform(0, 1, 1)[0]    
            for k in range(max_nj_ord):
                if (k==0):
                    if (x<=probor[l,k,i]): 
                        y[l,p - 1] = k
                else:
                    if (probor[l,k-1,i]<x)&(x<=probor[l,k,i]):
                        y[l,p - 1] = k 
  
    return(y)
    
##############################################################################
# Debugging part
##############################################################################

#y = np.genfromtxt('../asta_Rcode_compMCGH/y.csv', delimiter = ',', skip_header = True)[:,1:]

# Univariate data

numobs = 1000
M = 300

z, labels = gen_z(numobs, seed)
y = gen_data(z, 1)

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
seed = 1

## Veri values
r = 1
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

## Random init
init = init_params(r, p, p1, p2, o, szo, k, 1)
init = init_cv(y, k, var_distrib, r, nj, seed = 1)
# Launching the alg
#out = gllvm_alg_mc_pilot(y, numobs, r, k, p, p1, p2, 50, o, szo, init, eps, maxstep, 
                             #var_distrib, nj, M, None)

out['sigma']
misc(labels, out['classes'])
new_misc(labels, out['classes'])
plt.scatter(labels, out['classes'])


print(np.vstack([labels, out['classes']]).T[:200])

########################
# MV debug (thingers crossed!!)
#########################
r = 2
p1 = 3
numobs = 1500
M = 600
k = 3

seed = 1
init_seed = 2

nj= np.array([1,1,10,5])
nj_bin = np.array([1,1,10])
nj_ord = np.array([5])
init = init_params(r, nj_bin, nj_ord, k, init_seed)
var_distrib = np.array(["bernoulli","bernoulli","binomial","ordinal"])

# For working purposis we can double the ordinal column
#var_distrib = np.concatenate([var_distrib, ['ordinal']])
#nj = np.concatenate([nj, [5]])
#y_ord = np.repeat(y_ord, 2, 1) 

eps = 1E-05
it = 1
maxstep = 10

z, labels = gen_mv_z(numobs, r, init, seed)
y = gen_mvdata(z, init, seed = None)


random_init = init_params(r, nj_bin, nj_ord, k, init_seed)

out = gllvm(y, numobs, r, k, p, p1, p2, it, o, szo, random_init, eps, maxstep, 
                             var_distrib, nj, M, None)

misc(labels, out['classes'])

out['mu']
init['mu']

out['sigma']
init['sigma']

out["classes"] == 1

plt.scatter(range(numobs),labels)

plt.scatter(Ez_y[:,0], z[:,0]) 
plt.scatter(Ez_y[:,1], z[:,1])

plt.scatter(Ez_y[:,0], z[:,0]) # Pas un soucis de variance ?
E_z_sy[:, 0, :].max(axis = 0)

zM.var(axis = 0)
zM.mean(axis = 0)

zM.max(axis = 0)
zM.max(axis = 0)

z.max(axis = 0)
z.min(axis = 0)

for key, value in init.items():
    if key != 'alpha.tot':
        print('Actual vs estimated', key)
        print(init[key])
        print(out[key])

true = labels 
pred = out["classes"] - 1

from copy import deepcopy



