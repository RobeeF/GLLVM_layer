# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:45:45 2020

@author: Utilisateur
"""

import numpy as np
from numpy.random import multivariate_normal, uniform, binomial
import matplotlib.pyplot as plt
import os

os.chdir('C:/Users/Utilisateur/Documents_importants/scolarité/These/Stats/binary_dgmm/asta/mvasta_Python')
from init_params import init_params
from misc import misc

from gllvm_alg_mc import gllvm_alg_mc_pilot

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
    numobs = 1000

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
    p=p1+p2+p3
    szo=5
    y = np.zeros(shape = (numobs, p))
    
    ######## gen binary and count #################
    pred = init['alpha'] @ np.vstack([np.ones((1, numobs)), z.T]) 
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
    probor = np.zeros(shape = (numobs,szo,p3))
      
    for k in range(szo): # Truc chelou avec thr
        if (k<szo - 1):
            pred = init['thr'][:,k] - init['alphaor'] @ z.T
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
    
##############################################################################
# Debugging part
##############################################################################

#y = np.genfromtxt('../asta_Rcode_compMCGH/y.csv', delimiter = ',', skip_header = True)[:,1:]

# Univariate data

numobs = 2000
M = 100

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
 
# Launching the alg
out = gllvm_alg_mc_pilot(y, numobs, r, k, p, p1, p2, 50, o, szo, init, eps, maxstep, 
                             var_distrib, nj, M, None)

out['w']
misc(labels, out['classes'])
new_misc(labels, out['classes'])
plt.scatter(labels, out['classes'])


true = labels
pred = out['classes']
correct_classes = out['classes'] + 5
correct_classes = np.where(correct_classes == 5, 1, correct_classes)
correct_classes = np.where(correct_classes == 6, 0, correct_classes)
correct_classes = np.where(correct_classes == 7, 2, correct_classes)
misc(labels, correct_classes)


print(np.vstack([labels, out['classes']]).T[:200])

# For r > 1:
r = 3

init = init_params(r, p, p1, p2, o, szo, k, 1)
init
# Debug all dimensions are the same
init['mu'][:,0] = mu[:,0]
init['mu'][:,1] = mu[:,0]
init['sigma'][0] = np.eye(2) * sigma[0,0]
init['sigma'][1] = np.eye(2) * sigma[1,0]
init['sigma'][2] = np.eye(2) * sigma[2,0]
init['alpha'][:,0] = alpha[:,0]
init['alpha'][:,1] = alpha[:,1]
init['alpha'][:,2] = alpha[:,1]
init['alphaor'][:,0] = alphaor[0]
init['alphaor'][:,1] = alphaor[0]

init['thr'] = thr
init['w'] = w


out = gllvm_alg_mc_pilot(y, numobs, r, k, p, p1, p2, 20, o, szo, init, eps, lik, maxstep, 
                             var_distrib, nj, M, None)

out['likelihood']

# Debugging E(zz, y, s)
zex = np.load('C:/Users/Utilisateur/Documents_importants/scolarité/These/Stats/binary_dgmm/asta/bad_asta_code/zex.npy')
temp0 = np.load('C:/Users/Utilisateur/Documents_importants/scolarité/These/Stats/binary_dgmm/asta/bad_asta_code/temp0.npy')
temp1 = np.load('C:/Users/Utilisateur/Documents_importants/scolarité/These/Stats/binary_dgmm/asta/bad_asta_code/temp1.npy')
temp2 = np.load('C:/Users/Utilisateur/Documents_importants/scolarité/These/Stats/binary_dgmm/asta/bad_asta_code/temp2.npy')


temp2.shape
zexTzex = zex[...,np.newaxis] @ np.transpose(zex[...,np.newaxis], (0, 2, 1))
np.diagonal(zexTzex).shape
np.atleast_3d(np.diagonal(np.mean(zTz, axis = 0),axis1 = 1, axis2 = 2)) 

np.mean(zTz, 0).shape


import matplotlib.pyplot as plt
plt.scatter(Ez_y[:,0], z[:,0])
plt.scatter(Ez_y[:,1], z[:,1])

z[:,0].min()

plt.hist(Ez_y[:,1])
E_zz_sy.shape

z.shape
Ez_y[:,0].shape

help(plt.plot)


z[...,np.newaxis] @ np.transpose(z[...,np.newaxis], (1, 0))

Ezz_y = 0
for i in range(k):
    Ezz_y = Ezz_y + ps_y[:, i][...,np.newaxis, np.newaxis] * E_zz_sy[:, i, :, :]
plt.scatter(Ezz_y[:,0, 1], z.T.reshape(500,1) * z.reshape(500,1))


########################
# MV debug (thingers crossed!!)
#########################
r = 2
p1 = 3
numobs = 1000
init = init_params(r, p, p1, p2, o, szo, k, seed)

init['alpha']

z, labels = gen_mv_z(numobs, r, init, seed)
y = gen_mvdata(z, init, seed = None)

it = 10
var_distrib = np.array(["bernoulli","bernoulli","binomial","ordinal"])
nj= np.array([1,1,10,5])
eps = 1E-05
it = 20
maxstep = 10
seed = 1
M = 900
k = 3

random_init = init_params(r, p, p1, p2, o, szo, k, None)

random_init['alpha']

out = gllvm_alg_mc_pilot(y, numobs, r, k, p, p1, p2, it, o, szo, random_init, eps, maxstep, 
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



