# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:22:03 2020

@author: Utilisateur
"""

from lik_gradients import binom_gr_lik, binom_gr_lik_opt
import time

# Test avec different optimizers
# tester avec "Nelder-Mead", "BFGS", 

# Pour la binomial
nb_trials = 10
times = []
for i in range(nb_trials):
    a = time.time()
    opt = minimize(binom_lik, alpha[j,:], args = (y[:,j], zM, k, ps_y, p_z_ys, nj[j]), 
                   tol = 0.1, method='BFGS', jac = binom_gr_lik) 
    b = time.time()
    times.append(b-a)
    print(b-a)
    
times_neldermead = times
times_bfgs

# Pour l'ordinal
j = 3
a = time.time()
opt = minimize(categ_lik, theta, args = (y[:,j], zM, k, o1, ps_y, p_z_ys), \
                               options = {'maxiter': 2}, tol = 1)
b = time.time()
times.append(b-a)
print(b-a)

#################################################################################
# Binomial optimized vs non optimized
#################################################################################
nb_trials = 10
times = []

for i in range(nb_trials):
    a = time.time()
    opt = minimize(binom_lik, alpha[j,:], args = (y[:,j], zM, k, ps_y, p_z_ys, nj[j]), 
                   tol = 0.1, method='Nelder-Mead') 
    b = time.time()
    times.append(b-a)
    print(b-a)

times_opt = []
for i in range(nb_trials):
    a = time.time()
    opt = minimize(binom_lik_opt, alpha[j,:], args = (y[:,j], zM, k, ps_y, p_z_ys, nj[j]), 
                   tol = 0.1, method='Nelder-Mead') 
    b = time.time()
    times_opt.append(b-a)
    print(b-a)
    
    
np.mean(times)/np.mean(times_opt) # Optimized is 1.5 times faster !! 

######################################################################################
# Ordinal optimized vs non optimized
######################################################################################
nb_trials = 10
times = []

for i in range(nb_trials):
    a = time.time()
    opt = minimize(categ_lik, theta, args = (y[:,j], zM, k, o1, ps_y, p_z_ys), 
                   tol = 0.5, method='Nelder-Mead') 
    b = time.time()
    times.append(b-a)
    print(b-a)

times_opt = []
for i in range(nb_trials):
    a = time.time()
    opt = minimize(categ_lik_opt, theta, args = (y_oh, zM, k, o1, ps_y, p_z_ys), 
                   tol = 0.5, method='Nelder-Mead') 
    b = time.time()
    times_opt.append(b-a)
    print(b-a)
    
    
np.mean(times)/np.mean(times_opt) # Optimized is more than 3 times faster !! 


#####################################################################################
# Binominal with Gradient vs w/o gradient
##################################################################################### 
nb_trials = 10
times = []
tol = 1E-5

for i in range(nb_trials):
    a = time.time()
    opt = minimize(binom_lik_opt, alpha[j,:], args = (y[:,j], zM, k, ps_y, p_z_ys, nj[j]), 
                   tol = tol, method='Nelder-Mead') 
    b = time.time()
    times.append(b-a)
    print(b-a)

times_opt = []
for i in range(nb_trials):
    a = time.time()
    opt = minimize(binom_lik_opt, alpha[j,:], args = (y[:,j], zM, k, ps_y, p_z_ys, nj[j]), 
                   tol = tol, method='BFGS', jac = binom_gr_lik_opt) 
    b = time.time()
    times_opt.append(b-a)
    print(b-a)
    
np.mean(times)/np.mean(times_opt) # Optimized is more than 4 times faster !! 

#####################################################################################
# Binominal with Hessian vs w/o Hessian
##################################################################################### 
nb_trials = 10
times = []
tol = 1E-5

for i in range(nb_trials):
    a = time.time()
    opt = minimize(binom_lik_opt, alpha[j,:], args = (y[:,j], zM, k, ps_y, p_z_ys, nj[j]), 
                   tol = tol, method='BFGS', jac = binom_gr_lik_opt) 
    b = time.time()
    times.append(b-a)
    print(b-a)

times_opt = []
for i in range(nb_trials):
    a = time.time()
    opt = minimize(binom_lik_opt, alpha[j,:], args = (y[:,j], zM, k, ps_y, p_z_ys, nj[j]), 
                   tol = tol, method='Newton-CG', jac = binom_gr_lik_opt, hess = binom_hess) 
    b = time.time()
    times_opt.append(b-a)
    print(b-a)
    
# Hessian based method seem to be slower. Might try with bigger dataset
# BFGS performs good !

#########################################################################################
# Ordinal with gradient vs. w/o gradient
#########################################################################################

nb_trials = 10
times = []
tol = 1E-5

for i in range(nb_trials):
    a = time.time()
    opt = minimize(categ_lik_opt, theta, args = (y_oh, zM, k, o1, ps_y, p_z_ys), 
                   tol = 0.1, method='Nelder-Mead') 
    b = time.time()
    times.append(b-a)
    print(b-a)

times_opt = []
for i in range(nb_trials):
    a = time.time()
    opt = minimize(categ_lik_opt, theta, args = (y_oh, zM, k, o1, ps_y, p_z_ys), 
                   tol = 0.1, method='BFGS', jac = categ_gr_lik) 
    b = time.time()
    times_opt.append(b-a)
    print(b-a)
    
np.mean(times)/np.mean(times_opt) # Optimized is more than 4 times faster !! 

from scipy.optimize import Bounds 
from scipy.optimize import LinearConstraint

b = [-np.inf, *theta[:o-1], np.inf]
bounds = Bounds(lb = b[:(o - 1)], ub = b[1:]) # Bounds have to evolve hence there are linear constraints

minimize(categ_lik_opt, theta, args = (y_oh, zM, k, o1, ps_y, p_z_ys), 
                   tol = 0.1, jac = categ_gr_lik, bounds = bounds)

# Create constraint matrices (very dirty)
nb_cst_params = thr.shape[1] - 1
np_params = theta.shape[0]
lcs = np.full(nb_cst_params - 1, -1)
lcs = np.diag(lcs, 1)
np.fill_diagonal(lcs, 1)
lcs = np.hstack([lcs[:nb_cst_params - 1,:], np.zeros([np_params - nb_cst_params, np_params - nb_cst_params])])

linear_constraint = LinearConstraint(lcs, np.full(nb_cst_params - 1, -np.inf), \
                                     np.full(nb_cst_params - 1, 0), keep_feasible = True)

opt = minimize(categ_lik_opt, theta, args = (y_oh, zM, k, o1, ps_y, p_z_ys), 
                   tol = 0.1, method='trust-constr',  jac = categ_gr_lik, \
                   constraints = linear_constraint, hess = '2-point')


dir(linear_constraint)
linear_constraint.A @ theta[np.newaxis].T
lcs @ theta[np.newaxis].T



# Hard computed gradient vs JAX

from time import time
t = np.zeros(10)
for i in range(10):
    start = time()
    a = log_py_zM_ord(lambda_ord, y_ord, zM, k, nj_ord)[:,:,:,0]
    end = time()
    t[i] = end - start
    
t.mean()

from time import time
t = np.zeros(10)
for i in range(10):
    start = time()
    a = log_py_zM_ord_block(lambda_ord, y_ord, zM, k, nj_ord)
    end = time()
    t[i] = end - start
    
t.mean()

