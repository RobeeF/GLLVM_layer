# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:13:29 2020

@author: Utilisateur
"""

lb_flat = lambda_bin.flatten()
lo_flat = lambda_ord.flatten()

bin_lik = binom_lik_block(lb_flat, y_bin, zM, k, ps_y_new, p_z_ys_new, nj_bin)
ord_lik = ord_lik_block(lo_flat, y_oh, zM, k, ps_y_new, p_z_ys_new, nj_ord)

bin_gr = binom_gr_lik_block(lambda_bin, y_bin, zM, k, ps_y_new, p_z_ys_new, nj_bin)
ord_gr = ord_gr_lik_block(lambda_ord, y_oh, zM, k, ps_y_new, p_z_ys_new, nj_ord)


t = np.zeros(10)
for i in range(10):
    start = time()
    opt = minimize(binom_lik_block, lb_flat, args = (y_bin, zM, k, ps_y_new, p_z_ys_new, nj_bin), 
                       tol = tol, method='BFGS', jac = binom_gr_lik_block)
    end = time()
    t[i] = end - start
    
t.mean()


t2 = np.zeros(10)
for i in range(10):
    start = time()
    for j in range(3):
        if var_distrib[j] == "bernoulli" or var_distrib[j] == "binomial":
            # Add initial guess and lim iterations
            opt = minimize(binom_lik_opt, alpha[j,:], args = (y[:,j], zM, k, ps_y, p_z_ys, nj[j]), 
                   tol = tol, method='BFGS', jac = binom_gr_lik_opt)
    end = time()
    t2[i] = end - start

t2.mean()

print(end-start)
            

old = opt.x

from autograd import grad
from time import time

grad_bin_lik = grad(binom_lik_opt)

t = np.zeros(100)
for i in range(100):
    start = time()
    binom_gr_lik_opt(lambda_bin[j,:], y_bin[:,j], zM, k, ps_y_new, p_z_ys_new, nj_bin[j])
    end = time()
    t[i] = (end-start)

t2 = np.zeros(100)
for i in range(100):
    start = time()
    grad_bin_lik(lambda_bin[j,:], y_bin[:,j], zM, k, ps_y_new, p_z_ys_new, nj_bin[j])
    end = time()
    t2[i] = (end-start)
  
print(t.mean())
print(t2.mean())
grad_ord_lik = grad(ord_lik_opt)
hess_ord_lik = grad(grad_ord_lik)


t = np.zeros(10)
for i in range(10):
    start = time()
    ord_gr_lik(lambda_ord[col_nb], y_oh, zM, k, nj_ord[col_nb], ps_y_new, p_z_ys_new)
    end = time()
    t[i] = (end-start)

t2 = np.zeros(10)
for i in range(10):
    start = time()
    grad_ord_lik(lambda_ord[col_nb], y_oh, zM, k, nj_ord[col_nb], ps_y_new, p_z_ys_new)
    end = time()
    t2[i] = (end-start)
      
print(t.mean())
print(t2.mean())


# Full ordinal 
start = time()
opt2 = minimize(ord_lik_opt, lambda_ord[col_nb] , args = (y_oh, zM, k, nj_ord[col_nb], ps_y_new, p_z_ys_new), 
                               tol = tol, method='trust-constr',  jac = ord_gr_lik, \
                               constraints = linear_constraint, hess = '2-point')
end = time()
print(end - start)

start = time()
opt = minimize(ord_lik_opt, lambda_ord[col_nb] , args = (y_oh, zM, k, nj_ord[col_nb], ps_y_new, p_z_ys_new), 
                               tol = tol, method='trust-constr',  jac = grad_ord_lik, \
                               constraints = linear_constraint, hess = '2-point')
end = time()
print(end- start)

hess_ord_lik(lambda_ord[col_nb], y_oh, zM, k, nj_ord[col_nb], ps_y_new, p_z_ys_new)




t = np.zeros(10)
for i in range(10):
    start = time()
    log_py_zM_bin(lambda_bin, y_bin, zM, k, nj_bin)    
    end = time()
    t[i] = (end-start)

t2 = np.zeros(10)
for i in range(10):
    start = time()
    grad_ord_lik(lambda_ord[col_nb], y_oh, zM, k, nj_ord[col_nb], ps_y_new, p_z_ys_new)
    end = time()
    t2[i] = (end-start)
      
print(t.mean())
print(t2.mean())


from time import time

t = np.zeros(100)
for i in range(100):
    start = time()
    log_py_zM_bin(lambda_bin, y_bin, zM, k, nj_bin)
    end = time()
    t[i] = end - start
    
t.mean()

t = np.zeros(100)
for i in range(100):
    start = time()
    log_py_zM_bin_seq(lambda_bin, y_bin, zM, k, nj_bin)
    end = time()
    t[i] = end - start
    
t.mean()
