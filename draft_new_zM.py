# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:10:47 2020

@author: Utilisateur
"""


els = np.array([1, 2, 3]).reshape(1,3)
prob = np.array([[ 0.8 ,  0.1 ,  0.1 ],
       [ 0.3 ,  0.3 ,  0.4 ],
       [ 0.25,  0.5 ,  0.25],
       [ 0.1, 0.3, 0.6]])


c = prob.cumsum(axis=1)

u = np.random.rand(len(c), 2)
exp_choices = np.expand_dims(u, 1) < c[..., np.newaxis]
idx = exp_choices.argmax(1)
np.select(
els[exp_choices]

choices = (u < c).argmax(axis=1)



np.less(u, c)

els[choices]



prob2 = qM_new[:,:,i].T
c = prob2.cumsum(axis=1)
u = np.random.rand(len(c), 1)
choices = (u < c).argmax(axis=1)

choices.shape

zM[choices].shape

u.shape

