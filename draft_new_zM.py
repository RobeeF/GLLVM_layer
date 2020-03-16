# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:10:47 2020

@author: Utilisateur
"""


els = np.array([1, 2, 3]).reshape(3,1)
prob = np.array([[ 0.96 ,  0.02 ,  0.02 ],
       [ 0.02 ,  0.02 ,  0.96 ],
       [ 0.25,  0.5 ,  0.25],
       [ 0.1, 0.3, 0.6]])


c = prob.cumsum(axis=1)

u = np.random.rand(len(c), 1, 5)
exp_choices = u < c[..., np.newaxis]
idx = exp_choices.argmax(1)
new_els = np.take(els, idx, axis=0)[:,:,0]

# For each path

i = 0
M2 = 200

new_zM2 = np.zeros((M, numobs, r, k))
for i in range(k):
    qM_cum = qM_new[:,:, i].T.cumsum(axis=1)
    u = np.random.rand(numobs, 1, M2)
    
    choices = u < qM_cum[..., np.newaxis]
    idx = choices.argmax(1)
    
    new_zM2[:,:,:,i] = np.take(zM[:,:, i], idx.T, axis=0)

new_zM.mean((0,1))
new_zM2.mean((0,1))

# For all path simultaneously
i = 0
M2 = 100

qM_new.shape
qM_cum = qM_new.cumsum(axis=0)

u = np.random.rand(1, M2, numobs, k)

choices = u < np.expand_dims(qM_cum, 1)
idx = choices.argmax(0)

zM.shape
idx.shape

new_els = np.take(zM, idx, axis=[0, 2])
np.transpose(new_els, (1,0, 2)).mean((0,1))

new_zM.shape
new_zM[:,:,:,i].mean((0,1))






# Old version 
indices = range(0,M)
for i in range(k):
    for obs in range(numobs):
        drawn_ids = np.random.choice(indices, M, p = qM_new[:,obs, i], replace = True)
        new_zM[:,obs,:, i] = zM[drawn_ids,:,i]



