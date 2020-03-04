# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 19:16:41 2020

@author: Utilisateur
"""

import numpy as np
from copy import deepcopy
from itertools import permutations

def misc(true, pred):
    ''' Compute a label invariant misclassification error '''
    best_misc = 0
    true_classes = np.unique(true).astype(int)
    nb_classes = len(true_classes)
    best_misc = 1
    
    # Compute of the possible labels
    all_possible_labels = [list(l) for l in list(permutations(true_classes))]
    
    # And compute the misc for each label
    for l in all_possible_labels:
        shift = max(true_classes) + 1
        shift_pred = pred + max(true_classes) + 1
        
        for i in range(nb_classes):
            shift_pred = np.where(shift_pred == i + shift, l[i], shift_pred)
        
        current_misc = np.mean(true != shift_pred)
        if current_misc < best_misc:
            best_misc = deepcopy(current_misc)
    return best_misc




    