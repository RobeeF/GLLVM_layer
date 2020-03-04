# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 19:16:41 2020

@author: Utilisateur
"""

import numpy as np
from copy import deepcopy
from itertools import permutations

def misc(true, pred):
    best_misc = 0
    true_classes = np.unique(true)
    nb_classes = len(true_classes)
    best_misc = 1

    for i in range(nb_classes):
        relabeled_pred = (pred + i) % nb_classes
        current_misc = np.mean(true != relabeled_pred)
        if current_misc < best_misc:
            best_misc = deepcopy(current_misc)
    return best_misc

i = 0
correct_classes - relabeled_pred

def new_misc(true, pred):
    best_misc = 0
    true_classes = np.unique(true)
    nb_classes = len(true_classes)
    best_misc = 1
    
    all_possible_labels = list(permutations(true))