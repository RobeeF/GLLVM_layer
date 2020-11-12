# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:24:27 2020

@author: rfuchs
"""

import autograd.numpy as np
from copy import deepcopy
import sys

#=============================================================================
# Numeric stability
#=============================================================================

def log_1plusexp(eta_):
    ''' Numerically stable version np.log(1 + np.exp(eta)) '''

    eta_original = deepcopy(eta_)
    eta_ = np.where(eta_ >= np.log(sys.float_info.max), np.log(sys.float_info.max) - 1, eta_) 
    return np.where(eta_ >= 50, eta_original, np.log1p(np.exp(eta_)))
        
def expit(eta_):
    ''' Numerically stable version of 1/(1 + exp(eta)) '''
    
    max_value_handled = np.log(np.sqrt(sys.float_info.max) - 1)
    eta_ = np.where(eta_ <= - max_value_handled + 3, - max_value_handled + 3, eta_) 
    eta_ = np.where(eta_ >= max_value_handled - 3, np.log(sys.float_info.max) - 3, eta_) 

    return np.where(eta_ <= -50, np.exp(eta_), 1/(1 + np.exp(-eta_)))
