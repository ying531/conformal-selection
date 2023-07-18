#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:07:05 2022

@author: ying
"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt 
import sys
import os
from sklearn import svm
from utils import gen_data, gen_weight, gen_data_weighted, weighted_BH, weighted_CS 


sig_id = int(sys.argv[1]) - 1 
out_prop = int(sys.argv[2]) /10
seed = int(sys.argv[3]) 
    
    
n = 1000 
ntest = 1000
sig_seq = np.linspace(1, 4, 9)
sig = sig_seq[sig_id]
q = 0.1

Wset = pd.read_csv("./W.csv") 
Wset = np.array(Wset.iloc[:,1:])
theta = np.zeros(50).reshape((50,1))
theta[1:5,] = 0.1

 
# set random seed
np.random.seed(seed)

# Wset = np.random.uniform(low=-3, high=3, size=50*50).reshape((50,50)) 

Xtrain = gen_data(Wset, n, 1)
Xcalib = gen_data(Wset, n, 1)

Xtest0 = gen_data_weighted(Wset, int(ntest *(1-out_prop)), 1, theta)
Xtest1 = gen_data_weighted(Wset, int(ntest * out_prop), sig, theta)
Xtest = np.concatenate((Xtest0, Xtest1))

# training phase
classifier = svm.OneClassSVM(nu=0.004, kernel="rbf", gamma=0.1)
classifier.fit(Xtrain)

# calibration  
calib_weights = gen_weight(Xcalib, theta)
calib_scores = classifier.score_samples(Xcalib) * calib_weights[:,0]

# compute weighted conformal p-values 
test_weights = gen_weight(Xtest, theta)
test_scores = classifier.score_samples(Xtest) * test_weights[:,0]

Ytest = np.concatenate((np.zeros(int(ntest *(1-out_prop))), 
                       np.ones(int(ntest * out_prop))))

# =============================================================================
# BH with weighted conformal p-values
# =============================================================================

wBH = weighted_BH(calib_scores, calib_weights, test_scores, test_weights, q = q)
if len(wBH) == 0:
    wBH_fdp = 0
    wBH_power = 0
else:
    wBH_fdp = np.sum(np.array(wBH) < int(ntest*(1-out_prop))) / len(wBH)
    wBH_power = np.sum(np.array(wBH) >= int(ntest*(1-out_prop))) / int(ntest*out_prop)

 

# =============================================================================
# weighted conformalized selection
# =============================================================================

wCC0, wCC_hete, wCC_homo, wCC_dete = weighted_CS(calib_scores, calib_weights, test_scores, test_weights, q = q)


# =============================================================================
# # summarize fdp, power and number of rejections
# =============================================================================

if len(wCC_hete) == 0:
    wCC_hete_fdp = 0
    wCC_hete_power = 0
else:
    wCC_hete_fdp = np.sum(Ytest[wCC_hete]==0) / len(wCC_hete)
    wCC_hete_power = np.sum(Ytest[wCC_hete]==1) / int(ntest*out_prop)
# check randomization 
wCC_nfilter_hete = len(wCC0) - len(wCC_hete)

if len(wCC_homo) == 0:
    wCC_homo_fdp = 0
    wCC_homo_power = 0
else:
    wCC_homo_fdp = np.sum(Ytest[wCC_homo]==0) / len(wCC_homo)
    wCC_homo_power = np.sum(Ytest[wCC_homo]==1) / int(ntest*out_prop) 
# check randomization 
wCC_nfilter_homo = len(wCC0) - len(wCC_homo)
 
if len(wCC_dete) == 0:
    wCC_dete_fdp = 0
    wCC_dete_power = 0
else:
    wCC_dete_fdp = np.sum(Ytest[wCC_dete]==0) / len(wCC_dete)
    wCC_dete_power = np.sum(Ytest[wCC_dete]==1) / int(ntest*out_prop)
# check randomization 
wCC_nfilter_dete = len(wCC0) - len(wCC_dete)
    

this_res = pd.DataFrame({"fdp": [wBH_fdp, wCC_hete_fdp, wCC_homo_fdp, wCC_dete_fdp], 
                         "power": [wBH_power, wCC_hete_power, wCC_homo_power, wCC_dete_power],
                         "nsel": [len(wBH), len(wCC_hete), len(wCC_homo), len(wCC_dete)],
                         "method": ["WBH", "WCS.hete", "WCS.homo", "WCS.dete"],
                         "seed": seed, "sig": sig}) 


save_path = "./results"
if not os.path.exists(save_path): 
   os.makedirs(save_path)

this_res.to_csv("./results/seed"+str(seed)+"sig"+str(sig_id)+"prop"+str(int(out_prop*10))+".csv")















    
    