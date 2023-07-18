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
from utils import weighted_BH, weighted_CS, eval_FDR

q = int(sys.argv[1]) / 10 
seed = int(sys.argv[2])  

data = pd.read_csv("bank_data.csv")
data = data.iloc[:, 1:data.shape[1]]

 
np.random.seed(seed)

# =============================================================================
# # select test sample and create covariate shift
# =============================================================================
in_test = np.random.binomial(n=1, p=data['ex']*0.5)
test_data = data.iloc[in_test==1,:]
test_data.index = np.arange(0, np.sum(in_test))

remain_data = data.iloc[in_test!=1, :]
remain_data.index = np.arange(0, np.sum(1-in_test))

# =============================================================================
# # super-population approach
# =============================================================================
ncalib = np.sum(1-in_test) // 2
calib_data = remain_data.iloc[0:ncalib,]
train_data = remain_data.iloc[(ncalib+1):,]

sup_mdl = svm.SVC(kernel="rbf", gamma=0.01, probability=True).fit(train_data.iloc[:,0:62], train_data['class'])
sup_calib_scores = np.array(100 * calib_data['class'] - sup_mdl.predict_proba(calib_data.iloc[:,0:62])[:,1])
sup_test_scores = np.array(- sup_mdl.predict_proba(test_data.iloc[:,0:62]))[:,1]

sup_calib_weights = np.array(calib_data['ex'] * 0.5 / (1 - 0.5 * calib_data['ex']))
sup_test_weights = np.array(test_data['ex'] * 0.5 / (1 - 0.5 * test_data['ex']))

sup_wBH = weighted_BH(sup_calib_scores, sup_calib_weights, 
                      sup_test_scores, sup_test_weights, q = q)

sup_ , sup_wCC_hete, sup_wCC_homo, sup_wCC_dtm = weighted_CS(sup_calib_scores, sup_calib_weights, sup_test_scores, sup_test_weights, q = q)

truth_test = 1 - np.array(test_data['class'])
sup_wBH_eval = eval_FDR(sup_wBH, truth_test)
sup_wCC_hete_eval = eval_FDR(sup_wCC_hete, truth_test)
sup_wCC_homo_eval = eval_FDR(sup_wCC_homo, truth_test)
sup_wCC_dtm_eval = eval_FDR(sup_wCC_dtm, truth_test)


# =============================================================================
# # classification, conditional approach
# =============================================================================
cond_calib_scores = sup_calib_scores[calib_data['class']==0]
cond_calib_weights = sup_calib_weights[calib_data['class']==0]
cond_wBH = weighted_BH(cond_calib_scores, cond_calib_weights, 
                       sup_test_scores, sup_test_weights, q = q)

cond_, cond_wCC_hete, cond_wCC_homo, cond_wCC_dtm = weighted_CS(cond_calib_scores, cond_calib_weights, sup_test_scores, sup_test_weights, q = q)

cond_wBH_eval = eval_FDR(cond_wBH, truth_test)
cond_wCC_hete_eval = eval_FDR(cond_wCC_hete, truth_test)
cond_wCC_homo_eval = eval_FDR(cond_wCC_homo, truth_test)
cond_wCC_dtm_eval = eval_FDR(cond_wCC_dtm, truth_test)

# =============================================================================
# # outlier detection, one-class classification
# =============================================================================
# change a few parameters
out_mdl = svm.OneClassSVM(kernel="rbf", gamma=0.01)
out_mdl.fit(train_data[train_data['class']==0].iloc[:,0:62])

out_calib_scores = out_mdl.score_samples(calib_data[calib_data['class']==0].iloc[:,0:62])
out_test_scores = out_mdl.score_samples(test_data.iloc[:,0:62])

out_wBH = weighted_BH(out_calib_scores, cond_calib_weights, out_test_scores, sup_test_weights, q = q)
out_wBH_eval = eval_FDR(out_wBH, truth_test)
print(out_wBH_eval)
 
out_, out_wCC_hete, out_wCC_homo, out_wCC_dtm = weighted_CS(out_calib_scores, cond_calib_weights, out_test_scores, sup_test_weights, q = q)

out_wCC_hete_eval = eval_FDR(out_wCC_hete, truth_test)
out_wCC_homo_eval = eval_FDR(out_wCC_homo, truth_test)
out_wCC_dtm_eval = eval_FDR(out_wCC_dtm, truth_test)
     

# =============================================================================
# # summarize fdp, power and number of rejections
# =============================================================================
results = pd.DataFrame(np.array((sup_wBH_eval, sup_wCC_hete_eval, 
                                 sup_wCC_homo_eval,sup_wCC_dtm_eval, 
                    cond_wBH_eval, cond_wCC_hete_eval, cond_wCC_homo_eval, cond_wCC_dtm_eval,
                    out_wBH_eval, out_wCC_hete_eval, out_wCC_homo_eval, out_wCC_dtm_eval)))
results.columns = ("nrej", "fdp", "power")
results['method'] = ["WBH", "WCS.hete", "WCS.homo", "WCS.dtm"] * 3
results['setting'] = ["sup_class"]*4+["cond_class"]*4+["outlier"]*4

# summarize diff in set
results['ndiff1'] = [0, len(set(sup_wBH)-set(sup_wCC_hete)), len(set(sup_wBH)-set(sup_wCC_homo)), len(set(sup_wBH)-set(sup_wCC_dtm)),
                     0, len(set(cond_wBH)-set(cond_wCC_hete)), len(set(cond_wBH)-set(cond_wCC_homo)), len(set(cond_wBH)-set(cond_wCC_dtm)),
                     0, len(set(out_wBH)-set(out_wCC_hete)), len(set(out_wBH)-set(out_wCC_homo)), len(set(out_wBH)-set(out_wCC_dtm))]

results['ndiff2'] = [0, len(set(sup_wCC_hete)-set(sup_wBH)), len(set(sup_wCC_homo)-set(sup_wBH)), len(set(sup_wCC_dtm)-set(sup_wBH)),
                     0, len(set(cond_wCC_hete)-set(cond_wBH)), len(set(cond_wCC_homo)-set(cond_wBH)), len(set(cond_wCC_dtm)-set(cond_wBH)),
                     0, len(set(out_wCC_hete)-set(out_wBH)), len(set(out_wCC_homo)-set(out_wBH)), len(set(out_wCC_dtm)-set(out_wBH))]

results['ntest'] = np.sum(in_test)
results['seed'] = seed
 

save_path = "./results"
if not os.path.exists(save_path): 
   os.makedirs(save_path)
results.to_csv("./results/seed_"+str(seed)+"_q_"+str(q)+".csv")








 
    
    