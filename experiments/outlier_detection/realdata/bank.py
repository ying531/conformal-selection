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


















# =============================================================================
# 
# def gen_data(Wset, n, a): 
#     Wi = Wset[np.random.choice(range(50), n),:]
#     Vi = np.random.normal(size=n*50).reshape((n,50))
#     Xi = np.sqrt(a) * Vi + Wi
#     return(Xi)
#     
# def gen_weight(Xi, theta):
#     linx = Xi @ theta   
#     wx = np.exp(linx) / (1+ np.exp(linx))    
#     return(wx)
#     
# def gen_data_weighted(Wset, n, a, theta):
#     Wi = Wset[np.random.choice(range(50), n),:]
#     Vi = np.random.normal(size=n*50).reshape((n,50))
#     Xi = np.sqrt(a) * Vi + Wi
#     wx = gen_weight(Xi, theta)
#     sel = np.random.binomial(n=1, p=wx[:,0])
#     X_sel = Xi[sel==1,:]
#     X = X_sel
#     
#     while X.shape[0] < n:
#         Wi = Wset[np.random.choice(range(50), n),:]
#         Vi = np.random.normal(size=n*50).reshape((n,50))
#         Xi = np.sqrt(a) * Vi + Wi
#         wx = gen_weight(Xi, theta)
#         sel = np.random.binomial(n=1, p=wx[:,0])
#         X_sel = Xi[sel==1,:]
#         X = np.concatenate((X, X_sel))
#     
#     X = X[range(n),:]
#     
#     return(X)
# 
# def weighted_BH(calib_scores, calib_weights, test_scores, test_weights, q = 0.1):
#     pvals = np.zeros(len(test_scores))
#     df_all = pd.concat((pd.DataFrame({"score": calib_scores, "weight": calib_weights[:,0], "cal": 1}), pd.DataFrame({"score": test_scores, "weight": test_weights[:,0], "cal": 0})))
#     df_sorted = df_all.sort_values(by = 'score')
#     all_sorted = np.array(df_sorted)
#     sum_calib_weight = np.sum(calib_weights)
#     
#     p_vals = []
#     for j in range(all_sorted.shape[0]):
#         if all_sorted[j,2] == 0:
#             p_vals.append( (np.sum(all_sorted[range(j),1] * all_sorted[range(j),2]) + all_sorted[j,1] * np.random.uniform(size=1)[0]) / (sum_calib_weight + all_sorted[j,1]) )
#         else:
#             p_vals.append(-1)
#             
#     df_sorted['pvals'] = p_vals
#         
#     df_test = df_sorted[df_sorted['cal'] == 0]
#     df_test_sorted = df_test.sort_values(by='pvals')
#     
#     # BH(q)
#     ntest = len(test_scores)
#     df_test_sorted['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
#     idx_smaller = [j for j in range(ntest) if df_test_sorted.iloc[j,3] <= df_test_sorted.iloc[j,4]]
#     
#     if len(idx_smaller) == 0:
#         return(np.array([]))
#     else:
#         idx_sel = np.array(df_test_sorted.index[range(np.max(idx_smaller)+1)])
#         return(idx_sel)
#         
# 
# def weighted_CC(calib_scores, calib_weights, test_scores, test_weights, 
#                 q = 0.1, xi = None):
#     sum_calib_weight = np.sum(calib_weights)
#     
#     ntest = len(test_scores)
#     # sel_0 = np.zeros((ntest, ntest)) # row j indicates hat{R}_j
#     Rj_sizes = np.zeros(ntest)
#     w_pvals = np.zeros(ntest)
#     for j in range(ntest): 
#         # compute all other pvals 
#         pval_j = np.zeros(ntest)
#         for k in range(ntest):
#             if k != j:
#                 pval_j[k] =  np.sum(calib_weights[calib_scores < test_scores[k]]) + test_weights[k] * (test_scores[j] < test_scores[k]) 
#         pval_j = pval_j / (sum_calib_weight + test_weights[j])
#         w_pvals[j] = ( np.sum(calib_weights[calib_scores < test_scores[j]]) + (np.sum(calib_weights[calib_scores == test_scores[j]]) + test_weights[j])  * np.random.uniform(size=1)[0] ) / (sum_calib_weight + test_weights[j])
#         
#         # run BH
#         df_j = pd.DataFrame({"id": range(ntest), "pval": pval_j}).sort_values(by = 'pval')
#         df_j['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
#         idx_small_j = [s for s in range(ntest) if df_j.iloc[s, 1] <= df_j.iloc[s, 2]]
#         Rj = np.array(df_j['id'])[range(np.max(idx_small_j)+1)] 
#         Rj_sizes[j] = len(Rj)
#         
#     Cj = q * Rj_sizes / ntest
#     
#     if xi is None: 
#         xi = np.random.uniform(size=ntest)
#     
#     df_all = pd.DataFrame({"id": range(ntest), "pval": w_pvals, "c": Cj, 
#                            "xiRj": Rj_sizes * xi, "xiRj_homo": Rj_sizes * xi[0]})
#     # selection without random pruning 
#     pj_sel0 = w_pvals[w_pvals <= Cj]
#     
#     ECj = 1*(w_pvals <= Cj) * ntest / (q * Rj_sizes)
#     
#     
#     if len(pj_sel0) == 0:
#         return np.array([]), np.array([]), np.array([])
#     else:
#         # hetero pruning 
#         df_sel0 = df_all[df_all['pval'] <= df_all['c']].sort_values(by='xiRj')
#         df_sel0['threshold'] = np.linspace(1, df_sel0.shape[0], num = df_sel0.shape[0])
#         hete_smaller = [j for j in range(df_sel0.shape[0]) if df_sel0.iloc[j,3] <= df_sel0.iloc[j,5]]
#         
#         # homo pruning 
#         df_sel0_homo = df_all[df_all['pval'] <= df_all['c']].sort_values(by='xiRj_homo')
#         df_sel0_homo['threshold'] = np.linspace(1, df_sel0_homo.shape[0], num = df_sel0_homo.shape[0])
#         homo_smaller = [j for j in range(df_sel0_homo.shape[0]) if df_sel0.iloc[j,4] <= df_sel0_homo.iloc[j,5]]
#         if len(hete_smaller) == 0:
#             hete_sel = np.array([])
#         else:
#             hete_sel = np.array(df_sel0['id'])[range(np.max(hete_smaller)+1)]
#             # how much do we need randomization
#         
#         if len(homo_smaller) == 0:
#             homo_sel = np.array([])
#         else:
#             homo_sel = np.array(df_sel0_homo['id'])[range(np.max(homo_smaller)+1)]
#             
#         return np.array(df_sel0['id']), hete_sel, homo_sel, ECj
#     
#     
# def weighted_eBH(calib_scores, calib_weights, test_scores, test_weights, 
#                  q = 0.1):
#     sum_calib_weight = np.sum(calib_weights)
#     df_all = pd.concat((pd.DataFrame({"score": calib_scores, "weight": calib_weights[:,0], "cal": 1}), pd.DataFrame({"score": test_scores, "weight": test_weights[:,0], "cal": 0})))
#     df_sorted = df_all.sort_values(by = 'score')
#     arr_sorted = np.array(df_sorted)
#     
#     ntest = len(test_scores) 
#     hat_fdps = np.zeros(ntest) 
#     
#     # compute cumulative weights sum_i w(Xi)ind{V_i <= t} 
#     df_sorted['cw_calib'] = 0
#     df_sorted['cw_test'] = 0
#     
#     for k in range(df_sorted.shape[0]):
#         cw_calib = np.sum(arr_sorted[range(k), 1] * arr_sorted[range(k),2])
#         cw_test = max(1, np.sum(1-arr_sorted[range(k),2]))
#         df_sorted.iloc[k,3] = cw_calib
#         df_sorted.iloc[k,4] = cw_test
#     
#     w_evals = np.zeros(df_sorted.shape[0])
#     for j in range(df_sorted.shape[0]):
#         if df_sorted.iloc[j,2] == 0: # test point
#             hat_fdps = np.array((df_sorted['cw_calib'] + df_sorted.iloc[j,1]) * ntest / (df_sorted['cw_test'] * (sum_calib_weight + df_sorted.iloc[j, 1])))
#             smaller = [s for s in range(df_sorted.shape[0]) if hat_fdps[s]<=q]
#             if len(smaller) > 0: 
#                 Tj = df_sorted.iloc[np.max(smaller), 0] 
#                 if df_sorted.iloc[j, 0] <= Tj:
#                     w_evals[j] = (sum_calib_weight + arr_sorted[j,1]) / (df_sorted.iloc[np.max(smaller), 3] + arr_sorted[j,1])
#         else: 
#             w_evals[j] = -1
#     
#     df_sorted['eval'] = w_evals
#     df_test = df_sorted[df_sorted['cal']==0]
#     
#     # eBH(q)
#     df_test = df_test.sort_values(by='eval', ascending=False)
#     df_test['threshold'] = ntest / (q * np.linspace(1, ntest, num=ntest))
#     ebh_smaller = [j for j in range(ntest) if df_test.iloc[j, 5] >= df_test.iloc[j,6]]
#     
#     if len(ebh_smaller) == 0:
#         return(np.array([]))
#     else:
#         idx_sel = np.array(df_test.index)[range(np.max(ebh_smaller)+1)]
#         return(idx_sel)
#     
#     
#     
#     
# def weighted_eBH_boost(calib_scores, calib_weights, test_scores, test_weights, 
#                              q = 0.1, c = 0.1, xi = None):
#     sum_calib_weight = np.sum(calib_weights)
#     df_all = pd.concat((pd.DataFrame({"score": calib_scores, "weight": calib_weights[:,0], "cal": 1}), pd.DataFrame({"score": test_scores, "weight": test_weights[:,0], "cal": 0})))
#     df_sorted = df_all.sort_values(by = 'score')
#     arr_sorted = np.array(df_sorted)
#     
#     ntest = len(test_scores) 
#     hat_fdps = np.zeros(ntest) 
#     
#     # compute cumulative weights sum_i w(Xi)ind{V_i <= t} 
#     df_sorted['cw_calib'] = 0
#     df_sorted['cw_test'] = 0
#     
#     for k in range(df_sorted.shape[0]):
#         cw_calib = np.sum(arr_sorted[range(k), 1] * arr_sorted[range(k),2])
#         cw_test = max(1, np.sum(1-arr_sorted[range(k),2]))
#         df_sorted.iloc[k,3] = cw_calib
#         df_sorted.iloc[k,4] = cw_test
#         
#     if xi is None:
#         xi = np.random.uniform(size=ntest)
#     
#     w_evals_raw = np.zeros(df_sorted.shape[0])
#     # w_evals_homo = np.zeros(df_sorted.shape[0])
#     # w_evals_hetero = np.zeros(df_sorted.shape[0])
#     homo_xi = np.random.uniform(size=1)[0]
#     for j in range(df_sorted.shape[0]):
#         if df_sorted.iloc[j,2] == 0: # test point
#             hat_fdps = np.array((df_sorted['cw_calib'] + df_sorted.iloc[j,1]) * ntest / (df_sorted['cw_test'] * (sum_calib_weight + df_sorted.iloc[j, 1])))
#             smaller = [s for s in range(df_sorted.shape[0]) if hat_fdps[s]<= c]
#             if len(smaller) > 0: 
#                 Tj = df_sorted.iloc[np.max(smaller), 0] 
#                 if df_sorted.iloc[j, 0] <= Tj:
#                     w_evals_raw[j] = (sum_calib_weight + arr_sorted[j,1]) / (df_sorted.iloc[np.max(smaller), 3] + arr_sorted[j,1])
#                     # w_evals_hetero[j] = w_evals_raw[j] / np.random.uniform(size=1)[0]
#                     # w_evals_homo[j] = w_evals_raw[j] / homo_xi
#         # else: 
#         #     w_evals_hetero[j] = -1
#         #     w_evals_hetero[j] = -1
#     
#     # df_sorted['eval_hetero'] = w_evals_hetero
#     # df_sorted['eval_homo'] = w_evals_homo
#     df_sorted['eval_raw'] = w_evals_raw
#     df_test = df_sorted[df_sorted['cal']==0]
#     df_test['eval_hetero'] = df_test['eval_raw'] / xi 
#     df_test['eval_homo'] = df_test['eval_raw'] / xi[0]
#     
#     # eBH(q) with heterogenous boosting
#     df_test = df_test.sort_values(by='eval_hetero', ascending=False)
#     df_test['threshold'] = ntest / (q * np.linspace(1, ntest, num=ntest))
#     ebh_smaller = [j for j in range(ntest) if df_test.iloc[j, 6] >= df_test.iloc[j,8]]
#     
#     if len(ebh_smaller) == 0:
#         sel_hetero = np.array([]) #return(np.array([]))
#     else:
#         sel_hetero = np.array(df_test.index)[range(np.max(ebh_smaller)+1)] 
#         
#     # eBH(q) with homo boosting
#     df_test = df_test.sort_values(by='eval_homo', ascending=False)
#     df_test['threshold'] = ntest / (q * np.linspace(1, ntest, num=ntest))
#     ebh_smaller = [j for j in range(ntest) if df_test.iloc[j, 7] >= df_test.iloc[j,8]]
#     
#     if len(ebh_smaller) == 0:
#         sel_homo = np.array([]) #return(np.array([]))
#     else:
#         sel_homo = np.array(df_test.index)[range(np.max(ebh_smaller)+1)] 
#     
#     return sel_homo, sel_hetero, np.array(df_test.sort_index()['eval_raw'])
# =============================================================================
        
 
    
    