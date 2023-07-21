#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:07:05 2022

@author: ying
"""

import numpy as np
import pandas as pd   
 

def gen_data(Wset, n, a): 
    Wi = Wset[np.random.choice(range(50), n),:]
    Vi = np.random.normal(size=n*50).reshape((n,50))
    Xi = np.sqrt(a) * Vi + Wi
    return(Xi)
    
def gen_weight(Xi, theta):
    linx = Xi @ theta   
    wx = np.exp(linx) / (1+ np.exp(linx))    
    return(wx)
    
def gen_data_weighted(Wset, n, a, theta):
    Wi = Wset[np.random.choice(range(50), n),:]
    Vi = np.random.normal(size=n*50).reshape((n,50))
    Xi = np.sqrt(a) * Vi + Wi
    wx = gen_weight(Xi, theta)
    sel = np.random.binomial(n=1, p=wx[:,0])
    X_sel = Xi[sel==1,:]
    X = X_sel
    
    while X.shape[0] < n:
        Wi = Wset[np.random.choice(range(50), n),:]
        Vi = np.random.normal(size=n*50).reshape((n,50))
        Xi = np.sqrt(a) * Vi + Wi
        wx = gen_weight(Xi, theta)
        sel = np.random.binomial(n=1, p=wx[:,0])
        X_sel = Xi[sel==1,:]
        X = np.concatenate((X, X_sel))
    
    X = X[range(n),:]
    
    return(X)


def BH(pvals, q=0.1):
    mh = len(pvals)
    df_all = pd.DataFrame({"id": range(mh), "pval": pvals})
    df_sorted = df_all.sort_values(by = "pval")
    df_sorted['threshold'] = (1+np.arange(mh)) * q / mh
    id_in = [j for j in range(mh) if df_sorted.iloc[j,1] <= df_sorted.iloc[j,2]]
    if len(id_in)==0:
        return(np.array([]))
    else:
        return(np.array(df_sorted.index[range(np.max(id_in)+1)]))
    

def weighted_BH(calib_scores, calib_weights, test_scores, test_weights, q = 0.1):
    pvals = np.zeros(len(test_scores))
    df_all = pd.concat((pd.DataFrame({"score": calib_scores, "weight": calib_weights, "cal": 1}), pd.DataFrame({"score": test_scores, "weight": test_weights, "cal": 0})))
    df_sorted = df_all.sort_values(by = 'score')
    all_sorted = np.array(df_sorted)
    sum_calib_weight = np.sum(calib_weights)
    
    p_vals = []
    for j in range(all_sorted.shape[0]):
        if all_sorted[j,2] == 0:
            p_vals.append( (np.sum(all_sorted[range(j),1] * all_sorted[range(j),2]) + all_sorted[j,1] * np.random.uniform(size=1)[0]) / (sum_calib_weight + all_sorted[j,1]) )
        else:
            p_vals.append(-1)
            
    df_sorted['pvals'] = p_vals
        
    df_test = df_sorted[df_sorted['cal'] == 0]
    df_test_sorted = df_test.sort_values(by='pvals')
    
    # BH(q)
    ntest = len(test_scores)
    df_test_sorted['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
    idx_smaller = [j for j in range(ntest) if df_test_sorted.iloc[j,3] <= df_test_sorted.iloc[j,4]]
    
    if len(idx_smaller) == 0:
        return(np.array([]))
    else:
        idx_sel = np.array(df_test_sorted.index[range(np.max(idx_smaller)+1)])
        return(idx_sel)
        

def weighted_CS(calib_scores, calib_weights, test_scores, test_weights, q = 0.1):
    sum_calib_weight = np.sum(calib_weights)
    
    ntest = len(test_scores)
    # sel_0 = np.zeros((ntest, ntest)) # row j indicates hat{R}_j
    Rj_sizes = np.zeros(ntest)
    w_pvals = np.zeros(ntest)
    xis = np.random.uniform(size=ntest)
    
    for j in range(ntest): 
        # compute all other pvals 
        pval_j = np.zeros(ntest)
        for k in range(ntest):
            if k != j:
                pval_j[k] =  np.sum(calib_weights[calib_scores < test_scores[k]]) + test_weights[k] * (test_scores[j] < test_scores[k]) 
        pval_j = pval_j / (sum_calib_weight + test_weights[j])
        w_pvals[j] = ( np.sum(calib_weights[calib_scores < test_scores[j]]) + (np.sum(calib_weights[calib_scores == test_scores[j]]) + test_weights[j])  * np.random.uniform(size=1)[0] ) / (sum_calib_weight + test_weights[j])
        
        # run BH
        df_j = pd.DataFrame({"id": range(ntest), "pval": pval_j}).sort_values(by = 'pval')
        df_j['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
        idx_small_j = [s for s in range(ntest) if df_j.iloc[s, 1] <= df_j.iloc[s, 2]]
        Rj = np.array(df_j['id'])[range(np.max(idx_small_j)+1)] 
        Rj_sizes[j] = len(Rj)
        
    Cj = q * Rj_sizes / ntest
    
    df_all = pd.DataFrame({"id": range(ntest), "pval": w_pvals, "c": Cj, 
                           "hete_Rj": Rj_sizes * xis, 
                           "homo_Rj": Rj_sizes * np.random.uniform(size=1)[0], 
                           "Rj": Rj_sizes})
    
    pj_sel0 = w_pvals[w_pvals <= Cj]
    
    
    if len(pj_sel0) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]) 
    else:
        # heterogeneous pruning
        df_sel0 = df_all[df_all['pval'] <= df_all['c']].sort_values(by='hete_Rj')
        df_sel0['threshold'] = np.linspace(1, df_sel0.shape[0], num = df_sel0.shape[0])
        smaller = [j for j in range(df_sel0.shape[0]) if df_sel0.iloc[j,3] <= df_sel0.iloc[j,6]]
        if len(smaller) == 0:
            idx_sel_hete = np.array([])
        else:
            idx_sel_hete = np.array(df_sel0['id'])[range(np.max(smaller)+1)]
        
        # homogeneous pruning
        df_sel0 = df_all[df_all['pval'] <= df_all['c']].sort_values(by='homo_Rj')
        df_sel0['threshold'] = np.linspace(1, df_sel0.shape[0], num = df_sel0.shape[0])
        smaller = [j for j in range(df_sel0.shape[0]) if df_sel0.iloc[j,4] <= df_sel0.iloc[j,6]]
        if len(smaller) == 0:
            idx_sel_homo = np.array([])
        else:
            idx_sel_homo = np.array(df_sel0['id'])[range(np.max(smaller)+1)]
        
        # deterministic pruning
        df_sel0 = df_all[df_all['pval'] <= df_all['c']].sort_values(by='homo_Rj')
        df_sel0['threshold'] = np.linspace(1, df_sel0.shape[0], num = df_sel0.shape[0])
        smaller = [j for j in range(df_sel0.shape[0]) if df_sel0.iloc[j,5] <= df_sel0.iloc[j,6]]
        if len(smaller) == 0:
            idx_sel_dete = np.array([])
        else:
            idx_sel_dete = np.array(df_sel0['id'])[range(np.max(smaller)+1)]
            
            
        return np.array(df_sel0['id']), idx_sel_hete, idx_sel_homo, idx_sel_dete #, Cj
    
    
    
def eval_sel(sel_idx, ys, cs):
    if len(sel_idx) == 0:
        fdp = 0
        power = 0
    else:
        fdp = np.sum(ys[sel_idx] <= cs[sel_idx]) / len(sel_idx)
        power = np.sum(ys[sel_idx] > cs[sel_idx]) / sum(ys > cs) 
    return fdp, power
    
    
    
     
    
    
    
    
    

















