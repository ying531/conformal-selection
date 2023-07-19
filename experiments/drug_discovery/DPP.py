from DeepPurpose import utils, dataset, CompoundPred
import warnings
warnings.filterwarnings("ignore")

X_drugs, y, drugs_index = dataset.load_HIV(path = './data')
drug_encoding = 'Morgan'

import numpy as np 
import os
import sys
import pandas as pd 
from utils import weighted_BH, weighted_CS, eval_sel 

seed = int(sys.argv[1]) 
q = int(sys.argv[2]) / 10

np.random.seed(seed)
n = len(y)
reind = np.random.permutation(n)

X_drugs_train = X_drugs[reind[0:int(n*0.4+1)]]
y_train = y[reind[0:int(n*0.4+1)]]
X_drugs_other = X_drugs[reind[int(1+n*0.4):n]]
y_other = y[reind[int(1+n*0.4):n]]

# =============================================================================
# # train prediction model on the training data
# =============================================================================

ttrain, tval, ttest = utils.data_process(X_drug = X_drugs_train, y = y_train, 
                                         drug_encoding = drug_encoding,
                                         split_method='random',frac=[0.7,0.1,0.2],
                                         random_seed = seed)
 
# small neural network
config = utils.generate_config(drug_encoding = drug_encoding, 
                       cls_hidden_dims = [1024,1024,512], 
                       train_epoch = 3, 
                       LR = 0.001, 
                       batch_size = 128,
                       hidden_dim_drug = 128,
                       mpnn_hidden_size = 128,
                       mpnn_depth = 3
                      )
model = CompoundPred.model_initialize(**config)
model.train(ttrain, tval, ttest)


# =============================================================================
# # weighted split into calibration and test data
# =============================================================================

d_, d__, dother = utils.data_process(X_drug = X_drugs_other, y = y_other, 
                                     drug_encoding = drug_encoding,
                                     split_method='random',frac=[0,0,1],
                                     random_seed = seed)

all_pred = model.predict(dother)
train_pred = model.predict(ttrain)

p_x = np.minimum(0.8, np.exp(all_pred - np.mean(train_pred)) / (1+np.exp(all_pred - np.mean(train_pred))))
in_calib = np.random.binomial(1, p_x, size=len(p_x))

dcalib = dother[pd.Series(in_calib==1).values]
dtest = dother[pd.Series(in_calib==0).values]
  

hat_mu_calib = np.array(model.predict(dcalib))
hat_mu_test = np.array(model.predict(dtest))
y_calib = np.array(dcalib["Label"])
w_calib = np.array(1/p_x[in_calib==1] - 1)
y_test = np.array(dtest['Label'])
w_test = np.array(1/p_x[in_calib==0] - 1)

# =============================================================================
# # run testing procedures
# =============================================================================

c = 0
 
calib_scores_res = y_calib - hat_mu_calib
calib_scores_sub = - hat_mu_calib 
calib_scores_clip = 100 * (y_calib > c) + c * (y_calib <= c) - hat_mu_calib
 
test_scores = c - hat_mu_test

 
# ========================= 
# ## weighted BH procedure
# ========================= 

# use scores res, sub, and clip
BH_res = weighted_BH(calib_scores_res, w_calib, test_scores, w_test, q)  
BH_sub = weighted_BH(calib_scores_sub[y_calib <= c], w_calib[y_calib<=c], test_scores, w_test, q) 
BH_clip = weighted_BH(calib_scores_clip, w_calib, test_scores, w_test, q)



# ==================================== 
# ## weighted conformalized selection
# ==================================== 

# use scores res, sub, and clip
WCS_res_0, WCS_res_hete, WCS_res_homo, WCS_res_dtm = weighted_CS(calib_scores_res, 
                                                                 w_calib, test_scores, w_test, q)
WCS_sub_0, WCS_sub_hete, WCS_sub_homo, WCS_sub_dtm = weighted_CS(calib_scores_sub[y_calib <= c],
                                                                 w_calib[y_calib<=c], 
                                                                 test_scores, w_test, q)
WCS_clip_0, WCS_clip_hete, WCS_clip_homo, WCS_clip_dtm = weighted_CS(calib_scores_clip, 
                                                                     w_calib, test_scores,w_test, q)




# =============================================================================
# # summarize FDP, power and selection sizes
# =============================================================================


BH_res_fdp, BH_res_power = eval_sel(BH_res, y_test, np.array([c]*len(y_test)))
BH_sub_fdp, BH_sub_power = eval_sel(BH_sub, y_test, np.array([c]*len(y_test)))
BH_clip_fdp, BH_clip_power = eval_sel(BH_clip, y_test, np.array([c]*len(y_test))) 


all_BH = [BH_res, BH_sub, BH_clip]
all_sel = [[WCS_res_hete, WCS_res_homo, WCS_res_dtm], 
           [WCS_sub_hete, WCS_sub_homo, WCS_sub_dtm],
           [WCS_clip_hete, WCS_clip_homo, WCS_clip_dtm]]
fdp = [BH_res_fdp, BH_sub_fdp, BH_clip_fdp]
power = [BH_res_power, BH_sub_power, BH_clip_power] 
ndiff = [0] * 3
nsel = [len(BH_res), len(BH_sub), len(BH_clip)]
nsame = [len(BH_res), len(BH_sub), len(BH_clip)]

for ii in range(3):
    sels = all_sel[ii]
    tpowers = []
    tfdps = []
    tnsels = []
    tndiffs = []
    tnsames = []
    for jj in range(3):
        tfdp, tpower = eval_sel(sels[jj], y_test, np.array([c]*len(y_test)))
        tpowers.append(tpower)
        tfdps.append(tfdp)
        tnsels.append(len(sels[jj]))
        tndiffs.append(len(np.setxor1d(all_BH[ii], sels[jj])))
        tnsames.append(len(np.intersect1d(all_BH[ii], sels[jj])))
    fdp += tfdps
    power += tpowers
    ndiff += tndiffs
    nsel += tnsels
    nsame += tnsames
 

res = pd.DataFrame({"FDP": fdp, "power": power, "nsel": nsel, "ndiff": ndiff, "nsame": nsame,
                       "score": ["res", "sub", "clip"] + ["res"]*3 + ["sub"]*3 + ["clip"]*3,
                       "method": ["WBH"]*3 + ['WCS.hete', 'WCS.homo', "WCS.dtm"] *3,
                                   "q": q, "seed": seed})

save_path = "./DPP_results"
if not os.path.exists(save_path): 
   os.makedirs(save_path)

res.to_csv("./DPP_results/seed"+str(seed)+"q"+str(q)+".csv")








