from DeepPurpose import utils, dataset, CompoundPred
import warnings
import numpy as np
import random 
import pandas as pd
import os
import sys
from DeepPurpose import DTI as models
from utils import weighted_BH, weighted_CS, eval_sel
warnings.filterwarnings("ignore")


X_drugs, X_targets, y = dataset.load_process_DAVIS(path = './data', binary = False, convert_to_log = True, threshold = 30) 

drug_encoding, target_encoding = 'Morgan', 'Conjoint_triad' 

seed = int(sys.argv[1])  
q = int(sys.argv[2]) / 10  
qpop = int(sys.argv[3])

n = len(y) 
 
np.random.seed(seed)
reind = np.random.permutation(n)

X_drugs_train = X_drugs[reind[0:int(n*0.2+1)]]
X_targets_train = X_targets[reind[0:int(n*0.2+1)]]
y_train = y[reind[0:int(n*0.2+1)]]
X_drugs_other = X_drugs[reind[int(1+n*0.2):n]]
X_targets_other = X_targets[reind[int(1+n*0.2):n]]
y_other = y[reind[int(1+n*0.2):n]]

# =============================================================================
# # train the prediction model
# =============================================================================

ttrain, tval, ttest = utils.data_process(X_drugs_train, X_targets_train, y_train, 
                                drug_encoding, target_encoding, 
                                split_method='random', frac=[0.7,0.1,0.2],
                                random_seed = seed) 

config = utils.generate_config(drug_encoding = drug_encoding, 
                        target_encoding = target_encoding, 
                        cls_hidden_dims = [1024,1024,512], 
                        train_epoch = 10, 
                        LR = 0.001, 
                        batch_size = 128,
                        hidden_dim_drug = 128,
                        mpnn_hidden_size = 128,
                        mpnn_depth = 3, 
                        cnn_target_filters = [32,64,96],
                        cnn_target_kernels = [4,8,12]
                        )

model = models.model_initialize(**config)
model.train(ttrain, tval, ttest)


# =============================================================================
# # randomly selet into calibration and test data
# =============================================================================

d_, d__, dother = utils.data_process(X_drugs_other, X_targets_other, y_other, 
                                drug_encoding, target_encoding, 
                                split_method='random', frac=[0,0,1],
                                random_seed = seed) 

all_pred = model.predict(dother)
train_pred = model.predict(ttrain)
p_x = np.exp(2*(all_pred - np.mean(train_pred))) / (1+np.exp(2*(all_pred - np.mean(train_pred))))

in_calib = np.random.binomial(1, p_x, size=len(p_x))

dcalib = dother[pd.Series(in_calib==1).values]
dtest = dother[pd.Series(in_calib==0).values]



# =============================================================================
# # specify selection thresholds
# =============================================================================

testq2 = np.zeros(dtest.shape[0])
testq5 = np.zeros(dtest.shape[0])
testq7 = np.zeros(dtest.shape[0])
testq8 = np.zeros(dtest.shape[0])
testq9 = np.zeros(dtest.shape[0])


for i in range(dtest.shape[0]):
  tenc = dtest['Target Sequence'][i]
  tsub = ttrain['Target Sequence'] == tenc 
  if sum(tsub) == 0:
    allb = ttrain 
  else:
    allb = ttrain[tsub]
 
  q2, q5, q7, q8, q9 = np.quantile(allb['Label'], 0.2), np.quantile(allb['Label'], 0.5), np.quantile(allb['Label'], 0.7), np.quantile(allb['Label'], 0.8), np.quantile(allb['Label'], 0.9)
  testq2[i] = q2
  testq5[i] = q5
  testq7[i] = q7 
  testq8[i] = q8 
  testq9[i] = q9

calibq2 = np.zeros(dcalib.shape[0])
calibq5 = np.zeros(dcalib.shape[0])
calibq7 = np.zeros(dcalib.shape[0])
calibq8 = np.zeros(dcalib.shape[0])
calibq9 = np.zeros(dcalib.shape[0])


for i in range(dcalib.shape[0]):
  tenc = dcalib['Target Sequence'][i]
  tsub = ttrain['Target Sequence'] == tenc
  # print(tsub)
  if sum(tsub) == 0:
    allb = ttrain 
  else:
    allb = ttrain[tsub]
 
  q2, q5, q7, q8, q9 = np.quantile(allb['Label'], 0.2), np.quantile(allb['Label'], 0.5), np.quantile(allb['Label'], 0.7), np.quantile(allb['Label'], 0.8), np.quantile(allb['Label'], 0.9)
  calibq2[i] = q2
  calibq5[i] = q5
  calibq7[i] = q7 
  calibq8[i] = q8 
  calibq9[i] = q9
 

hat_mu_calib = np.array(model.predict(dcalib))
hat_mu_test = np.array(model.predict(dtest))
y_calib = np.array(dcalib["Label"])
w_calib = np.array(1/p_x[in_calib==1] - 1)
y_test = np.array(dtest['Label'])
w_test = np.array(1/p_x[in_calib==0] - 1)


data_calib = pd.DataFrame({"calib_pred": hat_mu_calib, "calib_true": y_calib, "calib_w": w_calib,
                          "q2": calibq2, "q5": calibq5, "q7": calibq7, "q8": calibq8, "q9": calibq9})
data_test = pd.DataFrame({"test_pred": hat_mu_test, "test_true": y_test, "test_w": w_test,
                          "q2": testq2, "q5": testq5, "q7": testq7, "q8": testq8, "q9": testq9})

# subset test data to sample size 5000
test_sub = np.random.permutation(data_test.shape[0])[0:5000]
data_test = data_test.iloc[test_sub,:]
data_test.index = np.arange(5000)

# =============================================================================
# # run testing procedures
# =============================================================================
  
# hat_mu_calib = data_calib['calib_pred']
# y_calib = data_calib['calib_true']
# w_calib = data_calib['calib_w']

hat_mu_test = data_test['test_pred']
y_test = data_test['test_true']
w_test = data_test['test_w']

ntest = data_test.shape[0]
 
# specify c_n+j for test data and c_i for calibration data
cname = 'q'+str(int(qpop))
c_calib = data_calib[cname]
c_test = data_test[cname]

calib_scores_res = y_calib - hat_mu_calib 
calib_scores_clip = 100 * (y_calib > c_calib) + c_calib * (y_calib <= c_calib) - hat_mu_calib
 

test_scores = c_test - hat_mu_test


# weighted BH procedure
BH_res = weighted_BH(calib_scores_res, w_calib, test_scores, w_test, q)   
BH_clip = weighted_BH(calib_scores_clip, w_calib, test_scores,w_test,  q )

# summarize
BH_res_fdp, BH_res_power = eval_sel(BH_res, y_test, c_test)
BH_clip_fdp, BH_clip_power = eval_sel(BH_clip, y_test, c_test)

# conditional calibration 
CS_res_0, CS_res_hete, CS_res_homo, CS_res_dtm = weighted_CS(calib_scores_res, w_calib, 
                                                             test_scores, w_test, q) 
CS_clip_0, CS_clip_hete, CS_clip_homo, CS_clip_dtm = weighted_CS(calib_scores_clip, w_calib,
                                                                 test_scores,w_test,  q )

all_BH = [BH_res, BH_clip]
all_sel = [[CS_res_hete, CS_res_homo, CS_res_dtm], [CS_clip_hete, CS_clip_homo, CS_clip_dtm]]
fdp = [BH_res_fdp, BH_clip_fdp]
power = [BH_res_power, BH_clip_power] 
ndiff = [0] * 2
nsel = [len(BH_res), len(BH_clip)]
nsame = [len(BH_res), len(BH_clip)]

for ii in range(2):
    sels = all_sel[ii]
    tpowers = []
    tfdps = []
    tnsels = []
    tndiffs = []
    tnsames = []
    for jj in range(3):
        tfdp, tpower = eval_sel(sels[jj], y_test, c_test)
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
                       "score": ["res", "clip"] + ["res"]*3 + ["clip"]*3,
                       "method": ["WBH"]*2 + ['WCS.hete', 'WCS.homo', "WCS.dtm"] *2,
                       "q": q, "seed": seed, "qpop": qpop})

save_path = "./DTI_results"
if not os.path.exists(save_path): 
   os.makedirs(save_path)

res.to_csv("./DTI_results/seed"+str(seed)+"q"+str(q)+"qpop"+str(qpop)+".csv")

