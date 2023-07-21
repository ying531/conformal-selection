<h1 align="center">
<p>  Python package ConSelect
</h1>

Python package  `ConfSelect`  implements the Weighted Conformalized Selection procedure in our paper "Model-free selective inference under covariate shift via weighted conformal p-values" (Jin and Candes, 2023).
 
### Installing the  package

The Python package can be installed in two ways:

1. Local install 

```bash
cd confselect-python
python setup.py install
```

2. Install from GitHub 

```bash
pip install -e "git+https://github.com/ying531/conformal-selection#egg=conformal-selection&subdirectory=confselect-python" 
```


## Documentation
 

This  package `ConfSelect` implements three procedures/functions for selecting promising candidiates whose unobserved outcomes are above user-sepcified values while controlling the FDR. 

- `weighted_CS`: Weighted Conformalized Selection, which controls the FDR in finite sample under covariate shift. Its arguments are as below:

    ```R
    weighted_CS(cal.score, # vector of scores V_i = V(X_i,Y_i) for calibration data
                test.score, # vector of scores hat{V}_n+j = V(X_n+j, c_n+j) for test data
                cal.weight, # vector of weights w(X_i) for calibration data, where w() is the covariate shift from calibration to test distribution 
                test.weight, # vector of weights w(X_n+j) for test data
                q = 0.1, # nominal FDR level 
                rand = "hete" # pruning method, 'hete' for heterogeneous pruning, 'homo' for homogeneous pruning, 'dtm' for deterministic pruning
                )
    ```
    This function returns a vector of indices among the test scores that are selected, and the computed weighted conformal p-values.


- `weighted_BH`: The Benjamini Hochberg procedure applied to our weighted conformal p-values, which computes faster but does not necessarily control the FDR in finite sample under covariate shift. Its arguments are as below:

    ```R
    weighted_BH(cal.score, # vector of scores V_i = V(X_i,Y_i) for calibration data
                test.score, # vector of scores hat{V}_n+j = V(X_n+j, c_n+j) for test data
                cal.weight, # vector of weights w(X_i) for calibration data, where w() is the covariate shift from calibration to test distribution 
                test.weight, # vector of weights w(X_n+j) for test data
                q = 0.1, # nominal FDR level  
                )
    ```
    This function returns a vector of indices among the test scores that are selected, and the computed weighted conformal p-values.


- `Conformal_select`: The vanilla version for selecting promising candidates with i.i.d. or exchangeable data, introduced earlier in the paper [Selection by Prediction with Conformal P-values](https://arxiv.org/abs/2210.01408) by the authors. Its arguments are as below:

    ```R
    Conformal_select(cal.score, # vector of scores V_i = V(X_i,Y_i) for calibration data
                     test.score, # vector of scores hat{V}_n+j = V(X_n+j, c_n+j) for test data
                     q = 0.1, # nominal FDR level  
                     )
    ```
    This function returns a vector of indices among the test scores that are selected, and the computed conformal p-values.

 ### Use example

Below we provide a use example with pseudo-functions that generate data and train a prediction model.

```Python
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
import ConfSelect
from ConfSelect.confselect import weighted_BH, weighted_CS, conformal_select

# =====================================================
# create dataset with pseudo functions
# =====================================================

Xtest, Ytest, Xcalib, Ycalib, Xtrain, Ytrain = ...  # some function to generate data
calib_weights = ...  # some function to compute W_i = w(X_i) for calibration data 
test_weights = ...  # some function to compute W_n+j = w(X_n+j) for test data
c_test = ... # some function to generate c_n+j for test data
  
# ==================================================
# training a ML prediction model, e.g. random forest
# ==================================================
rf = RandomForestRegressor().fit(Xtest, Ytest)
# compute nonconformity scores using V(x,y) = y - mu(x)
calib_scores = Ycalib - rf.predict(Xcalib)   
test_scores = c_test  - rf.predict(Xtest) 

# ==========================================
# Run our procedures in the package at q=0.1
# ==========================================

# BH with weighted conformal p-values
wBH, w_pvals = weighted_BH(calib_scores, calib_weights, test_scores, test_weights, q = 0.1)
# WCS with heterogeneous pruning 
WCS_hete, w_pvals = weighted_BH(calib_scores, calib_weights, 
                                test_scores, test_weights, q = 0.1, rand = 'hete')
# WCS with homogenous pruning 
WCS_homo, w_pvals = weighted_BH(calib_scores, calib_weights, 
                                test_scores, test_weights, q = 0.1, rand = 'homo')
# WCS with deterministic pruning 
WCS_dtm, w_pvals = weighted_BH(calib_scores, calib_weights, 
                                test_scores, test_weights, q = 0.1, rand = 'dtm')
```





## Citation 

Please use the following bibliography for citing our method and package. 


```
@article{jin2023model,
  title={Model-free selective inference under covariate shift via weighted conformal p-values},
  author={Jin, Ying and Cand\`es, Emmanuel J},
  journal={arXiv preprint arXiv:2307.09291},
  year={2023}
}
```

The unweighted version is proposed in the following reference:

```
@article{jin2022selection,
  title={Selection by prediction with conformal p-values},
  author={Jin, Ying and Cand{\`e}s, Emmanuel J},
  journal={arXiv preprint arXiv:2210.01408},
  year={2022}
}
```