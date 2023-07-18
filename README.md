<h1 align="center">
<p> conformal-selection
</h1>

This repository hosts the R package, `ConfSelect`, that implements the Weighted Conformalized Selection procedure in our paper [Model-free selective inference under covariate shift via weighted conformal p-values](). It also hosts reproduction code for numerical experiments in the paper. 


## Installing R package `ConfSelect`

1. Install the [devtools](https://github.com/hadley/devtools) package using `install.packages("devtools")`.
2. Install the latest development version using `devtools::install_github("ying531/conformal-selection")`.
 

## Documentation

This R package `ConfSelect` implements three procedures/functions for selecting promising candidiates whose unobserved outcomes are above user-sepcified values while controlling the FDR. 

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
    This function returns a vector of indices among the test scores that are selected.


- `weighted_BH`: The Benjamini Hochberg procedure applied to our weighted conformal p-values, which computes faster but does not necessarily control the FDR in finite sample under covariate shift. Its arguments are as below:

    ```R
    weighted_BH(cal.score, # vector of scores V_i = V(X_i,Y_i) for calibration data
                test.score, # vector of scores hat{V}_n+j = V(X_n+j, c_n+j) for test data
                cal.weight, # vector of weights w(X_i) for calibration data, where w() is the covariate shift from calibration to test distribution 
                test.weight, # vector of weights w(X_n+j) for test data
                q = 0.1, # nominal FDR level  
                )
    ```
    This function returns a vector of indices among the test scores that are selected.


- `Conformal_select`: The vanilla version for selecting promising candidates with i.i.d. or exchangeable data, introduced earlier in the paper [Selection by Prediction with Conformal P-values](https://arxiv.org/abs/2210.01408) by the authors. Its arguments are as below:

    ```R
    Conformal_select(cal.score, # vector of scores V_i = V(X_i,Y_i) for calibration data
                     test.score, # vector of scores hat{V}_n+j = V(X_n+j, c_n+j) for test data
                     q = 0.1, # nominal FDR level  
                     )
    ```
    This function returns a vector of indices among the test scores that are selected.


## Reproduction code

The folder `/experiments/` hosts reproduction code for numerical experiments in our paper. 

#### 1. Individual treatment effects

`/experiments/individual_TE/` reproduces the simulations and real data analysis in Section 4 of our paper. It has two sub-folders:

- `./simulations/` hosts the reproduction code for simulations in Section 4.4. 
- `./realdata/` hosts the processed real dataset and the script for analysis and plots in Section 4.5. 

To run the scripts in the two folders, the R package `grf` needs to be installed. 

The simulation script is in `simu.R`. It takes three inputs, `--rho.id` for whether the features are corrlated (=2) or not (=1), `--setting` in 1-9 for the data generating process, and `--seed` for the random seed. Here, `setting = 1,2,3` corresponds to Setting 1 with independent, positive, negative coupling in Figure 2,3,4 in our paper, and so on for `setting = 4,5,6` for Setting 2, and `setting = 7,8,9` for Setting 3. For a single run with `Setting 1.corr` and `indep. coupl.` in our paper, run the following in command line:

```bash
cd experiments/individual_TE/simulations
Rscript simu.R 1 1 1
```

One run as above saves a csv file in path `/experiments/individual_TE/simulations/results/setting1rho1seed1.csv`. It stores the FDP and power of all procedures and score functions used in our experiments, as well as the configuration of this run (the facet column argument in Figure 2,3,4 is in the `model` column, and the facet row argument is in the `coupling` column). 

You can also submit multiple runs in parallel by running the bash file via

```bash
sh run-batch.sh
```

This runs all configurations of the settings with seeds from 1 to 100, and saves the results for each run. 



#### 2. Drug discovery 




#### 3. Outlier detection

`experiments/outlier_detection` hosts the reproduction code for simulations and real data experiments in Section 6 of the paper for outlier detection. These experiments are currently implemented in Python for the convenience of training machine learning models. 

- `./simulations/` hosts the reproduction code for simulations in Section 6.1. 
- `./realdata/` hosts the dataset and reproduction code for experiments in Section 6.2. 

To run these experiments, python packages `numpy`, `pandas`, `scikit-learn` need to be installed.

The simulation script is in `./simulations/simu.py`. It takes three arguments, `--sig_id` for signal strength id from 1 to 9 (which corresponds to signal strength from 1 to 4 in our paper), `--out_prop` for the proportion of outliers in the test data from 1 to 5 (which corresponds to outlier prop from 0.1 to 0.5 in our paper), and `--seed` for the random seed. For a single run with signal strength 2.5, outlier proportion 0.2, and seed 1 in our paper, run the following in command line:

```bash
cd experiments/outlier_detection/simulations
Python3 simu.py 5 2 1
```

Each run takes a couple of minutes to complete. You can also submit batched jobs in parallel by running `sh run-batch.sh` in this folder. 


The real data experiment script is in `./realdata/bank.py`, and the data is in `./realdata/bank.csv` which was downloaded from public resources. The script takes two arguments, `--q` for the nominal FDR level (the input is always an integer, e.g., set it as `1` for nominal level 0.1), and `--seed` for the random seed. To execute one run of our real data experiment at FDR level q=0.2 and random seed 16, run the following commands:

```bash
cd experiments/outlier_detection/realdata
Python3 bank.py 2 16
```

Parallel jobs with bash submission can be similarly configured. 


## Citation 

Please use the following bibliography for citing our method and package. 


```
@article{jin2023model,
  title={Model-free selective inference under covariate shift via weighted conformal p-values},
  author={Jin, Ying and Cand{\`e}s, Emmanuel J},
  journal={arXiv preprint arXiv:2210.01408},
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