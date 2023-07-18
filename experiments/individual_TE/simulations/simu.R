suppressPackageStartupMessages(library(grf))

######## ========================= ########
######## simulation configurations ########
######## ========================= ########

args <- commandArgs(trailingOnly = TRUE)
rho.id <- as.integer(args[1])
setting <- as.integer(args[2])
seed <- as.integer(args[3])

source("../util_qt.R")
source("../util_simu.R")

rhos = c(0, 0.9)
rho = rhos[rho.id]
noise.sig = 0.2
ntest = 100       # test sample size
n = 1000          # training + calibration sample size
p = 10            # feature dimension
q = 0.1           # FDR nominal level

# target: predict counterfactual Y(tt)
# training and calibration is from Y(tt) | T=tt
# testing is from Y(tt) | T=1-tt
tt = 1 # counterfactual treatment
pp = mean(gen.data(100000, p, rho, setting, obs=-1, sig=noise.sig)$ex)


# set random seed
set.seed(seed)

######## =============== ########
######## data generation ########
######## =============== ########

train.data = gen.data(n*0.75, p, rho, setting, obs=tt, sig=noise.sig)
calib.data = gen.data(n*0.25, p, rho, setting, obs=tt, sig=noise.sig)
test.data = gen.data(ntest, p, rho, setting, obs=1-tt, sig=noise.sig)


X.train = (train.data$X[train.data$T==tt,])[1:(n*0.75),]
Y.train = (train.data$Y[train.data$T==tt])[1:(n*0.75)]

X.calib = (calib.data$X[calib.data$T==tt,])[1:(n*0.25),]
Y.calib = (calib.data$Y[calib.data$T==tt])[1:(n*0.25)]
ex.calib = (calib.data$ex[calib.data$T==tt])[1:(n*0.25)]

ex.test = (test.data$ex[test.data$T==1-tt])[1:ntest]
X.test = (test.data$X[test.data$T==1-tt,])[1:ntest,]
Yobs.test = (test.data$Y[test.data$T==1-tt])[1:ntest]
if (tt==1){Ycon.test = (test.data$Y1[test.data$T==1-tt])[1:ntest]}
if (tt==0){Ycon.test = (test.data$Y0[test.data$T==1-tt])[1:ntest]}
truths = (test.data$Y1[test.data$T==1-tt])[1:ntest] <= (test.data$Y0[test.data$T==1-tt])[1:ntest]

# use ground truth of e(x)
cal.weight = ((1-pp) * ex.calib / (pp * (1-ex.calib)))^{1-2*tt}
test.weight  = ((1-pp) * ex.test / (pp * (1-ex.test)))^{1-2*tt}


######## ====================================== ########
######## nonconformity score by conditional cdf ########
######## ====================================== ########

### train the quantile regression model
dtrain = data.frame(cbind(X.train, Y.train))
covariate_name = sapply(1:p, function(x) paste("X",x,sep=''))
response_name = "Y"
colnames(dtrain) = c(covariate_name, response_name)
mdl.qrf = qtl_train(dtrain, covariate_name, response_name,
                    method='qrf', other.args= list("num.threads" = 1))

# ======================================================= #
### compute scores (the fitted cdf) on calibration set
cal.score = cdf_pred(mdl.qrf, data.frame(X.calib), Y.calib, method='qrf', eps=0.0001)

### test scores at Y(1-tt)
test.score = cdf_pred(mdl.qrf, data.frame(X.test), Yobs.test, method='qrf', eps=0.0001)

R.wBH = w_BH(cal.score, test.score, cal.weight, test.weight, q)
R.wCC.hete = w_CS(cal.score, test.score, cal.weight, test.weight, q, rand = 'hete')
R.wCC.homo = w_CS(cal.score, test.score, cal.weight, test.weight, q, rand = 'homo')
R.wCC.dtm = w_CS(cal.score, test.score, cal.weight, test.weight, q, rand = 'dtm')

# summarize results
cdf.res = rbind(eval.FDR(R.wBH, truths), eval.FDR(R.wCC.hete, truths),
                eval.FDR(R.wCC.homo, truths), eval.FDR(R.wCC.dtm, truths))
cdf.res$ndiff = c(0, length(setdiff(R.wBH, R.wCC.hete)),
                  length(setdiff(R.wBH, R.wCC.homo)), length(setdiff(R.wBH, R.wCC.dtm)))
cdf.res$nsame = c(0, length(intersect(R.wBH, R.wCC.hete)),
                  length(intersect(R.wBH, R.wCC.homo)), length(intersect(R.wBH, R.wCC.dtm)))
cdf.res$method = c("WBH", "WCS.hete", "WCS.homo", "WCS.dtm")
cdf.res$score = rep("cdf", 4)

# ======================================================= #
### the oracle procedure with true c.cdf
# compute oracle score, the conditional cdf at (X,Y(0))
if (setting %in% c(1,2,3)){
  test.mu1.x = 4/((1+exp(-12*(X.test[,1]-0.5)))*(1+exp(-12*(X.test[,2]-0.5))))
  test.sig1.x = (0.2 - log(X.test[,1])) * noise.sig
  test.cq.x = pnorm((Yobs.test - test.mu1.x)/test.sig1.x) * (Yobs.test >= 0)
  cal.mu1.x = 4/((1+exp(-12*(X.calib[,1]-0.5)))*(1+exp(-12*(X.calib[,2]-0.5))))
  cal.sig1.x = (0.2- log(X.calib[,1])) * noise.sig
  cal.cq.x = pnorm((Y.calib - cal.mu1.x)/cal.sig1.x) * (Y.calib >= 0)
}
if (setting %in% c(4,5,6)){
  test.mu1.x = 4/((1+exp(-12*(X.test[,1]-0.5)))*(1+exp(-12*(X.test[,2]-0.5))))
  test.sig1.x = (0.2 - log(X.test[,1])) * noise.sig
  test.cq.x = 0.9 * pnorm((Yobs.test - test.mu1.x)/test.sig1.x) * (Yobs.test > 0) + 0.1 * pnorm((2*Yobs.test +1)/noise.sig)
  cal.mu1.x = 4/((1+exp(-12*(X.calib[,1]-0.5)))*(1+exp(-12*(X.calib[,2]-0.5))))
  cal.sig1.x = (0.2 - log(X.calib[,1])) * noise.sig
  cal.cq.x = 0.9 * pnorm((Y.calib - cal.mu1.x)/cal.sig1.x) * (Y.calib > 0) + 0.1 * pnorm((2*Y.calib +1)/noise.sig)
}
if (setting %in% c(7,8,9)){
  test.mu1.x = 0.1 + 3/((1+exp(-3*(X.test[,1]-0.5)))*(1+exp(-3*(X.test[,2]-0.5))))
  test.sig1.x = (0.2 - log(X.test[,1])) * noise.sig
  test.cq.x = pnorm((Yobs.test - test.mu1.x)/test.sig1.x) * (Yobs.test > 0)
  cal.mu1.x = 0.1 + 3/((1+exp(-3*(X.calib[,1]-0.5)))*(1+exp(-3*(X.calib[,2]-0.5))))
  cal.sig1.x = (0.2 - log(X.calib[,1])) * noise.sig
  cal.cq.x = pnorm((Y.calib - cal.mu1.x)/cal.sig1.x) * (Y.calib > 0)
}

# oracle
orc.R.wBH = w_BH(cal.cq.x, test.cq.x, cal.weight, test.weight, q)
orc.R.wCC.hete = w_CS(cal.cq.x, test.cq.x, cal.weight, test.weight, q, rand = 'hete')
orc.R.wCC.homo = w_CS(cal.cq.x, test.cq.x, cal.weight, test.weight, q, rand = 'homo')
orc.R.wCC.dtm = w_CS(cal.cq.x, test.cq.x, cal.weight, test.weight, q, rand = 'dtm')

# summarize results
orc.res = rbind(eval.FDR(orc.R.wBH, truths), eval.FDR(orc.R.wCC.hete, truths),
                eval.FDR(orc.R.wCC.homo, truths), eval.FDR(orc.R.wCC.dtm, truths))
orc.res$ndiff = c(0, length(setdiff(orc.R.wBH, orc.R.wCC.hete)),
                  length(setdiff(orc.R.wBH, orc.R.wCC.homo)), length(setdiff(orc.R.wBH, orc.R.wCC.dtm)))
orc.res$nsame = c(0, length(intersect(orc.R.wBH, orc.R.wCC.hete)),
                  length(intersect(orc.R.wBH, orc.R.wCC.homo)), length(intersect(orc.R.wBH, orc.R.wCC.dtm)))
orc.res$method = c("WBH", "WCS.hete", "WCS.homo", "WCS.dtm")
orc.res$score = rep("orc", 4)

######## =========================================== ########
######## nonconformity score by conditional quantile ########
######## =========================================== ########

### train the quantile regression model
mdl.qrf = quantile_forest(X.train, Y.train, num.thread=1)

# ======================================================= #
### compute scores (the fitted cdf) on calibration set
cqr.res = data.frame()
for (cqr.q in c(0.2, 0.5, 0.8)){
  cal.score = Y.calib - predict(mdl.qrf, X.calib, quantiles = 0.2, num.threads=1)
  ### test scores at Y(1-tt)
  test.score = Yobs.test - predict(mdl.qrf, X.test, quantiles = 0.2, num.threads=1)

  cqr.R.wBH = w_BH(cal.score, test.score, cal.weight, test.weight, q)
  cqr.R.wCC.hete = w_CS(cal.score, test.score, cal.weight, test.weight, q, rand = 'hete')
  cqr.R.wCC.homo = w_CS(cal.score, test.score, cal.weight, test.weight, q, rand = 'homo')
  cqr.R.wCC.dtm = w_CS(cal.score, test.score, cal.weight, test.weight, q, rand = 'dtm')

  # summarize results
  cqr.res.t = rbind(eval.FDR(cqr.R.wBH, truths), eval.FDR(cqr.R.wCC.hete, truths),
                    eval.FDR(cqr.R.wCC.homo, truths), eval.FDR(cqr.R.wCC.dtm, truths))
  cqr.res.t$ndiff = c(0, length(setdiff(cqr.R.wBH, cqr.R.wCC.hete)),
                    length(setdiff(cqr.R.wBH, cqr.R.wCC.homo)), length(setdiff(cqr.R.wBH, cqr.R.wCC.dtm)))
  cqr.res.t$nsame = c(0, length(intersect(cqr.R.wBH, cqr.R.wCC.hete)),
                      length(intersect(cqr.R.wBH, cqr.R.wCC.homo)), length(intersect(cqr.R.wBH, cqr.R.wCC.dtm)))
  cqr.res.t$method = c("WBH", "WCS.hete", "WCS.homo", "WCS.dtm")
  cqr.res.t$score = rep(paste("cqr_",cqr.q,sep=''), 4)
  cqr.res = rbind(cqr.res, cqr.res.t)
}

######## ======================================= ########
######## nonconformity score by conditional mean ########
######## ======================================= ########

### train the conditional regression model
hat.mu = regression_forest(X.train, Y.train, num.threads=1)

cal.score = Y.calib - predict(hat.mu, newdata = X.calib)$predictions
test.score = Yobs.test - predict(hat.mu, newdata = X.test)$predictions

reg.wBH = w_BH(cal.score, test.score, cal.weight, test.weight, q)
reg.wCC.hete = w_CS(cal.score, test.score, cal.weight, test.weight, q, rand = 'hete')
reg.wCC.homo = w_CS(cal.score, test.score, cal.weight, test.weight, q, rand = 'homo')
reg.wCC.dtm = w_CS(cal.score, test.score, cal.weight, test.weight, q, rand = 'dtm')

# summarize results
reg.res = rbind(eval.FDR(reg.wBH, truths), eval.FDR(reg.wCC.hete, truths),
                eval.FDR(reg.wCC.homo, truths), eval.FDR(reg.wCC.dtm, truths))
reg.res$ndiff = c(0, length(setdiff(reg.wBH, reg.wCC.hete)),
                  length(setdiff(reg.wBH, reg.wCC.homo)), length(setdiff(reg.wBH, reg.wCC.dtm)))
reg.res$nsame = c(0, length(intersect(reg.wBH, reg.wCC.hete)),
                  length(intersect(reg.wBH, reg.wCC.homo)), length(intersect(reg.wBH, reg.wCC.dtm)))
reg.res$method = c("WBH", "WCS.hete", "WCS.homo", "WCS.dtm")
reg.res$score = rep("reg", 4)

######## ======================================= ########
########     save results with configurations    ########
######## ======================================= ########

this.res = rbind(cdf.res, orc.res, cqr.res, reg.res)
this.res$seed = seed
this.res$setting = setting
this.res$model = paste("setting", floor((setting-1)/3)+1, ifelse(rho.id==1, ".ind", ".corr"), sep='')
this.res$rho = rho
if (setting %in% c(1,4,7)){this.res$coupling = 'indep. coupl.'}
if (setting %in% c(2,5,8)){this.res$coupling = 'pos. coupl.'}
if (setting %in% c(3,6,9)){this.res$coupling = 'neg. coupl.'}

save.dir = "./results/"
if (!dir.exists(save.dir)){
  dir.create(save.dir, recursive = TRUE)
}
path = paste("./results/setting", setting, "rho", rho.id, "seed", seed, ".csv", sep='')
write.csv(this.res, path)




