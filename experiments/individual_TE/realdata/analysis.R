library(tidyverse)
library(grf)

source("../util_qt.R")
source("../util_simu.R")

library("scales")
reverselog_trans <- function(base = exp(1)) {
  trans <- function(x) -log(x, base)
  inv <- function(x) base^(-x)
  trans_new(paste0("reverselog-", format(base)), trans, inv,
            log_breaks(base = base),
            domain = c(1e-100, Inf))
}


set.seed(1)

org.data = read.csv("./acic_data.csv")
n = nrow(org.data)
X = org.data %>% select(-Z,-Y) %>% as.matrix

reind = sample(n,n)

# counterfactual: test T=0, training T=1
# training data
Ttrain = org.data$Z[reind[1:5000]]
Xtrain = X[reind[1:5000],]
Ytrain = org.data$Y[reind[1:5000]]

# calibration data
Tcalib = org.data$Z[reind[5001:8000]]
Xcalib = X[reind[5001:8000],][Tcalib==1,]
Ycalib = org.data$Y[reind[5001:8000]][Tcalib==1]

# test data
Ttest = org.data$Z[reind[10001:n]]
Xtest = X[reind[10001:n],][Ttest==0,]
Ytest = org.data$Y[reind[10001:n]][Ttest==0]

## train propensity score
ps.glm = glm(Ttrain~., data = data.frame(Xtrain), family='binomial')
ps.rf = regression_forest(Xtrain, Ttrain, num.threads=1)
ps.glm.pred = predict(ps.glm, type = 'response')
ps.rf.pred = predict(ps.rf)$predictions

## compute cqr scores
qt.rf = quantile_forest(Xtrain[Ttrain==1,], Ytrain[Ttrain==1], num.threads=1)
cal.score = Ycalib - predict(qt.rf, Xcalib, quantiles = 0.2)[,1]
test.score = Ytest - predict(qt.rf, Xtest, quantiles = 0.2)[,1]
## compute regression scores
reg.rf = regression_forest(Xtrain[Ttrain==1,], Ytrain[Ttrain==1], num.threads=1)
cal.score = Ycalib - predict(reg.rf, Xcalib)$predictions
test.score = Ytest - predict(reg.rf, Xtest)$predictions

# compute weights
pp = mean(Ttrain)
ex.calib = predict(ps.rf)$prediction
ex.test = predict(ps.rf)$prediction

cal.weight = ((1-pp) * ex.calib / (pp * (1-ex.calib)))^{-1}
test.weight  = ((1-pp) * ex.test / (pp * (1-ex.test)))^{-1}


# weighted p-values
pvals = rep(0, length(test.score))
Us = runif(length(test.score))
csum = sum(cal.weight)
for (i in 1:length(test.score)){
  pvals[i] = (sum(cal.weight[cal.score<test.score[i]]) + (test.weight[i] + sum(cal.weight[cal.score==test.score[i]])) * Us[i]) / (test.weight[i] + csum)
}
m = nrow(Xtest)
test.all = data.frame(Xtest)
test.all$pval = pvals

# run testing procedures
R.wBH_1 = w_BH(cal.score, test.score, cal.weight, test.weight, q=0.1)
R.wCC.hete_1 = w_CS(cal.score, test.score, cal.weight, test.weight, q=0.1, rand = 'hete')
R.wCC.homo_1 = w_CS(cal.score, test.score, cal.weight, test.weight, q=0.1, rand = 'homo')
R.wCC.dtm_1 = w_CS(cal.score, test.score, cal.weight, test.weight, q=0.1, rand = 'dtm')


R.wBH_2 = w_BH(cal.score, test.score, cal.weight, test.weight, q=0.2)
R.wCC.hete_2 = w_CS(cal.score, test.score, cal.weight, test.weight, q=0.2, rand = 'hete')
R.wCC.homo_2 = w_CS(cal.score, test.score, cal.weight, test.weight, q=0.2, rand = 'homo')
R.wCC.dtm_2 = w_CS(cal.score, test.score, cal.weight, test.weight, q=0.2, rand = 'dtm')


R.wBH_5 = w_BH(cal.score, test.score, cal.weight, test.weight, q=0.5)
R.wCC.hete_5 = w_CS(cal.score, test.score, cal.weight, test.weight, q=0.5, rand = 'hete')
R.wCC.homo_5 = w_CS(cal.score, test.score, cal.weight, test.weight, q=0.5, rand = 'homo')
R.wCC.dtm_5 = w_CS(cal.score, test.score, cal.weight, test.weight, q=0.5, rand = 'dtm')


real.plot = test.all %>% ggplot(aes(x = X2, y = pval)) +
  geom_point(size=0.8, alpha = 0.2) + theme_bw() +
  geom_point(aes(x = X2, y = pval), color = 'red', size = 0.8, alpha = 0.2,
             data = test.all[R.wBH_5,]) +
  geom_point(aes(x = X2, y = pval), color = 'red', size = 0.8, alpha = 0.3,
             data = test.all[R.wBH_2,]) +
  geom_point(aes(x = X2, y = pval), color = 'red', size = 0.8, alpha = 0.4,
             data = test.all[R.wBH_1,]) +
  geom_hline(aes(yintercept = length(R.wBH_1)*0.1/m),
             linetype = 'dashed', color = 'red', size = 0.5, alpha=0.9) +
  geom_hline(aes(yintercept = length(R.wBH_2)*0.2/m),
             linetype = 'dashed', color = 'red', size = 0.5, alpha=0.6) +
  geom_hline(aes(yintercept = length(R.wBH_5)*0.5/m),
             linetype = 'dashed', color = 'red', size = 0.5, alpha=0.4) +
  # coord_cartesian(xlim=c(-2.5,3), clip="off") +
  xlim(c(-3.5, 2.2)) +
  annotate("text", x = -3.4, y = length(R.wBH_1)*0.1/m -0.002,
           label = "FDR level = 0.1", size=3.2, family='Times',
           col = 'red', alpha=0.9)+
  annotate("text", x = -3.4, y = length(R.wBH_2)*0.2/m -0.005,
           label = "FDR level = 0.2", size=3.2, family='Times',
           col = 'red', alpha=0.6)+
  annotate("text", x = -3.4, y = length(R.wBH_5)*0.5/m -0.04,
           label = "FDR level = 0.5", size=3.2, family='Times',
           col = 'red', alpha=0.4)+
  annotate("text", x = -3.4, y = length(R.wBH_1)*0.1/m -0.002,
           label = "FDR level = 0.1", size=3.2, family='Times',
           col = 'black', alpha=0.2)+
  annotate("text", x = -3.4, y = length(R.wBH_2)*0.2/m -0.005,
           label = "FDR level = 0.2", size=3.2, family='Times',
           col = 'black', alpha=0.2)+
  annotate("text", x = -3.4, y = length(R.wBH_5)*0.5/m -0.04,
           label = "FDR level = 0.5", size=3.2, family='Times',
           col = 'black', alpha=0.2)+
  scale_y_continuous(trans=reverselog_trans(10))  +
  theme(text= element_text(family="Times", size=12),
        axis.text= element_text(family="Times", size=10),
        strip.text.x= element_text(family="Times", size=10),
        strip.text.y= element_text(family="Times", size=10),
        legend.title = element_text(family="Times", size=12),
        axis.ticks.x = element_blank(),
        axis.text.x=element_blank(),
        legend.position="bottom") +
  xlab("School achievement level") + ylab("Conforml p-value")

ggsave("ite_real_pval.pdf", plot = real.plot, width = 9.6, height=2.4, units="in")




test.all = test.all[order(test.all$pval),]
test.all$cdf = 1:nrow(test.all)/nrow(test.all)
# survival plot
test.all$suv = 1 - test.all$cdf

cdf.plot = test.all %>% ggplot(aes(x = pval, y = cdf)) +
  geom_line(size=0.8) + theme_bw() +
  geom_line(aes(x=x, y=y), data = data.frame("x"=seq(0,1,by=0.001), "y"=seq(0,1,by=0.001)),
            col = 'red', size = 0.5, linetype='dashed') +
  theme(text= element_text(family="Times", size=12),
        axis.text= element_text(family="Times", size=10),
        strip.text.x= element_text(family="Times", size=10),
        strip.text.y= element_text(family="Times", size=10),
        legend.title = element_text(family="Times", size=12),
        # panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position="bottom") +
  xlab("Conformal p-value") + ylab("Empirical cdf")

ggsave("ite_cdf.pdf", plot = cdf.plot, width = 5, height=3, units="in")





