# suppressPackageStartupMessages(library(grf))
# suppressPackageStartupMessages(library(gbm))
# suppressPackageStartupMessages(library(conTree))

gen.data <- function(n, p, rho=0, setting=1, obs=1, sig=0.1){
  if (setting %in% c(1,2,3)){#
    Sigma <- toeplitz(c(1, rep(rho,p-1)))
    cSigma <- chol(Sigma)
    X <- pnorm(matrix(rnorm(n * p),n) %*% cSigma)

    noise = rnorm(n)
    Y1 = 4/( (1+exp(-12*(X[,1]-0.5)))*(1+exp(-12*(X[,2]-0.5))) )
    if(setting == 1){# no coupling
      Y0 = rnorm(n) * 0.5 * sig
      Y1 = pmax(0, Y1 + noise * (0.2 -log(X[,1])) * sig)
    }
    if (setting == 2){# positive coupling
      Y0 = noise * 0.5 * sig
      Y1 = pmax(0, Y1 + noise * (0.2 -log(X[,1])) * sig)
    }
    if (setting == 3){# negative coupling
      Y0 = - noise * 0.5 * sig
      Y1 = pmax(0, Y1 + noise * (0.2 -log(X[,1])) * sig)
    }
    e.x = (1 + pbeta(X[,1], shape1 = 2, shape2=4)) / 4
    TT = rbinom(n, 1, e.x)
    Y = Y0 * (1-TT) + Y1 * TT
  }

  if (setting %in% c(4,5,6)){ # mixture model
    Sigma <- toeplitz(c(1, rep(rho,p-1)))
    cSigma <- chol(Sigma)
    X <- pnorm(matrix(rnorm(n * p),n) %*% cSigma)

    # 0.2 proportion of ineffective people
    ineff = rbinom(n, 1, 0.1)
    Y1.ineff = rnorm(n) * 0.5 * sig - 0.5
    Y0.ineff = Y1.ineff + 0.05
    noise.eff = rnorm(n)
    mu1.x.eff = 4/((1+exp(-12*(X[,1]-0.5)))*(1+exp(-12*(X[,2]-0.5))))

    if(setting == 4){# no coupling
      Y0.eff = rnorm(n) * 0.5 * sig
      Y1.eff = pmax(0, mu1.x.eff + noise.eff * (0.2 -log(X[,1])) * sig)
    }
    if (setting == 5){# positive coupling
      Y0.eff = noise.eff * 0.5 * sig
      Y1.eff = pmax(0, mu1.x.eff + noise.eff * (0.2 -log(X[,1])) * sig)
    }
    if (setting == 6){# negative coupling
      Y0.eff = - noise.eff * 0.5 * sig
      Y1.eff = pmax(0, mu1.x.eff + noise.eff * (0.2 -log(X[,1])) * sig)
    }
    Y0 = Y0.eff * (1-ineff) + Y0.ineff * ineff
    Y1 = Y1.eff * (1-ineff) + Y1.ineff * ineff
    e.x = (1 + pbeta(X[,1], shape1 = 2, shape2=4)) / 4
    TT = rbinom(n, 1, e.x)
    Y = Y0 * (1-TT) + Y1 * TT
  }


  if (setting %in% c(7,8,9)){ # more trends
    Sigma <- toeplitz(c(1, rep(rho,p-1)))
    cSigma <- chol(Sigma)
    X <- pnorm(matrix(rnorm(n * p),n) %*% cSigma)

    # 0.2 proportion of ineffective people
    mu0.x = 2/((1+exp(-3*(X[,1]-0.5)))*(1+exp(-3*(X[,2]-0.5))))
    mu1.x = 0.1 + 3 /((1+exp(-3*(X[,1]-0.5)))*(1+exp(-3*(X[,2]-0.5))))
    noise = rnorm(n)

    if(setting == 7){# no coupling
      Y0 = mu0.x + rnorm(n) * 0.5 * sig
      Y1 = pmax(0, mu1.x + noise * (0.2 -log(X[,1])) * sig)
    }
    if (setting == 8){# positive coupling
      Y0 = mu0.x + noise * 0.5 * sig
      Y1 = pmax(0, mu1.x + noise * (0.2 -log(X[,1])) * sig)
    }
    if (setting == 9){# negative coupling
      Y0 = mu0.x - noise * 0.5 * sig
      Y1 = pmax(0, mu1.x + noise * (0.2 -log(X[,1])) * sig)
    }
    e.x = (1 + pbeta(X[,1], shape1 = 2, shape2=4)) / 4
    TT = rbinom(n, 1, e.x)
    Y = Y0 * (1-TT) + Y1 * TT
  }



  ## generate sufficiently many observational data in one group
  flag = FALSE
  if (obs >= 0){flag = TRUE}
  while (flag){
    s.dat = gen.data(n, p, rho, setting, obs=-1, sig)
    X = rbind(X, s.dat$X)
    Y = c(Y, s.dat$Y)
    TT = c(TT, s.dat$T)
    Y0 = c(Y0, s.dat$Y0)
    Y1 = c(Y1, s.dat$Y1)
    e.x = c(e.x, s.dat$ex)
    if (obs==1){flag = (sum(TT)<n)}
    if (obs==0){flag = (sum(1-TT)<n)}
    if (obs==2){flag = (sum(1-TT)<n)|(sum(TT)<n)}
  }

  return(list("X" = X, "Y" = Y, "T" =TT, "Y0" = Y0, "Y1" = Y1, "ex" = e.x))
}



w_BH <- function(cal.score, test.score, cal.weight, test.weight, q=0.1){
  pvals = rep(0, length(test.score))
  Us = runif(length(test.score))
  csum = sum(cal.weight)
  for (i in 1:length(test.score)){
    pvals[i] = (sum(cal.weight[cal.score<test.score[i]]) + (test.weight[i] + sum(cal.weight[cal.score==test.score[i]])) * Us[i]) / (test.weight[i] + csum)
  }
  t.all = data.frame("id" = 1:length(test.score), "pval" = pvals, "score" = test.score)
  t.all = t.all[order(t.all$pval),]
  t.all$threshold = q * (1:length(test.score)) / length(test.score)
  pass.id = which(t.all$pval <= t.all$threshold)
  Rej = c()
  if (length(pass.id)>0){
    Rej = t.all$id[1:max(pass.id)]
  }
  return(Rej)
}

w_CS <- function(cal.score, test.score, cal.weight, test.weight, q=0.1, rand="hete"){
  # if (hetero){xi = runif(length(test.score))}else{xi = runif(1)}
  if (rand == 'hete'){
    xi = runif(length(test.score))
  }
  if (rand == 'homo'){
    xi = runif(1)
  }
  if (rand == 'dtm'){
    xi = 1
  }
  csum = sum(cal.weight)
  # matrix of p-values, M_{ij} = hat{p}_i^j, each column is the p-values at X_n+j
  pvals = matrix(0, length(test.score), length(test.score))
  nt = length(test.score)

  for (i in 1:nt){
    pvals[i,] = sum(cal.weight[cal.score<test.score[i]])
  }
  for (j in 1:nt){
    sml.id = (test.score[j] < test.score)
    pvals[sml.id, j] = pvals[sml.id, j] + test.weight[j]
    pvals[j,j] = pvals[j,j] + test.weight[j]
    pvals[,j] = pvals[,j] / (csum + test.weight[j])
  }
  # first step rejection
  Rs = rep(0, nt)
  Sel = rep(0, nt)
  for (j in 1:nt){
    pval.j = pvals[,j]
    pval.ord = order(pval.j)
    R.j = c()
    if (length(which(pval.j[pval.ord] < (q * (1:nt) / nt)))>0){
      R.j = pval.ord[1:max(which(pval.j[pval.ord] < (q * (1:nt) / nt)))]
    }
    Rs[j] = length(R.j)
    Sel[j] = (j %in% R.j)
  }
  # second step pruning
  xiR = xi * Rs
  xiR[!Sel] = 1+nt
  x.ord = order(xiR)
  id.pass = which(xiR[x.ord] < 1:nt)
  Rej = c()
  if (length(id.pass)>0){
    Rej = x.ord[1:max(which(xiR[x.ord] < 1:nt))]
  }


  return(Rej)
}


eval.FDR <- function(rej.id, truths){
  nrej = length(rej.id)
  if (nrej>0){
    fdp = sum(truths[rej.id]) / length(rej.id)
    power = sum(!truths[rej.id]) / sum(!truths)
  }else{
    fdp = 0
    power = 0
  }
  return(data.frame("nrej" = nrej, "fdp" = fdp, "power" = power))
}





