# suppressPackageStartupMessages(library(grf))
# suppressPackageStartupMessages(library(gbm))
# suppressPackageStartupMessages(library(conTree))



#####################################################################
# Train a quantile regression model hat{q}(x,beta)
#####################################################################
# covariate_name is a vector of column names of covariates
# response_name is the column name of rsponse
## if method = 'qrf', return the quantile regression forest object
## if method = 'ctree', return a list `mdl` of contrast tree:
#### - mdl$mdlrb is the modtrast object
#### - mdl$resid.train.bgm is the residual of training data from bgm model
#### - mdl$bgm.mdl is the bgm model object

qtl_train <- function(train.data, covariate_name, response_name, method='qrf', other.args=NULL){
  if (method=='qrf'){# quantile forest
    if (is.null(other.args)){
      mdl = quantile_forest(train.data[covariate_name], train.data[,response_name], num.threads=1)
    }else if (is.null(other.args$num.threads)){
      mdl = quantile_forest(train.data[covariate_name], train.data[,response_name], num.threads=1)
    }else{
      mdl = quantile_forest(train.data[covariate_name], train.data[,response_name], num.threads=other.args$num.threads)
    }
    
  }
  if (method=='ctree'){# contrast tree, input multiple models
    fmla <- with(train.data, as.formula(paste(response_name,"~",
                                              paste(covariate_name, collapse= "+"))))
    gbm_mdl <- gbm(fmla, data = train.data, distribution = "gaussian", n.trees = 1000)
    
    capture.output(qt.fit<- predict(object=gbm_mdl, newdata = train.data),
                   file=NULL)
    resid.gbm <- train.data[,response_name]- qt.fit
    resamp.res <- qt.fit + resid.gbm[sample.int(dim(train.data)[1])]
    mdlrb <- modtrast(train.data[covariate_name], train.data[,response_name],
                      resamp.res, min.node = NULL)
    mdl = list("resid.train.bgm" = resid.gbm, "mdlrb" = mdlrb, "gbm.mdl" = gbm_mdl)
    
  }
  return(mdl)
}


#####################################################################
# Compute estimated quantile from a quantile regression model object `mdl` for hat{q}(x,beta)
# for a dataframe of covariates and a vector of target quantiles
#####################################################################
## if method = 'qrf', mdl is the quantile regression forest object
## if method = 'ctree', mdl is a list:
#### - mdl$mdlrb is the modtrast object
#### - mdl$resid.train.bgm is the residual of training data from bgm model
#### - mdl$bgm.mdl is the bgm model object
## newX is the new covariates, a dataframe
## qtls is a vector of target quantiles
## returns a matrix, each row is a datapoint, each column is a target quantile

qtl_pred <- function(mdl, newX, qtls, method='qrf'){
  if (method=='qrf'){# quantile forest
    fitted.quantiles = predict(mdl, newX, quantiles = qtls, num.threads=1)
  }
  if (method=='ctree'){# contrast tree
    qt.fit.gbm = suppressWarnings(predict(object=mdl$gbm.mdl, newdata = newX))
    # 
    # fitted.quantiles = matrix(0, nrow=dim(newdata)[1], ncol=length(qtls))
    # for (i in 1:dim(newdata)[1]){
    #   pred.i = qtl_pred_ctree(mdl, newdata[i,], qt.fit.gbm[i], qtls)
    #   fitted.quantiles[i,] = pred.i
    # }
    ## vectorize
    if (is.null(dim(newX)[1])){
      fitted.quantiles = qtl_pred_ctree(mdl, newX, qt.fit.gbm, qtls)
    }else if (dim(newX)[1]==1){
      fitted.quantiles = qtl_pred_ctree(mdl, newX, qt.fit.gbm, qtls)
    }else{
      Dlist = split(newX, seq(nrow(newX)))
      qt.fit.list = as.list(qt.fit.gbm)
      fitted.quantiles = mapply(qtl_pred_ctree, newx=Dlist, qt.fit.gbm=qt.fit.list,
                                MoreArgs=list(mdl=mdl, qtls=qtls))
      fitted.quantiles = t(fitted.quantiles)
    }
  }
  return(fitted.quantiles)
}

#####################################################################
# auxiliary: predict hat{q}(x,beta) for one datapoint using contrast tree
qtl_pred_ctree <- function(mdl, newx, qt.fit.gbm, qtls){
  resid.train.gbm = mdl$resid.train.bgm
  mdlrb = mdl$mdlrb
  qres = as.numeric(quantile(resid.train.gbm, qtls))
  fitted.quantile = ydist(mdlrb, newx, qt.fit.gbm + qres)
  return(fitted.quantile)
}

#####################################################################
# Compute estimated cdf from a quantile regression model object `mdl` for hat{q}(x,beta)
# def: \hat{F}(y|x) = \hat{P}( Y<= y | x) = sup{beta: hat{q}(x,beta) <= y}
# for a single datapoint
#####################################################################
## if method = 'qrf', mdl is the quantile regression forest object
## if method = 'ctree', mdl is a list:
#### - mdl$mdlrb is the modtrast object
#### - mdl$resid.train.bgm is the residual of training data from bgm model
#### - mdl$bgm.mdl is the bgm model object
## Xpiece is the covariates, for one single datapoint
## y_target is the target of cdf
## eps is the resolution of returned cdf, default = 0.001
## if method = 'ctree', gbm.fitted is the fitted value for Xpiece on mdl$gbm.mdl model

cdf_pred_piece <- function(mdl, Xpiece, y_target, method='qrf', eps=0.001, gbm.fitted=NULL){
  p_grid = seq(eps,1-eps,by=eps)
  # predict fitted quantiles on a grid of probabilities
  if (method=='qrf'){
    q_grid = qtl_pred(mdl, Xpiece, p_grid, method='qrf')
  }
  if (method=='ctree'){
    q_grid = ydist(mdl$mdlrb, Xpiece, 
                   gbm.fitted + as.numeric(quantile(mdl$resid.train.bgm, p_grid))) 
  }
  
  # find the largest beta s.t. hat{q}(x,beta) <= y.target
  if (y_target>q_grid[length(q_grid)]){
    return(1.0)
  }else if (y_target < q_grid[1]){
    return(0.0)
  }else{
    ind_est = max(which(q_grid <= y_target))
    return(p_grid[ind_est])
  }
}

#####################################################################
# Compute estimated cdf from a quantile regression model object `mdl` for hat{q}(x,beta)
# def: \hat{F}(y|x) = \hat{P}( Y<= y | x) = sup{beta: hat{q}(x,beta) <= y}
# for a dataframe of covariates and a vector of target y's
#####################################################################
## if method = 'qrf', mdl is the quantile regression forest object
## if method = 'ctree', mdl is a list:
#### - mdl$mdlrb is the modtrast object
#### - mdl$resid.train.bgm is the residual of training data from bgm model
#### - mdl$bgm.mdl is the bgm model object
## newX is the new covariates, a dataframe
## newY is the targets, a vector
## eps is the resolution of returned cdf, default = 0.001

cdf_pred <- function(mdl, newX, newY, method='qrf', eps=0.001){
  if (method=='qrf'){
    Dlist = split(newX, seq(nrow(newX)))
    Ylist = as.list(newY)
    res = mapply(cdf_pred_piece, Xpiece = Dlist, y_target=Ylist,
                 MoreArgs=list(mdl=mdl, method='qrf', eps=eps))
  }
  if (method=='ctree'){
    gbm.fitted = suppressWarnings(predict(object=mdl$gbm.mdl, newdata = newX, n.trees=1000))
    gbm.fitted.list = as.list(gbm.fitted)
    Ylist = as.list(newY)
    Dlist = split(newX, seq(nrow(newX)))
    res = mapply(cdf_pred_piece, Xpiece = Dlist, y_target=Ylist, gbm.fitted=gbm.fitted.list,
                 MoreArgs=list(mdl=mdl, method='ctree', eps=eps))
  }
  return(res)
}


#####################################################################
# split the data into three folds
# p_train is the proportion of training data
# p_calib is the proportion of calibration data
Split.Data <- function(data, p_train, p_calib){
  n = dim(data)[1]
  n_train = floor(n*p_train)
  n_calib = floor(n*p_calib)
  n_test = n - n_train - n_calib
  re.ind = sample(n,n)
  new_data = data[re.ind,]
  train_data = new_data[1:n_train,]
  calib_data = new_data[(1+n_train):(n_train+n_calib),]
  test_data = new_data[(1+n_train+n_calib):n,]
  return(list("train" = train_data, "calib" = calib_data, "test" = test_data))
}


