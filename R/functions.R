#' @import stats
#' @title Weighted Conformalized Selection
#'
#' @description This function implements the Weighted Conformalized Selection algorithm. It takes evaluated scores and weights for the calibration and test data as input, and returns the index set of the selected units with a specified FDR nominal level.
#' @param cal.score Vector of scores V_i = V(X_i,Y_i) for calibration data
#' @param test.score Vector of scores hat{V}_{n+j} = V(X_{n+j}, c_{n+j}) for test data
#' @param cal.weight Vector of weights W_i = W(X_i) for cailbration data
#' @param test.weight Vector of weights W_{n+j} = W(X_{n+j}) for test data
#' @param q Nominal FDR level, default at 0.1
#' @param rand Method for random pruning, can be 'hete' for heterogeneous pruning, 'homo' for homogeneous pruning, and 'dtm' for deterministic pruning
#' @return A vector of indices in the test data that are selected as 'promising candidates'
#' @export
weighted_CS <- function(cal.score, test.score, cal.weight, test.weight, q=0.1, rand="hete"){
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



#' @import stats
#' @title Benjamini Hochberg procedure for weighed conformal p-values
#' @description This function implements the Benjamini Hochberg procedure with weighted conformal p-values. It takes evaluated scores and weights for the calibration and test data as input, and returns the index set of the selected units with a specified FDR nominal level. It computes fast, but does not necessarily controls the FDR in finite sample.
#' @param cal.score Vector of scores V_i = V(X_i,Y_i) for calibration data
#' @param test.score Vector of scores hat{V}_{n+j} = V(X_{n+j}, c_{n+j}) for test data
#' @param cal.weight Vector of weights W_i = W(X_i) for cailbration data
#' @param test.weight Vector of weights W_{n+j} = W(X_{n+j}) for test data
#' @param q Nominal FDR level, default at 0.1
#' @return A vector of indices in the test data that are selected as 'promising candidates'
#' @export
weighted_BH <- function(cal.score, test.score, cal.weight, test.weight, q=0.1){
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
