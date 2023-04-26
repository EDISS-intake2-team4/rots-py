import numpy as np
from numba import njit, jit
import pandas as pd
from tqdm import tqdm

@jit(nopython=True, error_model='numpy')
def bootstrapSamples(B, labels, paired):
  samples = np.zeros((B, len(labels)))#matrix(nrow=B, ncol=length(labels))
  
  for i in range(B):
    for label in np.unique(labels):
      pos = np.where(labels == label)[0]
      #samples[i, pos] = np.random.choice(pos, size=len(pos), replace=True)            
      _samp = samples[i]
      _samp[pos] = np.random.choice(pos, size=len(pos), replace=True)
      samples[i] = _samp

  if paired:
    for i in range(B):
      for label in np.unique(labels)[1:]:
        pos = np.where(labels == label)[0]
        #samples[i, pos] = samples[i, np.where(labels == 1)[0]] + pos[0]-1
        _samp = samples[i]
        _samp[pos] = _samp[np.where(labels == 1)[0]] + pos[0]-1
        samples[i] = _samp
  
  return samples

@jit(nopython=True, error_model='numpy')
def permutatedSamples(B, cl):
  samples = np.zeros((B, len(cl)))
  for i in range(B):
    samples[i, :] = np.random.permutation(len(cl))

  return samples


def testStatistic(paired, samples):
   # Two groups
   if len(samples)==2:
     X = samples[0]
     Y = samples[1]
     
     ## Calculates the test statistic for each row.
     ## X and Y are the data matrices of the two groups.
     ## Each row of these two matrices must contain at least TWO not NA values.
     ## Thus the "variance" always exists.  
     
     #ipdb.set_trace(context=6)   ## BREAKPOINT

     ## Row means
     mX = np.nanmean(X, axis=1, keepdims=True) #rowMeans(X, na.rm=TRUE)
     mY = np.nanmean(Y, axis=1, keepdims=True) #rowMeans(Y, na.rm=TRUE)
     
     #mX[np.isnan(mX)] = 0
     #mY[np.isnan(mY)] = 0

     ## Pooled standard deviations for each row
     sX = np.nansum((X - mX)**2, axis=1) #rowSums((X - mX)^2, na.rm=TRUE)
     sY = np.nansum((Y - mY)**2, axis=1) #rowSums((Y - mY)^2, na.rm=TRUE)

     #sX[np.isnan(sX)] = 0
     #sY[np.isnan(sY)] = 0
     
     if not paired:
       ## Number of not NA values in each row
       nX = np.sum(~np.isnan(X), axis=1)
       nY = np.sum(~np.isnan(Y), axis=1)             

       ## d == difference between the group means for each row (==gene)
       ## s == pooled standard deviation for each row (==gene)        
       d = mY - mX
       s = np.sqrt(((sX + sY) / (nX + nY - 2)) * (1 / nX + 1 / nY))
       
       ## Cases with less than two non-missing values.
       ## Set d = 0, s = 1
       ind = np.where( (nY < 2) | (nX < 2) )
       d[ind] = 0
       s[ind] = 1
     
     
     if paired:
       ## Add for paired
       sXY = np.nansum((X - mX)*(Y - mY), axis=1)
       
       ## Number of not NA values in each row
       n = np.sum(~np.isnan(X*Y), axis=1)
       
       ## d == difference between the group means for each row (==gene)
       ## s == pooled standard deviation for each row (==gene)        
       d = mY - mX
       s = np.sqrt(((sX + sY) / (n + n - 2)) * (2 / n) - 2/(n*n-n)*sXY)
       
       ## Cases with less than two non-missing values.
       ## Set d = 0, s = 1
       ind = np.where( n < 2 )
       d[ind] = 0
       s[ind] = 1
     
     return  {'d': d.reshape(-1), 's': s.reshape(-1)}
   
   # Multiple groups
   if len(samples)>2:
     
     samples_all = np.concatenate(samples, axis=1) # do.call("cbind",samples)
     
     if not paired:
       sum_cols = np.sum([sample.shape[1] for sample in samples])
       prod_cols = np.prod([sample.shape[1] for sample in samples])
       f = sum_cols / prod_cols #f <- sum(sapply(samples, ncol)) / prod(sapply(samples, ncol))

       r = np.zeros(samples_all.shape[0]) #r <- vector(mode="numeric", length=nrow(samples.all))
       for k in range(len(samples)):
         r = r + (np.nanmean(samples[k], axis=1)-np.nanmean(samples_all, axis=1))**2 #r <- r + (rowMeans(samples[[k]], na.rm=TRUE)-rowMeans(samples.all, na.rm=TRUE))^2       
       d = (f*r)**0.5
       
       f = 1/np.sum([sample.shape[1] for sample in samples]-1) * np.sum(1/[sample.shape[1] for sample in samples]) #f <- 1/sum(sapply(samples, ncol)-1) * sum(1/sapply(samples, ncol))
       s = np.zeros(samples_all.shape[0])  # s <- vector(mode="numeric", length=nrow(samples.all))
       for k in range(len(samples)):
         s = s + np.sum(np.apply_along_axis(lambda x: (x - np.nanmean(x))**2, axis=1, arr=samples[k,:]), axis=0, where=~np.isnan(samples[k,:]).any(axis=0)) #s <- s + colSums(apply(samples[[k]], 1, function(x) (x-mean(x,na.rm=TRUE))^2), na.rm=TRUE)       
       s = (f*s)**0.5       
     
     if paired:
       raise ValueError("Multiple paired groups not supported!")     
     
     return {'d': d.reshape(-1), 's': s.reshape(-1)}

def calculateP(observed, permuted):  
  # Store order for later use
  observed_order = sorted(range(len(observed)), key=lambda k: abs(observed[k]), reverse=True) # order(abs(observed), decreasing=TRUE)
  
  # Sort observed and permuted values
  observed = -np.sort(-abs(observed)) #sort(abs(observed), decreasing=TRUE)
  permuted = -np.sort(-np.abs(permuted.flatten())) #sort(abs(as.vector(permuted)), decreasing=TRUE)
  
  # Get p-values from C++ code
  # (expects ordered vectors)
  p = pvalue(observed, permuted)        
  
  # Revert to original ordering
  results = np.zeros(len(p)) #vector(mode="numeric", length=length(p))
  for i in range(len(results)):
    results[observed_order[i]] =  p[i]
  
  return results


def calculateFDR(observed, permuted, progress):
  observed = abs(observed)
  permuted = abs(permuted)
  ord = np.argsort(-observed, kind='mergesort') #order(observed, decreasing=TRUE, na.last=TRUE)
  a = observed[ord]
  
  A = np.empty((len(a), permuted.shape[1])) #matrix(NA, nrow=length(a), ncol=ncol(permuted))
  A[:] = np.nan
  if progress:
    pb = tqdm(total=A.shape[1]) #txtProgressBar(min=0, max=ncol(A), style=3)
  for i in range(A.shape[1]): #seq_len(ncol(A))
    sorted_column = np.sort(permuted[:,i])[::-1] #sort(permuted[,i], decreasing=TRUE, na.last=TRUE)
    a_rand = np.concatenate([sorted_column[~np.isnan(sorted_column)], sorted_column[np.isnan(sorted_column)]])
    n_bigger = biggerN(a, a_rand)
    A[ord,i] = n_bigger/range(1, len(a)+1)
    if progress:
      pb.update()      

  if progress:
    pb.close()
  
  
  FDR = np.apply_along_axis(np.median, 1, A) #apply(A, 1, median)
  FDR[FDR>1] = 1
  FDR[ord] = list(reversed([min(FDR[ord][x-1:]) for x in range(len(FDR), 0, -1)])) #rev(sapply(length(FDR):1, function(x) return(min(FDR[ord][x:length(FDR)]))))

  return FDR


def biggerN(x, y):  
  sorted_x = np.sort(x)[::-1]
  x = np.concatenate([sorted_x[~np.isnan(sorted_x)], sorted_x[np.isnan(sorted_x)]]) #sort(x, decreasing=TRUE, na.last=TRUE)		# sort x in decreasing order
  sorted_y = np.sort(y)[::-1]
  y = np.concatenate([sorted_y[~np.isnan(sorted_y)], sorted_y[np.isnan(sorted_y)]]) #sort(y, decreasing=TRUE, na.last=TRUE)		# sort y in decreasing order
  #a <- match(x, x)				 # vector of the positions of (first) matches of the first argument in the second
  a = np.array([np.where(x == v)[0][0] if v in x else None for v in x])
  #b <- x %in% y				   # a logical vector indicating if there is a match or not for its left operand
  b = np.isin(x, y)
  sorted_z = np.sort(np.concatenate([x, y]))[::-1]
  z = np.concatenate([sorted_z[~np.isnan(sorted_z)], sorted_z[np.isnan(sorted_z)]]) #sort(c(x, y), decreasing=TRUE, na.last=TRUE)		# sort c(x,y) in decreasing order
  #d <- match(x, z)				 # vector of the positions of (first) matches of the first argument in the second
  # z_indices = np.argsort(z)
  # match_indices = np.searchsorted(z[z_indices], x, side='left')
  # matches = np.full(x.shape, np.nan)
  # matches[np.argsort(z_indices)] = z_indices[match_indices]
  # d = matches
  #ipdb.set_trace(context=10)   ## BREAKPOINT
  d = np.array([np.where(z == v)[0][0] if v in z else None for v in x])
  res = d - a + b

  return res

@jit(nopython=True, error_model='numpy')
def pvalue(a, b):
  observed = a.ravel()
  permuted = b.ravel()
  pvalues = np.zeros(len(observed))

  j = 0
  for i in range(len(observed)):
    while permuted[j] >= observed[i] and j < len(permuted):
      j += 1
    pvalues[i] = float(j) / len(permuted)

  return pvalues.reshape(a.shape)