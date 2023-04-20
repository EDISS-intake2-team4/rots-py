#import ipdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

from helpers import bootstrapSamples, permutatedSamples, testStatistic, calculateP, calculateFDR
from calculateOverlaps1 import calculateOverlaps1
from calculateOverlaps2 import calculateOverlaps2

def rots(data, groups, B=1000, K=None, paired=False, seed=None, a1=None, a2=None, log=None, progress=False, verbose=True):
  if isinstance(data, pd.DataFrame):
    data_val = data.values    
  if seed is not None:
    random.seed(seed)
  if data.shape[1] != len(groups):
    raise ValueError("Number of samples in the data does not match the groups.")
  if not data.index.any():
    data.index = np.arange(1, data.shape[0]+1)

  ssq = np.concatenate((
    np.arange(0, 0.21, 0.01), 
    np.arange(0.22, 1.01, 0.02),
    np.arange(1.2, 5.2, 0.2)
  ))
  N = np.concatenate([
    np.arange(1, 21) * 5,
    np.arange(11, 51) * 10,
    np.arange(21, 41) * 25, 
    np.arange(11, 1001) * 100
  ])
  
  #ipdb.set_trace(context=6)   ## BREAKPOINT

  if K is None:
    K = data.shape[0] // 4
    if (verbose):
      print("No top list size K given, using ", K)

  K = min(K, data.shape[0])
  N = N[N < K]

  if isinstance(groups[0], str):
    groups = pd.factorize(groups)[0]
    groups_levels = pd.factorize(groups)[1]
  else:
    groups_levels = None

  data = data.iloc[:, np.argsort(groups)]
  groups = np.sort(groups)
  
  for i in np.unique(groups):
    if np.any(np.sum(np.isnan(data.iloc[:, np.where(groups == i)[0]]), axis=1) >= len(np.where(groups == i)[0]) - 1):
      if groups_levels is None:
        target = i
      else:
        target = groups_levels[i]
      raise ValueError("The data matrix of group " + str(target) + " contains rows with less than two non-missing values, please remove these rows.")

  cl = groups + (1-np.min(groups))

  ## Check number of samples for paired test
  if paired:
    for i in np.unique(cl)[-1]:
      if len(np.where(cl == 1)[0]) != len(np.where(cl == i)[0]):
        raise ValueError("Uneven number of samples for paired test.")

  ## Calculate fold change
  if len(np.unique(cl)) == 2:
    if log:
      logfc = np.mean(data.iloc[:, np.where(cl==1)[0]], axis=1) - np.mean(data.iloc[:, np.where(cl==2)[0]], axis=1)
    else:
      logfc = np.mean(np.log2(data.iloc[:,np.where(cl==1)[0]]+1), axis=1) - np.mean(np.log2(data.iloc[:,np.where(cl==2)[0]]+1), axis=1)
  else:
    logfc = np.repeat(np.nan, data.shape[0])

  ## ---------------------------------------------------------------------------

  ## Bootstrap samples
  if (verbose): 
    print("Bootstrapping samples")    

  samples = bootstrapSamples(2*B, cl, paired)
  ## Permutated samples
  pSamples = permutatedSamples(len(samples), cl)

  ## Test statistics in the bootstrap (permutate) datasets.
  ## For each bootstrap (permutated) dataset, determine the signal log-ratio
  ## (D) and the standard error (S).
  D = np.empty((data.shape[0], samples.shape[0]))
  S = np.empty((data.shape[0], samples.shape[0]))
  pD = np.empty((data.shape[0], samples.shape[0]))
  pS = np.empty((data.shape[0], samples.shape[0]))

  # Progress bar  
  pb = tqdm(total=samples.shape[0])

  #ipdb.set_trace(context=6)   ## BREAKPOINT
  for i in range(samples.shape[0]):
    samples_R = np.split(samples[i, :], np.where(np.diff(cl))[0]+1)
    pSamples_R = np.split(pSamples[i, :], np.where(np.diff(cl))[0]+1)
    
    ## If a1 and a2 parameters are given, we don't need the bootstrap
    ## dataset
    if a1 == None or a2 == None:
      fit = testStatistic(paired, [data.iloc[:, x].to_numpy() for x in samples_R])
      D[:,i] = fit['d']
      S[:,i] = fit['s']
    
      
    pFit = testStatistic(paired, [data.iloc[:, x].to_numpy() for x in pSamples_R])
    pD[:,i] = pFit['d']
    pS[:,i] = pFit['s']
    
    if progress: 
      pb.update()
    
  if progress: 
    pb.close()

  ## Free up memory
  ##del samples
  ##del pSamples
    
  ## ---------------------------------------------------------------------------

  if  a1==None or a2==None:
    ## Optimize the parameters
    if verbose:
      print("Optimizing parameters")
    
    ## Calculate the reproducibility values for all the given a1-values and top
    ## list sizes in both bootstrap and permuted data and their standard
    ## deviations in the bootstrap case
    
    ## Reproducibility matrix (bootstrap data):
    ## the rows correspond to the a1-values given in ssq (+ 1 for signal
    ## log-ratio only: a1=1, a2=0), the columns correspond to the different top
    ## list sizes
    reprotable = pd.DataFrame(index=np.append(ssq, "slr"), columns = N)

    ## Reproducibility matrix (permuted data):
    ## the rows correspond to the a1-values given in ssq (+ 1 for signal
    ## log-ratio only: a1=1, a2=0), the columns correspond to the different top
    ## list sizes
    reprotable_P = pd.DataFrame(index=np.append(ssq, "slr"), columns = N)

    ## Standard deviation matrix for the reproducibility values (bootstrap
    ## data): the rows correspond to the a1-values given in ssq (+ 1 for signal
    ## log-ratio only: a1=1, a2=0), the columns correspond to the different
    ## top list sizes
    reprotable_sd = pd.DataFrame(index=np.append(ssq, "slr"),columns=N)

    # Progress bar
    if progress:
        pb = tqdm(total=len(ssq)) #txtProgressBar(min=0, max=length(ssq), style=3)      

    for i in range(len(ssq)):
      ## The overlaps between bootstrap samples. Rows correspond to different
      ## bootrsrap samples and columns corrospond to different top list size.
      ## Repeat for each parameter combination a1 and a2.
      overlaps = np.zeros((B,len(N)))
      overlaps_P = np.zeros((B,len(N)))      

      ## Call the custom c++-loop 1.
      cResults = calculateOverlaps1(D, S, pD, pS, len(D), N.astype(int), len(N),
                        ssq[i], int(B), overlaps, overlaps_P)
      
      ## Colmeans & rowMeans are a lot faster than apply
      #         reprotable[i, ] <- colMeans(overlaps)
      #         reprotable.P[i, ] <- colMeans(overlaps.P)
      
      
      reprotable.iloc[i] = np.mean(cResults["overlaps"], axis=0)
      reprotable_P.iloc[i] = np.mean(cResults["overlaps_P"], axis=0)
      
      ## Standard deviation for each column
      ## same as reprotable.sd[i, ] <- apply(overlaps, 2, sd)
      ## or just sd(overlaps), but a lot faster.
      #         reprotable.sd[i,] <- sqrt(rowSums((t(overlaps) - reprotable[i,])^2) /
      #                                     (nrow(overlaps) - 1))
      #sqrt(rowSums((t(cResults[["overlaps"]]) - reprotable[i,])^2) / (nrow(cResults[["overlaps"]]) - 1)) 
      #reprotable_sd.iloc[i] = np.std(cResults["overlaps"])
      reprotable_sd.iloc[i] = np.sqrt(np.sum((cResults["overlaps"].T - reprotable.iloc[i].values[0])**2, axis=1) / 
                                      (cResults["overlaps"].shape[0] - 1))[0]
      
      if progress:
        pb.update()      

    if progress: 
      pb.close()

    i = len(ssq)
    overlaps = np.zeros((B,len(N)))
    overlaps_P = np.zeros((B,len(N)))

    ## Call the custom c++-loop 2.
    cResults = calculateOverlaps2(D, pD, len(D), N.astype(int), len(N),
                    int(B), overlaps, overlaps_P)

    ## Free up memory
    #rm(D, S)
    #del D
    #del S

    #ipdb.set_trace(context=6)   ## BREAKPOINT
    reprotable.iloc[i] = np.mean(cResults["overlaps"], axis=0) #colMeans(cResults[["overlaps"]])
    reprotable_P.iloc[i] = np.mean(cResults["overlaps_P"], axis=0) #colMeans(cResults[["overlaps_P"]])
    ## Standard deviation for each column
    #sqrt(rowSums((t(cResults[["overlaps"]]) - reprotable[i,])^2) / (nrow(cResults[["overlaps"]]) - 1))
    #reprotable_sd.iloc[i] = np.std(cResults["overlaps"])
    reprotable_sd.iloc[i] = np.sqrt(np.sum((cResults["overlaps"].T - reprotable.iloc[i].values[0])**2, axis=1) / 
                                      (cResults["overlaps"].shape[0] - 1))[0]

    ## Free up memory
    ##del overlaps
    #del overlaps_P
    #del cResults

    ## -------------------------------------------------------------------------

    ## Calculate the Z-statistic and find the top list size and the
    ## (a1,a2)-combination giving the maximal Z-value
    ztable = (reprotable - reprotable_P) / reprotable_sd
    ## Rownames of ztable are c(ssq, "slr") and colnames are N

    ztable = ztable.infer_objects()

    ## Free up memory
    #del reprotable_P
    #del reprotable_sd
    
    sel = np.unravel_index(np.argmax(ztable[np.isfinite(ztable)]), ztable.shape) #np.where(ztable == max(ztable[is.finite(ztable)]), arr.ind=TRUE)
    ## Sel is a matrix containing the location(s) of the largest value (row,
    ## col). If the location of the largest value is not unique then nrow(sel)
    ## > 2 (length(sel) > 2)

    #ipdb.set_trace(context=6)   ## BREAKPOINT
    if len(sel) > 2:
      sel = sel[0:2]

    if sel[0] < len(reprotable)-1:
      a1 = float(reprotable.index[sel[0]])
      a2 = 1
    
    if sel[0] == len(reprotable)-1:
      a1 = 1
      a2 = 0
    
    k = int(reprotable.columns[sel[1]])
    R = reprotable.iloc[sel[0],sel[1]]
    Z = ztable.iloc[sel[0],sel[1]]
    
    ## Free up memory
    #del reprotable
    
    #ipdb.set_trace(context=6)   ## BREAKPOINT
    ## Calculate the reproducibility-optimized test statistic based on the
    ## reproducibility-maximizing a1, a2 and k values and the corresponding FDR      
    #fit = testStatistic(paired, lapply(split(1:len(cl), cl), function(x) data[:,x]))
    fit = testStatistic(paired, np.split(data.to_numpy(), np.where(np.diff(cl) != 0)[0] + 1, axis=1))
    d = fit['d'] / (a1 + a2 * fit['s'])
    pD = pD/(a1 + a2 * pS)
    
    ## Free up memory
    #del pS
    
    if verbose: 
      print("Calculating p-values")
    p = calculateP(d, pD)
    
    if verbose:
      print("Calculating FDR")
    FDR = calculateFDR(d, pD, progress)
    
    ## Free up memory
    #del pD

    ROTS_output = {
      "data": data,
      "B": B,
      "d": d,
      "logfc": logfc,
      "p": p,
      "FDR": FDR,
      "a1": a1,
      "a2": a2,
      "k": k,
      "R": R,
      "Z": Z,
      "ztable": ztable,
      "cl": cl}
      
  else:
    ## Calculate statistic based on the given parameter values
    ## and the corresponding FDR
    #fit <- testStatistic(paired, lapply(split(1:length(cl), cl), function(x) data[,x]))
    fit = testStatistic(paired, np.split(data.to_numpy(), np.where(np.diff(cl) != 0)[0] + 1, axis=1))
    d = fit['d'] / (a1 + a2 * fit['s'])
    if verbose: 
      print("Calculating p-values")
    p = calculateP(d, pD/(a1 + a2 * pS))
    if verbose: 
      print("Calculating FDR")
    FDR = calculateFDR(d, pD/(a1 + a2 * pS), progress)
    
    ROTS_output = {
      "data": data,
      "B": B,
      "d": d,
      "logfc": logfc,
      "p": p,
      "FDR": FDR,
      "a1": a1,
      "a2": a2,
      "k": None,
      "R": None,
      "Z": None,      
      "cl": cl}       
      
  ## Define the class
  return ROTS_output