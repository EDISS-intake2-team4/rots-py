import numpy as np
import pandas as pd
from numba import njit, jit, prange
from numba.typed import List


def calculateOverlaps1(D, S, pD, pS, D_len, N, N_len, ssq_i, B, overlaps, overlaps_P):
  D_ovlp = D.ravel()
  S_ovlp = S.ravel()
  pD_ovlp = pD.ravel()
  pS_ovlp = pS.ravel()
  overlaps_ovlp = overlaps.ravel()
  overlaps_P_ovlp = overlaps_P.ravel()
    
  res1 = np.zeros(D_len)
  res2 = np.zeros(D_len)
  pres1 = np.zeros(D_len)
  pres2 = np.zeros(D_len)
  for b in range(1, B):
    res1, res2, pres1, pres2 = fast_make_res(res1, res2, pres1, pres2, D_ovlp, S_ovlp, pD_ovlp, pS_ovlp, D_len, ssq_i, b, B)
            
    overlaps_ovlp = calculateOverlap_1(res1, res2, D_len, N, N_len, b, B, overlaps_ovlp)        
    overlaps_P_ovlp = calculateOverlap_1(pres1, pres2, D_len, N, N_len, b, B, overlaps_P_ovlp)    

  return {'overlaps': overlaps_ovlp.reshape(overlaps.shape), 'overlaps_P': overlaps_P_ovlp.reshape(overlaps_P.shape)} 

@jit(nopython=True, error_model='numpy')
def fast_make_res(res1, res2, pres1, pres2, D_ovlp, S_ovlp, pD_ovlp, pS_ovlp, D_len, ssq_i, b, B):
  for i in range(D_len):
    res1[i] = np.abs( D_ovlp[(b-1) * D_len + i] / ( S_ovlp[(b-1) * D_len + i] + ssq_i) )
    res2[i] = np.abs( D_ovlp[(b + B - 1) * D_len + i] / ( S_ovlp[(b + B - 1) * D_len + i] + ssq_i) )
    pres1[i] = np.abs( pD_ovlp[(b-1) * D_len + i] / (pS_ovlp[(b-1) * D_len + i] + ssq_i))
    pres2[i] = np.abs( pD_ovlp[(b + B - 1) * D_len + i] / (pS_ovlp[(b + B -1) * D_len + i] + ssq_i))
  return res1, res2, pres1, pres2  

# Calculate the overlap
def calculateOverlap_1(r1, r2, r_len, N, N_len, b, B, overlaps):
  # Copy r2 and sort the copy.
  r3 = sorted(r2, reverse=True)  
  
  # Sort r2 by r1
  r1, r2 = sort2_1(r1, r2, r_len)
  
  typed_r3 = List()
  [typed_r3.append(x) for x in r3]
  #Calculate the overlap
  overlaps = fast_ovlp(r2, typed_r3, N_len, N, B, b, overlaps)

  return overlaps

@jit(nopython=True, error_model='numpy')
def fast_ovlp(r2, r3, N_len, N, B, b, overlaps):
  #Calculate the overlap
  for i in range(N_len):
    sum = 0
    for j in range(N[i]):
      sum += (r2[j] <= r3[N[i] - 1])
    overlaps[ (b-1) + i*B ] = sum / N[i]
  return overlaps

# Sort array b based on the array a (decreasingly)
def sort2_1(a, b, n):
  pairs = make_pairs(a, b, n)

  # Sort the pairs (inc). By default pairs are sorted by the first value and
  # in the case of a tie, the second values are used.
  pairs = pairs[np.lexsort((pairs[:, 0], pairs[:, 1]))]

  # Split the pairs back into the original vectors (dec).
  a, b = split_pairs(pairs, n)
  return (a, b)

@jit(nopython=True, parallel=True, error_model='numpy')
def make_pairs(a, b, n):
  pairs = np.empty((n, 2))
  for i in prange(n):
    pairs[i] = [a[i], b[i]]
  return pairs

@jit(nopython=True, parallel=True, error_model='numpy')
def split_pairs(pairs, n):
  # Split the pairs back into the original vectors (dec).
  a = np.empty(n)
  b = np.empty(n)
  for i in prange(n):
    a[n-1-i] = pairs[i][0]
    b[n-1-i] = pairs[i][1]
  return (a, b)