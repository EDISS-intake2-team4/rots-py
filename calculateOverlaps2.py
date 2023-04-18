import numpy as np
from numba import njit
import pandas as pd


def calculateOverlaps2(D, pD, D_len, N, N_len, B, overlaps, overlaps_P):
  D_ovlp = D.ravel()  
  pD_ovlp = pD.ravel()
  overlaps_ovlp = overlaps.ravel()
  overlaps_P_ovlp = overlaps_P.ravel()

  res1 = np.zeros((D_len,1))
  res2 = np.zeros((D_len,1))
  pres1 = np.zeros((D_len,1))
  pres2 = np.zeros((D_len,1))
  for b in range(1, B):
    for i in range(D_len):
      #ipdb.set_trace(context=6)   ## BREAKPOINT
      res1[i] = np.abs( D_ovlp[(b-1) * D_len + i] )
      res2[i] = np.abs( D_ovlp[(b + B - 1) * D_len + i] )
      pres1[i] = np.abs( pD_ovlp[(b-1) * D_len + i] )
      pres2[i] = np.abs( pD_ovlp[(b + B - 1) * D_len + i] )

    ##### CalculateOverlap_2: overlaps ####  overlaps_ovlp = calculateOverlap_2(res1, res2, D_len, N, N_len, b, B, overlaps)
    # Copy r2 and sort the copy.
    r3 = sorted(res2, reverse=True)    
    # Sort r2 by r1
    r1, r2 = sort2_2(res1, res2, D_len)    
    #Calculate the overlap
    for k in range(N_len):
      sum = 0
      for j in range(N[k]):
        sum += (r2[j] >= r3[N[k] - 1])
      overlaps_ovlp[ (b-1) + k*B ] = sum / N[k]
    del r3
    #########################
    
    ##### CalculateOverlap_2: overlaps_P_ovlp ####  overlaps_P_ovlp = calculateOverlap_2(pres1, pres2, D_len, N, N_len, b, B, overlaps_P)
    # Copy r2 and sort the copy.
    pr3 = sorted(pres2, reverse=True)    
    # Sort r2 by r1
    pr1, pr2 = sort2_2(pres1, pres2, D_len)    
    #Calculate the overlap
    for k in range(N_len):
      sum = 0
      for j in range(N[k]):
        sum += (pr2[j] >= pr3[N[k] - 1])
      overlaps_P_ovlp[ (b-1) + k*B ] = sum / N[k]
    del pr3    
    #########################

  return {'overlaps': overlaps_ovlp.reshape(overlaps.shape), 'overlaps_P': overlaps_P_ovlp.reshape(overlaps_P.shape)}


# Calculate the overlap

def calculateOverlap_2(r1, r2, r_len, N, N_len, b, B, overlaps):
  # Copy r2 and sort the copy.
  r3 = sorted(r2, reverse=True)
  
  # Sort r2 by r1
  r1, r2 = sort2_2(r1, r2, r_len)
  
  #Calculate the overlap
  for i in range(N_len):
    sum = 0
    for j in range(N[i]):
      sum += (r2[j] >= r3[N[i] - 1])
    overlaps[ (b-1) + i*B ] = sum / N[i]

  del r3

  return overlaps

# Sort array b based on the array a (decreasingly)

def sort2_2(a, b, n):
  pairs = [] #np.zeros([n, 2], dtype=float, order='C')
  for i in range(n):
    pairs.append((a[i], b[i]))
  
  # Sort the pairs (inc)
  pairs = sorted(pairs)

  # Split the pairs back into the original vectors (dec).
  for i in range(n):
    a[n-1-i] = pairs[i][0]
    b[n-1-i] = pairs[i][1]
  
  del pairs

  return (a, b)