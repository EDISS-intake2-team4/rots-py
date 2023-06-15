# cython: language_level=3
# cython: infer_types=True
# distutils: language = c++
import numpy as np
cimport cython 

cdef extern from "stdlib.h":
  void qsort(void* base, size_t nmemb, size_t size, int (*compar)(const void*, const void*))

cdef int compare_func(const void* a, const void* b):
  cdef double element_a = (<double*>a)[0]
  cdef double element_b = (<double*>b)[0]
  if element_a < element_b:
      return -1
  elif element_a > element_b:
      return 1
  else:
      return 0

# cpdef double[:] flatten_array(arr):
#   if len(arr.shape) != 2:
#     return arr
#   cdef Py_ssize_t x_max = arr.shape[0]
#   cdef Py_ssize_t y_max = arr.shape[1]
#   cdef double[:] res = np.zeros(x_max * y_max, dtype=np.double)
#   cdef Py_ssize_t i, j
#   cdef int k = 0
#   for i in range(x_max):
#     for j in range(y_max):
#       res[k] = arr[i, j]
#       k += 1
#   return res

#@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

def pvalue(double[:] observed, double[:] permuted):
  cdef Py_ssize_t a_max = observed.shape[0]
  cdef Py_ssize_t b_max = permuted.shape[0]
  pvalues = np.zeros(a_max, dtype=np.double)
  cdef double[:] pvalues_view = pvalues

  cdef Py_ssize_t i
  cdef int j = 0
  for i in range(a_max):
    while j < b_max and permuted[j] >= observed[i]:
      j += 1
    pvalues_view[i] = float(j) / b_max

  return pvalues

def calculateOverlaps1(D, S, pD, pS, int D_len, int[:] N, int N_len, double ssq_i, int B, overlaps, overlaps_P):  
  cdef double[:] D_ovlp = D.ravel()
  cdef double[:] S_ovlp = S.ravel()
  cdef double[:] pD_ovlp = pD.ravel()
  cdef double[:] pS_ovlp = pS.ravel()
  overlaps_ovlp = overlaps.ravel()
  overlaps_P_ovlp = overlaps_P.ravel()
    
  res1 = np.zeros(D_len, dtype=np.double)
  cdef double[:] res1_view = res1
  res2 = np.zeros(D_len, dtype=np.double)
  cdef double[:] res2_view = res2
  pres1 = np.zeros(D_len, dtype=np.double)
  cdef double[:] pres1_view = pres1
  pres2 = np.zeros(D_len, dtype=np.double)
  cdef double[:] pres2_view = pres2

  cdef Py_ssize_t b,i
  for b in range(1, B):
    #res1, res2, pres1, pres2 = fast_make_res(res1, res2, pres1, pres2, D_ovlp, S_ovlp, pD_ovlp, pS_ovlp, D_len, ssq_i, b, B)
    for i in range(D_len):
      res1_view[i] = np.abs( D_ovlp[(b-1) * D_len + i] / ( S_ovlp[(b-1) * D_len + i] + ssq_i) )
      res2_view[i] = np.abs( D_ovlp[(b + B - 1) * D_len + i] / ( S_ovlp[(b + B - 1) * D_len + i] + ssq_i) )
      pres1_view[i] = np.abs( pD_ovlp[(b-1) * D_len + i] / (pS_ovlp[(b-1) * D_len + i] + ssq_i))
      pres2_view[i] = np.abs( pD_ovlp[(b + B - 1) * D_len + i] / (pS_ovlp[(b + B -1) * D_len + i] + ssq_i))
            
    calculateOverlap_1(res1_view, res2_view, D_len, N, N_len, b, B, overlaps_ovlp)
    calculateOverlap_1(pres1_view, pres2_view, D_len, N, N_len, b, B, overlaps_P_ovlp)

  return {'overlaps': overlaps_ovlp.reshape(overlaps.shape), 'overlaps_P': overlaps_P_ovlp.reshape(overlaps_P.shape)} 

cdef int global_sum = 0

# Calculate the overlap
cpdef void calculateOverlap_1(double[:] r1, double[:] r2, int r_len, int[:] N, int N_len, Py_ssize_t b, int B, overlaps):
  # Copy r2 and sort the copy.
  r3 = np.zeros(r_len, dtype=np.double)
  cdef double[:] r3_view = r3
  cdef Py_ssize_t k
  for k in range(r_len):
    r3_view[k] = r2[k]
  
  #cdef double* r3_ptr = &r3_view[0]
  #cdef int r3_length = r3_view.shape[0]

  # Sort the memoryview
  sort_memview(r3_view)
  reverse_memview(r3_view) 
  # Sort r2 by r1
  sort2_1(r1, r2, r_len) 
  
  cdef double[:] overlaps_view = overlaps
  cdef Py_ssize_t i, j
  #Calculate the overlap
  #overlaps = fast_ovlp(r2, typed_r3, N_len, N, B, b, overlaps)
  for i in range(N_len):
    global global_sum    
    for j in range(N[i]):
      global_sum += (r2[j] <= r3_view[N[i] - 1])
    overlaps_view[ (b-1) + i*B ] = global_sum / N[i]
    global_sum = 0

cdef void sort_memview(double[:] mv):
  # data = [item for item in mv]  
  # data.sort()

  # cdef Py_ssize_t i
  # # Copy the sorted values back to the memoryview
  # for i in range(len(data)):
  #   mv[i] = data[i]
  cdef double* ptr = &mv[0]
  cdef Py_ssize_t length = mv.shape[0]

  qsort(ptr, length, sizeof(double), compare_func)

cdef void reverse_memview(double[:] mv):
  cdef Py_ssize_t len = mv.shape[0]
  cdef double* ptr = &mv[0]

  cdef double* start = ptr
  cdef double* end = ptr + len - 1

  while start < end:
    temp = start[0]
    start[0] = end[0]
    end[0] = temp
    start += 1
    end -= 1

# Sort array b based on the array a (decreasingly)
cdef void sort2_1(double[:] a, double[:] b, int n):
  #pairs = make_pairs(a, b, n)
  global pairs
  pairs = np.empty((n, 2), dtype=np.double)
  
  cdef Py_ssize_t k
  for k in range(n):
    pairs[k] = (a[k], b[k])  

  # Sort the pairs (inc). By default pairs are sorted by the first value and
  # in the case of a tie, the second values are used.
  # Sort the memoryview using std::sort
  #qsort(pairs_ptr, pairs_length, sizeof(double) * 2, compare_func)
  pairs = pairs[np.lexsort((pairs[:, 0], pairs[:, 1]))]
  cdef double[:, :] pairs_view = pairs

  cdef Py_ssize_t i
  # Split the pairs back into the original vectors (dec).
  #a, b = split_pairs(pairs, n)
  for i in range(n):
    a[n-1-i] = pairs_view[i][0]
    b[n-1-i] = pairs_view[i][1]


def calculateOverlaps2(D, pD, int D_len, int[:] N, int N_len, int B, overlaps, overlaps_P):
  cdef double[:] D_ovlp = D.ravel()  
  cdef double[:] pD_ovlp = pD.ravel()  
  overlaps_ovlp = overlaps.ravel()
  overlaps_P_ovlp = overlaps_P.ravel()
    
  res1 = np.zeros(D_len, dtype=np.double)
  cdef double[:] res1_view = res1
  res2 = np.zeros(D_len, dtype=np.double)
  cdef double[:] res2_view = res2
  pres1 = np.zeros(D_len, dtype=np.double)
  cdef double[:] pres1_view = pres1
  pres2 = np.zeros(D_len, dtype=np.double)
  cdef double[:] pres2_view = pres2

  cdef Py_ssize_t b,i
  for b in range(1, B):
    #res1, res2, pres1, pres2 = fast_make_res(res1, res2, pres1, pres2, D_ovlp, S_ovlp, pD_ovlp, pS_ovlp, D_len, ssq_i, b, B)
    for i in range(D_len):
      res1_view[i] = np.abs( D_ovlp[(b-1) * D_len + i] )
      res2_view[i] = np.abs( D_ovlp[(b + B - 1) * D_len + i] )
      pres1_view[i] = np.abs( pD_ovlp[(b-1) * D_len + i] )
      pres2_view[i] = np.abs( pD_ovlp[(b + B - 1) * D_len + i] )
            
    calculateOverlap_2(res1_view, res2_view, D_len, N, N_len, b, B, overlaps_ovlp)
    calculateOverlap_2(pres1_view, pres2_view, D_len, N, N_len, b, B, overlaps_P_ovlp)

  return {'overlaps': overlaps_ovlp.reshape(overlaps.shape), 'overlaps_P': overlaps_P_ovlp.reshape(overlaps_P.shape)} 

# Calculate the overlap
cpdef void calculateOverlap_2(double[:] r1, double[:] r2, int r_len, int[:] N, int N_len, Py_ssize_t b, int B, overlaps):
   # Copy r2 and sort the copy.
  r3 = np.zeros(r_len, dtype=np.double)
  cdef double[:] r3_view = r3
  cdef Py_ssize_t k
  for k in range(r_len):
    r3_view[k] = r2[k]
  
  #cdef double* r3_ptr = &r3_view[0]
  #cdef int r3_length = r3_view.shape[0]

  # Sort the memoryview
  sort_memview(r3_view)
  reverse_memview(r3_view) 
  # Sort r2 by r1
  sort2_1(r1, r2, r_len) 
  
  cdef double[:] overlaps_view = overlaps
  cdef Py_ssize_t i, j
  #Calculate the overlap
  #overlaps = fast_ovlp(r2, typed_r3, N_len, N, B, b, overlaps)
  for i in range(N_len):
    global global_sum    
    for j in range(N[i]):
      global_sum += (r2[j] >= r3_view[N[i] - 1])
    overlaps_view[ (b-1) + i*B ] = global_sum / N[i]
    global_sum = 0