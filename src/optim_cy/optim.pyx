# cython: language_level=3
# cython: infer_types=True
# distutils: language = c++
import numpy as np
cimport cython 
cimport numpy as np

from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.algorithm cimport sort
from libc.stdlib cimport malloc, free
from libc.math cimport fabs


# cdef extern from "stdlib.h":
#   void qsort(void* base, size_t nmemb, size_t size, int (*compar)(const void*, const void*))

cdef void print_array(double* arr, int length):
  cdef Py_ssize_t i
  for i in range(length):
    print(arr[i])

# cdef int compare_func(const void* a, const void* b):
#   cdef double element_a = (<double*>a)[0]
#   cdef double element_b = (<double*>b)[0]
#   if element_a < element_b:
#       return -1
#   elif element_a > element_b:
#       return 1
#   else:
#       return 0

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
def pvalue(double[:] observed, double[:] permuted):
  cdef Py_ssize_t a_max = observed.shape[0]
  cdef Py_ssize_t b_max = permuted.shape[0]
  #pvalues = np.zeros(a_max, dtype=np.double)
  #cdef double[:] pvalues_view = pvalues
  cdef double *pvalues = <double *>malloc(a_max * sizeof(double))

  cdef Py_ssize_t i
  cdef int j = 0
  for i in range(a_max):
    while permuted[j] >= observed[i] and j < b_max:
      j += 1
    pvalues[i] = float(j) / b_max

  # Convert the C-style array to a NumPy array
  cdef np.ndarray[np.double_t] pvalues_array = np.empty(a_max, dtype=np.double)
  for i in range(a_max):
    pvalues_array[i] = pvalues[i]

  # Free the memory allocated for pvalues
  free(pvalues)

  return pvalues_array

@cython.cdivision(True)
def calculateOverlaps1(double[:,:] D, double[:,:] S, double[:,:] pD, double[:,:] pS, int D_len, int[:] N, int N_len, double ssq_i, int B, double[:,:] overlaps, double[:,:] overlaps_P):  
  # cdef double[:] D_ovlp = flatten(D)
  # cdef double[:] S_ovlp = flatten(S)
  # cdef double[:] pD_ovlp = flatten(pD)
  # cdef double[:] pS_ovlp = flatten(pS)
  cdef vector[double] D_ovlp = flatten_to_vec(D)
  cdef vector[double] S_ovlp = flatten_to_vec(S)
  cdef vector[double] pD_ovlp = flatten_to_vec(pD)
  cdef vector[double] pS_ovlp = flatten_to_vec(pS)
  cdef vector[double] overlaps_ovlp = flatten_to_vec(overlaps)
  cdef vector[double] overlaps_P_ovlp = flatten_to_vec(overlaps_P)

  
    
  # res1 = np.zeros(D_len, dtype=np.double)
  # cdef double[:] res1_view = res1
  # res2 = np.zeros(D_len, dtype=np.double)
  # cdef double[:] res2_view = res2
  # pres1 = np.zeros(D_len, dtype=np.double)
  # cdef double[:] pres1_view = pres1
  # pres2 = np.zeros(D_len, dtype=np.double)
  # cdef double[:] pres2_view = pres2
  cdef double *res1 = <double *>malloc(D_len * sizeof(double))
  cdef double *res2 = <double *>malloc(D_len * sizeof(double))
  cdef double *pres1 = <double *>malloc(D_len * sizeof(double))
  cdef double *pres2 = <double *>malloc(D_len * sizeof(double))
   
  cdef Py_ssize_t b,i
  for b in range(1, B+1):
    #res1, res2, pres1, pres2 = fast_make_res(res1, res2, pres1, pres2, D_ovlp, S_ovlp, pD_ovlp, pS_ovlp, D_len, ssq_i, b, B)
    for i in range(D_len):      
      res1[i] = fabs( D_ovlp[(b-1) * D_len + i] / ( S_ovlp[(b-1) * D_len + i] + ssq_i) )
      res2[i] = fabs( D_ovlp[(b + B - 1) * D_len + i] / ( S_ovlp[(b + B - 1) * D_len + i] + ssq_i) )
      pres1[i] = fabs( pD_ovlp[(b-1) * D_len + i] / (pS_ovlp[(b-1) * D_len + i] + ssq_i))
      pres2[i] = fabs( pD_ovlp[(b + B - 1) * D_len + i] / (pS_ovlp[(b + B -1) * D_len + i] + ssq_i))
            
    calculateOverlap_1(res1, res2, D_len, N, N_len, b, B, overlaps_ovlp)
    calculateOverlap_1(pres1, pres2, D_len, N, N_len, b, B, overlaps_P_ovlp)
  
  free(res1)
  free(res2)
  free(pres1)
  free(pres2)

  # Reshape the vectors to 2D using NumPy
  #cdef np.ndarray[np.double_t, ndim=2] overlaps_reshaped = np.reshape(overlaps_ovlp, (B, N_len))
  #cdef np.ndarray[np.double_t, ndim=2] overlaps_P_reshaped = np.reshape(overlaps_P_ovlp, (B, N_len))

  cdef dict result = {
      "overlaps": np.reshape(overlaps_ovlp, (B, N_len)),
      "overlaps_P": np.reshape(overlaps_P_ovlp, (B, N_len)),
  }

  return result #{'overlaps': overlaps_ovlp.reshape(overlaps.shape), 'overlaps_P': overlaps_P_ovlp.reshape(overlaps_P.shape)} 

# cdef double[:] flatten(double[:,:] arr):
#   cdef Py_ssize_t x_max = arr.shape[0]
#   cdef Py_ssize_t y_max = arr.shape[1]
#   cdef double[:] res = np.zeros(x_max * y_max, dtype=np.double)
#   cdef Py_ssize_t i, j
#   cdef Py_ssize_t k = 0
#   for i in range(x_max):
#     for j in range(y_max):
#       res[k] = arr[i, j]
#       k += 1
#   return res

cdef double sum = 0
# Calculate the overlap
cdef void calculateOverlap_1(double *r1, double *r2, int r_len, int[:] N, int N_len, Py_ssize_t b, int B, vector[double]& overlaps):  
  # Copy r2 and sort the copy.
  #r3 = np.zeros(r_len, dtype=np.double)
  #cdef double[:] r3_view = r3
  cdef double *r3 = <double *>malloc(r_len * sizeof(double))

  cdef Py_ssize_t k
  for k in range(r_len):
    r3[k] = r2[k]

  # Sort the memoryview
  #sort_memview(r3_view)
  #reverse_memview(r3_view) 
  sort(r3, r3 + r_len)
  custom_reverse(r3, r_len)
  #reverse(r3, r3 + r_len)
  
  # Sort r2 by r1
  sort2_1(r1, r2, r_len) 

  #cdef double[:] overlaps_view = overlaps
  cdef Py_ssize_t i, j
  #Calculate the overlap  
  for i in range(N_len):
    global sum   
    for j in range(N[i]):
      if r2[j] >= r3[N[i] - 1]:
        sum = sum + 1
    overlaps[ (b-1) + i*B ] = sum / N[i]    
    sum = 0
  # Free memory for r3
  free(r3)



#cdef void sort_memview(double[:] mv):
  # data = [item for item in mv]  
  # data.sort()

  # cdef Py_ssize_t i
  # # Copy the sorted values back to the memoryview
  # for i in range(len(data)):
  #   mv[i] = data[i]
#  cdef double* ptr = &mv[0]
#  cdef Py_ssize_t length = mv.shape[0]

#  qsort(ptr, length, sizeof(double), compare_func)

# cdef void reverse_memview(double[:] mv):
#   cdef Py_ssize_t len = mv.shape[0]
#   cdef double* ptr = &mv[0]

#   cdef double* start = ptr
#   cdef double* end = ptr + len - 1

#   while start < end:
#     temp = start[0]
#     start[0] = end[0]
#     end[0] = temp
#     start += 1
#     end -= 1

cdef void custom_reverse(double* arr, int length):
  cdef int start = 0
  cdef int end = length - 1
  while start < end:
    # Swap elements
    arr[start], arr[end] = arr[end], arr[start]
    start += 1
    end -= 1

# Sort array b based on the array a (decreasingly)
cdef void sort2_1(double *a, double *b, int n):    
  #pairs = np.empty((n, 2), dtype=np.double)

  cdef vector[pair[double, double]] pairs
  pairs.reserve(n)
  
  cdef Py_ssize_t k
  for k in range(n):
    #pairs[k] = (a[k], b[k])  
    pairs.push_back( pair[double, double](a[k], b[k]) )

  # Sort the pairs (inc). By default pairs are sorted by the first value and
  # in the case of a tie, the second values are used.
  #pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
  sort(pairs.begin(), pairs.end())
  #cdef double[:, :] pairs_view = pairs

  cdef Py_ssize_t i
  # Split the pairs back into the original vectors (dec).  
  for i in range(n):
    a[n-1-i] = pairs[i].first#pairs_view[i][0]
    b[n-1-i] = pairs[i].second#pairs_view[i][1]
  
  pairs.clear()

@cython.cdivision(True)
def calculateOverlaps2(double[:,:] D, double[:,:] pD, int D_len, int[:] N, int N_len, int B, double[:,:] overlaps, double[:,:] overlaps_P):
  cdef vector[double] D_ovlp = flatten_to_vec(D)  
  cdef vector[double] pD_ovlp = flatten_to_vec(pD)
  cdef vector[double] overlaps_ovlp = flatten_to_vec(overlaps)
  #cdef double[:] overlaps_view = overlaps_ovlp
  cdef vector[double] overlaps_P_ovlp = flatten_to_vec(overlaps_P)
  #cdef double[:] overlaps_P_view = overlaps_P_ovlp
    
  # res1 = np.zeros(D_len, dtype=np.double)
  # cdef double[:] res1_view = res1
  # res2 = np.zeros(D_len, dtype=np.double)
  # cdef double[:] res2_view = res2
  # pres1 = np.zeros(D_len, dtype=np.double)
  # cdef double[:] pres1_view = pres1
  # pres2 = np.zeros(D_len, dtype=np.double)
  # cdef double[:] pres2_view = pres2
  cdef double *res1 = <double *>malloc(D_len * sizeof(double))
  cdef double *res2 = <double *>malloc(D_len * sizeof(double))
  cdef double *pres1 = <double *>malloc(D_len * sizeof(double))
  cdef double *pres2 = <double *>malloc(D_len * sizeof(double))

  cdef Py_ssize_t b,i
  for b in range(1, B):
    #res1, res2, pres1, pres2 = fast_make_res(res1, res2, pres1, pres2, D_ovlp, S_ovlp, pD_ovlp, pS_ovlp, D_len, ssq_i, b, B)
    for i in range(D_len):
      res1[i] = fabs( D_ovlp[(b-1) * D_len + i] )
      res2[i] = fabs( D_ovlp[(b + B - 1) * D_len + i] )
      pres1[i] = fabs( pD_ovlp[(b-1) * D_len + i] )
      pres2[i] = fabs( pD_ovlp[(b + B - 1) * D_len + i] )
            
    calculateOverlap_2(res1, res2, D_len, N, N_len, b, B, overlaps_ovlp)
    calculateOverlap_2(pres1, pres2, D_len, N, N_len, b, B, overlaps_P_ovlp)

  free(res1)
  free(res2)
  free(pres1)
  free(pres2)

  # Reshape the vectors to 2D using NumPy
  #cdef np.ndarray[np.double_t, ndim=2] overlaps_reshaped = np.reshape(overlaps_ovlp, (B, N_len))
  #cdef np.ndarray[np.double_t, ndim=2] overlaps_P_reshaped = np.reshape(overlaps_P_ovlp, (B, N_len))

  cdef dict result = {
        "overlaps": np.reshape(overlaps_ovlp, (B, N_len)),
        "overlaps_P": np.reshape(overlaps_P_ovlp, (B, N_len)),
    }
  return result #{'overlaps': overlaps_ovlp.reshape(overlaps.shape), 'overlaps_P': overlaps_P_ovlp.reshape(overlaps_P.shape)} 

cdef double sum_1 = 0
# Calculate the overlap
cdef void calculateOverlap_2(double *r1, double *r2, int r_len, int[:] N, int N_len, Py_ssize_t b, int B, vector[double]& overlaps):
  # Copy r2 and sort the copy.
  #r3 = np.zeros(r_len, dtype=np.double)
  #cdef double[:] r3_view = r3
  cdef double *r3 = <double *>malloc(r_len * sizeof(double))

  cdef Py_ssize_t k
  for k in range(r_len):
    r3[k] = r2[k]
  
  # Sort the memoryview
  #sort_memview(r3_view)
  #reverse_memview(r3_view) 
  sort(r3, r3 + r_len)
  custom_reverse(r3, r_len)

  # Sort r2 by r1
  sort2_2(r1, r2, r_len) 
  
  #cdef double[:] overlaps_view = overlaps
  cdef Py_ssize_t i, j
  #Calculate the overlap
  for i in range(N_len):
    global sum_1
    for j in range(N[i]):
      if r2[j] >= r3[N[i] - 1]:
        sum_1 = sum_1 + 1
    overlaps[ (b-1) + i*B ] = sum_1 / N[i]
    sum_1 = 0
  free(r3)



cdef vector[double] flatten_to_vec(double[:, :] arr):
    cdef Py_ssize_t rows = arr.shape[0]
    cdef Py_ssize_t cols = arr.shape[1]
    
    cdef vector[double] vec
    cdef Py_ssize_t i, j
    
    for i in range(rows):
        for j in range(cols):
            vec.push_back(arr[i, j])
    
    return vec

# Sort array b based on the array a (decreasingly)
cdef void sort2_2(double *a, double *b, int n):    
  #pairs = np.empty((n, 2), dtype=np.double)

  cdef vector[pair[double, double]] pairs
  pairs.reserve(n)
  
  cdef Py_ssize_t k
  for k in range(n):
    #pairs[k] = (a[k], b[k])  
    pairs.push_back( pair[double, double](a[k], b[k]) )

  # Sort the pairs (inc). By default pairs are sorted by the first value and
  # in the case of a tie, the second values are used.
  #pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
  sort(pairs.begin(), pairs.end())
  #cdef double[:, :] pairs_view = pairs

  cdef Py_ssize_t i
  # Split the pairs back into the original vectors (dec).  
  for i in range(n):
    a[n-1-i] = pairs[i].first#pairs_view[i][0]
    b[n-1-i] = pairs[i].second#pairs_view[i][1]
  
  pairs.clear()