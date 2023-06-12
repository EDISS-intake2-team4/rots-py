# cython: language_level=3
import numpy as np
cimport cython 

#@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
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

def pvalue(double[:] observed, double[:] permuted):
  cdef Py_ssize_t a_max = observed.shape[0]
  cdef Py_ssize_t b_max = permuted.shape[0]
  pvalues = np.zeros(a_max, dtype=np.double)
  cdef double[:] pvalues_view = pvalues

  cdef Py_ssize_t i, j = 0
  for i in range(a_max):
    while j < b_max and permuted[j] >= observed[i]:
      j += 1
    pvalues_view[i] = float(j) / b_max

  return pvalues