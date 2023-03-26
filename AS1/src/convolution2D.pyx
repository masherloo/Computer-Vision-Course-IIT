import numpy as np
cimport numpy as np
def convolve2d(np.ndarray image,
               np.ndarray kernel):
    cdef int hi = image.shape[0]
    cdef int wi = image.shape[1]
    cdef int wk = kernel.shape[1]
    cdef int hk = kernel.shape[0]
    cdef int h
    cdef int w
    cdef int sum
    cdef int i
    cdef int j
    cdef int row
    cdef int col
    cdef np.ndarray out = np.zeros([hi, wi])
    if hk % 2 == 0:
        return image     
    kernel = kernel/np.sum(kernel)
    h = hk//2
    w = wk//2
    for row in range(h, hi-h):
        for col in range(w, wi-w):
            sum = 0
            for i in range(hk):
                for j in range(wk):
                    sum += image[row - h + i, col - w + j] * kernel[i, j]
            out[row, col] = sum
    return out