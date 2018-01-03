
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

### Experimental CUDA CALL to normalized cross correntropy estimator - Not in use
def ncc_cuda(x,wx,y,wy,marray,s):
    """ x and y are 3d matrix. for every row pair of x and y, a normalized cross correntropy estimator is computed and stored in the nccr 3d matrix """

    mod = SourceModule(open('./ITLkernels.cu', 'r').read())
    ACm_func = mod.get_function('NCC')

    N = x.shape[2]
    nwin = x.shape[1]
    depth = x.shape[0]
    msize = marray.size

    xgpu = gpuarray.to_gpu(x.astype(np.float32))
    ygpu = gpuarray.to_gpu(y.astype(np.float32))
    wxgpu = gpuarray.to_gpu(wx.astype(np.float32))
    wygpu = gpuarray.to_gpu(wy.astype(np.float32))
    marraygpu = gpuarray.to_gpu(marray.astype(np.int32))

    nccrgpu = gpuarray.to_gpu(np.zeros((depth, nwin, msize)).astype(np.float32))

    ACm_func(nccrgpu.gpudata, xgpu.gpudata, ygpu.gpudata, wxgpu.gpudata, wygpu.gpudata, marraygpu.gpudata,
             np.float32(s), np.uint32(msize), np.uint32(N), np.uint32(nwin), np.uint32(depth), block=(4,4,4), grid=(msize//1,nwin//1,depth//1))

    nccr = nccrgpu.get()
    return nccr
