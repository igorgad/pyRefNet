
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

import tensorflow as tf
import numpy as np
from numpy.lib.stride_tricks import as_strided


def gkernel(x, y, s):
    return tf.divide(1.0,tf.sqrt(tf.multiply(tf.multiply(2.0,np.pi),s))) * tf.exp( tf.divide(tf.multiply(-1.0,tf.pow(tf.subtract(x,y), 2.0)),tf.multiply(2.0,tf.pow(s, 2.0))) )


def ncc(x,y,marray,s):
    with tf.name_scope('ncc') as scope:

        def nloop(m):
            mm = tf.cast(m, tf.int32)
            N = tf.shape(x)[0]
            nx = tf.range( start=tf.abs(tf.minimum(0,mm)), limit=tf.subtract(N + 1, tf.abs(mm)) )
            ny = tf.add(nx, mm)
            return tf.reduce_mean(gkernel(tf.gather(x, nx), tf.gather(y, ny), s))


        return tf.map_fn(lambda m: nloop(m), marray)


def ncc_cuda(x,wx,y,wy,marray,s):
    """ x and y are 3d matrix. for every row pair of x and y, a normalized cross correntropy estimator is computed and stored in the nccr 3d matrix """

    mod = SourceModule(open('./cuda/ITLkernels.cu', 'r').read())
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


def ncclayer(ins,marray):
    """ This function creates a computational graph for the correntropy layer.
    ins is a tensor of shape [batchSize NumberOfSamples NumberOfWindows NumberOfSignals==2]
    marray is a tensor with rank 1 containing m values to be analyzed """

    [x,y] = tf.unstack(ins, axis=3)

    N = tf.shape(ins)[1]
    batchsize = tf.shape(ins)[0]
    nwin = tf.shape(ins)[2]

    Sigma = tf.Variable(1.0)

    tf.summary.scalar('sigma', Sigma)

    sx = tf.reshape(tf.transpose(x, [0, 2, 1]), [batchsize * nwin, N])
    sy = tf.reshape(tf.transpose(y, [0, 2, 1]), [batchsize * nwin, N])

    return tf.expand_dims(tf.transpose(tf.reshape(tf.map_fn(lambda i: ncc(sx[i, :], sy[i, :], marray, Sigma),
                                                            tf.range(batchsize * nwin), dtype=tf.float32), [batchsize, nwin, marray.size]), [0, 2, 1]), 3)
