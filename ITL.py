

import tensorflow as tf
import numpy as np


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


def ncclayer(ins,marray):
    """ This function creates a computational graph for the correntropy layer.
    ins is a tensor of shape [batchSize NumberOfSamples NumberOfWindows NumberOfSignals==2]
    marray is a tensor with rank 1 containing m values to be analyzed """

    [x,y] = tf.unstack(ins, axis=3)

    N = tf.shape(ins)[1]
    batchsize = tf.shape(ins)[0]
    nwin = tf.shape(ins)[2]

    Sigma = tf.Variable(1.0)

    sx = tf.reshape(tf.transpose(x, [0, 2, 1]), [batchsize * nwin, N])
    sy = tf.reshape(tf.transpose(y, [0, 2, 1]), [batchsize * nwin, N])

    return tf.transpose(tf.reshape(tf.map_fn(lambda i: ncc(sx[i, :], sy[i, :], marray, Sigma), tf.range(batchsize * nwin), dtype=tf.float32), [batchsize, nwin, N]), [0, 2, 1])

