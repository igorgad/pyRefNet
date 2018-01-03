
import tensorflow as tf
import numpy as np


def gkernel(x, y, s):
    return tf.divide(1.0,tf.sqrt(tf.multiply(tf.multiply(2.0,np.pi),s))) * tf.exp( tf.divide(tf.multiply(-1.0,tf.pow(tf.subtract(x,y), 2.0)),tf.multiply(2.0,tf.pow(s, 2.0))) )


def gspace(x,y,s):
    with tf.name_scope('gspace'):
        def rloop(i):
            return gkernel(tf.gather(x, tf.range(tf.shape(x)[2]), axis=2), tf.expand_dims(tf.gather(y, i, axis=2), dim=2), s)

        return tf.transpose(tf.reduce_mean(tf.map_fn(rloop, tf.range(tf.shape(y)[2]), dtype=tf.float32), axis=2), [1, 0, 2])


def gspace_layer(ins,Sigma):
    [x, y] = tf.unstack(ins, axis=3)

    gsr = gspace(x,y,Sigma)
    gsr = tf.image.per_image_standardization(gsr)
    gsr = tf.expand_dims(gsr, dim=3)

    return gsr


def ncc(x, y, marray, s):
    with tf.name_scope('ncc') as scope:
        def nloop(m):
            N = tf.shape(x)[2]
            nx = tf.range(start=tf.abs(tf.minimum(0, m)), limit=tf.subtract(N + 1, tf.abs(m)))
            ny = tf.add(nx, m)

            return tf.reduce_mean(gkernel(tf.gather(x, nx, axis=2), tf.gather(y, ny, axis=2), s), axis=2)

        return tf.transpose(tf.map_fn(lambda m: nloop(m), marray, dtype=tf.float32), [1, 2, 0])


def ncc_layer(ins,marray,Sigma):
    """ This function creates a computational graph for the correntropy layer.
    ins is a tensor of shape [batchSize NumberOfSamples NumberOfWindows NumberOfSignals==2]
    marray is a tensor with rank 1 containing m values to be analyzed """

    [x,y] = tf.unstack(ins, axis=3)

    nccr = ncc(x, y, marray, Sigma)
    nccr = tf.image.per_image_standardization(nccr)
    nccr = tf.expand_dims(nccr, dim=3)

    return nccr
