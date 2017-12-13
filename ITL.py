

import tensorflow as tf
import numpy as np


def gkernel(x, y, s):
    return tf.divide(1.0,tf.sqrt(tf.multiply(tf.multiply(2.0,np.pi),s))) * tf.exp( tf.divide(-tf.pow(tf.subtract(x,y), 2.0),tf.multiply(2.0,tf.pow(s, 2.0))) )


def ncc(x,y,marray,s):
    def nloop(m):
        mm = tf.cast(m, tf.int32)
        N = tf.shape(x)[0]
        nx = tf.range(0, tf.subtract(N,tf.abs(mm)))
        ny = tf.subtract(nx, mm)
        return tf.reduce_mean(gkernel(tf.gather(x, nx), tf.gather(y, ny), s))


    return tf.map_fn(lambda m: nloop(m), marray)

