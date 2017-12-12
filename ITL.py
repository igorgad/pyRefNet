

import tensorflow as tf
import numpy as np


def gkernel(x, y, s):
    return (1 / tf.sqrt(2 * np.pi * s)) * tf.exp((- tf.pow(x - y, 2)) / (2 * tf.pow(s, 2)))


def ccsum(x,y,s):
    return tf.reduce_mean(gkernel(x,y,s))


def truefn(x,y,m):
    N = tf.shape(x)[1]
    m = tf.abs(m)
    xm = tf.gather(x, tf.range(0, tf.subtract(N, m)))
    ym = tf.gather(y, tf.range(m, N))
    return xm,ym


def falsefn(x,y,m):
    N = tf.shape(x)[1]
    m = tf.abs(m)
    xm = tf.gather(x, tf.range(m, N))
    ym = tf.gather(y, tf.range(0, tf.subtract(N, m)))
    return xm,ym


def msfunc(x,y,m,s):
    xm,ym = tf.cond(tf.less_equal(m,0), lambda : truefn(x,y,m), lambda : falsefn(x,y,m))
    return ccsum(xm,ym,s)


def nccfunc(x,y,marray,s):
    return tf.scan(lambda _,m: msfunc(x,y,m,s), marray)


