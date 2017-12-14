

import tensorflow as tf
import numpy as np


def gkernel(x, y, s):
    return tf.divide(1.0,tf.sqrt(tf.multiply(tf.multiply(2.0,np.pi),s))) * tf.exp( tf.divide(tf.multiply(-1,tf.pow(tf.subtract(x,y), 2.0)),tf.multiply(2.0,tf.pow(s, 2.0))) )


def ncc(x,y,marray,s):
    with tf.name_scope('ncc') as scope:

        def nloop(m):
            mm = tf.cast(m, tf.int32)
            N = tf.shape(x)[0]
            nx = tf.range( start=tf.abs(tf.minimum(0,mm)), limit=tf.subtract(N, tf.abs(mm)) )
            ny = tf.add(nx, mm)
            return tf.reduce_mean(gkernel(tf.gather(x, nx), tf.gather(y, ny), s))


        return tf.map_fn(lambda m: nloop(m), marray)


