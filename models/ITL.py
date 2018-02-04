
import tensorflow as tf
import numpy as np

def gram_op(x,y):
    return tf.pow(tf.subtract(x,y), 2.0)

def gram(x,y):
    with tf.name_scope('gspace') as scope:
        def rloop(i):
            return gram_op(tf.gather(x, tf.range(tf.shape(x)[2]), axis=2), tf.expand_dims(tf.gather(y, i, axis=2), dim=2))

        return tf.transpose(tf.reduce_mean(tf.map_fn(rloop, tf.range(tf.shape(y)[2]), dtype=tf.float32), axis=2), [1, 0, 2])


def gram_layer(ins):
    [x, y] = tf.unstack(ins, axis=3)

    grm = gram(x,y)
    grm = tf.image.per_image_standardization(grm)
    grm = tf.expand_dims(grm, dim=3)

    grm.set_shape([x.get_shape().as_list()[0], x.get_shape().as_list()[-1], y.get_shape().as_list()[-1], 1])  # Fix lost dimensions
    return grm


def gkernel(x, y, s):
    return tf.divide(1.0,tf.sqrt(tf.multiply(tf.multiply(2.0,np.pi),s))) * tf.exp( tf.divide(tf.multiply(-1.0,tf.pow(tf.subtract(x,y), 2.0)),tf.multiply(2.0,tf.pow(s, 2.0))) )


def gspace(x,y,s):
    with tf.name_scope('gspace') as scope:
        def rloop(i):
            return gkernel(tf.gather(x, tf.range(tf.shape(x)[2]), axis=2), tf.expand_dims(tf.gather(y, i, axis=2), dim=2), s)

        return tf.transpose(tf.reduce_mean(tf.map_fn(rloop, tf.range(tf.shape(y)[2]), dtype=tf.float32), axis=2), [1, 0, 2])


def gspace_layer(ins,Sigma):
    [x, y] = tf.unstack(ins, axis=3)

    gsr = gspace(x,y,Sigma)
    gsr = tf.image.per_image_standardization(gsr)
    gsr = tf.expand_dims(gsr, dim=3)

    gsr.set_shape([x.get_shape().as_list()[0], x.get_shape().as_list()[-1], y.get_shape().as_list()[-1], 1])  # Fix lost dimensions
    return gsr


def ncc(x, y, marray, s):
    with tf.name_scope('ncc') as scope:
        def nloop(m):
            N = tf.shape(x)[2]
            nx = tf.range(start=tf.abs(tf.minimum(0, m)), limit=tf.subtract(N - 1, tf.abs(m)))
            ny = tf.add(nx, m)

            return tf.reduce_mean(gkernel(tf.gather(x, nx, axis=2), tf.gather(y, ny, axis=2), s), axis=2)

        return tf.transpose(tf.map_fn(lambda m: nloop(m), marray, dtype=tf.float32), [1, 2, 0])


def ncc_layer(ins,marray,Sigma):
    [x,y] = tf.unstack(ins, axis=3)

    nccr = ncc(x, y, marray, Sigma)
    nccr = tf.image.per_image_standardization(nccr)
    nccr = tf.expand_dims(nccr, dim=3)

    # nccr.set_shape([x.get_shape().as_list()[0], x.get_shape().as_list()[1], marray.shape[0], 1])  # Fix lost dimensions
    return nccr

