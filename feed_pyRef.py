

import tensorflow as tf
import numpy as np
import ITL
import matplotlib.pylab as p

def gkernel(x, y, s):
    return tf.divide(1.0,tf.sqrt(tf.multiply(tf.multiply(2.0,np.pi),s))) * tf.exp( tf.divide(-tf.pow(tf.subtract(x,y), 2.0),tf.multiply(2.0,tf.pow(s, 2.0))) )


def mloop(mm):
    mm = tf.cast(mm,tf.int32)
    N = tf.shape(x)[0]
    nx = tf.range(0, tf.subtract(N,mm))
    ny = tf.subtract(nx,mm)
    return tf.reduce_mean(gkernel(tf.gather(x,nx),tf.gather(y,ny),sigma))


# Initialize numpy variables
mr = np.array(range(-64,65))
a = np.random.rand(1024) # Batch size, N, nwin, nsignals
b = np.random.rand(1024) # Batch size, N, nwin, nsignals
sigma = 1.0


# Initialize tf graph
x = tf.placeholder(tf.float32, shape=np.shape(a))
y = tf.placeholder(tf.float32, shape=np.shape(b))
m = tf.placeholder(tf.float32, shape=np.shape(mr))

ncc = ITL.ncc(x,y,m,sigma)

sess = tf.Session()
result = sess.run(ncc, {x: a, y: a, m: mr})
p.plot(mr,result)
