

import tensorflow as tf
import numpy as np
import ITL
import matplotlib.pylab as p


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
