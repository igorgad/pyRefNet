

import tensorflow as tf
import numpy as np
import ITL
import matplotlib.pylab as p


# Initialize numpy variables
mr = np.array(range(-64,64))
insample = np.random.randn(10,128,16,1)
insample = np.repeat(insample,2,axis=3)
sigma = 0.1

# Initialize tf graph
ins = tf.placeholder(tf.float32, shape=np.shape(insample))
m = tf.placeholder(tf.float32, shape=np.shape(mr))

rs = ITL.ncclayer(ins,m)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

feed_dict = {ins: insample, m: mr}
r = sess.run(rs, feed_dict=feed_dict)


print(r)
p.plot(mr,r[0,:])
