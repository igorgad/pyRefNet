

import tensorflow as tf
import numpy as np
import ITL


# TODO - encapsulate network params into a netparam dict
##### NETWORK PARAMS #####
N = 256     # VBR signal length
nwin = 64   # Number of windows
nsigs = 2   # Amount of signals

marray = np.array(range(-80,80)).astype(np.int32) # marray vary from -80 -79 ... 79
sigma = 1

medfiltersize = 8
medinit = 1/medfiltersize * np.ones((1, medfiltersize, 1, 1), dtype=np.float32)

shapeconv2 = [8, 1, 1, 64]
shapeconv3 = [4, 1, 64, 32]
shapeconv4 = [4, 1, 32, 16]

fc1_nhidden = 4096
fc2_nhidden = 4096
nclass = len(marray)
##########################


def inference(ins, keep_prob):

    # Conv 1 Layer (Mean Filter)
    with tf.name_scope('conv_1'):
        wc1x = tf.Variable(medinit, trainable=True)
        wc1y = tf.Variable(medinit, trainable=True)
        bc1x = tf.Variable(0.0, trainable=True)
        bc1y = tf.Variable(0.0, trainable=True)

        [insx, insy] = tf.unstack(ins, axis=3)
        insx = tf.expand_dims(insx, axis=3)
        insy = tf.expand_dims(insy, axis=3)

        conv1x = tf.nn.conv2d(insx, wc1x, strides=[1, 1, 1, 1], padding='SAME') + bc1x
        conv1y = tf.nn.conv2d(insy, wc1y, strides=[1, 1, 1, 1], padding='SAME') + bc1y

        conv1 = tf.concat((conv1x, conv1y), axis=3)

        tf.summary.histogram('wc1x-gram', wc1x)
        tf.summary.histogram('wc1y-gram', wc1y)
        tf.summary.histogram('bc1x-gram', bc1x)
        tf.summary.histogram('bc1y-gram', bc1y)

    # Normalized Cross Correntropy Layer
    with tf.name_scope('ccc'):
        Sigma = tf.Variable(np.float32(sigma), trainable=True)

        ccc1 = ITL.ncclayer(conv1, marray, Sigma)

        tf.summary.image('ccc_img', ccc1)
        tf.summary.scalar('sigma', Sigma)

    # Conv 2 Layer
    with tf.name_scope('conv_2'):
        wc2 = tf.Variable( tf.truncated_normal(shape=shapeconv2, stddev=0.1) )
        bc2 =  tf.Variable(np.zeros(shapeconv2[3]).astype(np.float32))

        conv2 = tf.nn.relu( tf.nn.conv2d(ccc1, wc2, strides=[1,1,1,1], padding='SAME') + bc2 )

        tf.summary.histogram('wc2-gram', wc2)
        tf.summary.histogram('bc2-gram', bc2)

    with tf.name_scope('pool_2'):
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

    with tf.name_scope('dropout_conv2'):
        drop2 = tf.nn.dropout(pool2, keep_prob)

    # Conv 3 Layer
    with tf.name_scope('conv_3'):
        wc3 = tf.Variable( tf.truncated_normal(shape=shapeconv3, stddev=0.1) )
        bc3 =  tf.Variable(np.zeros(shapeconv3[3]).astype(np.float32))

        conv3 = tf.nn.relu( tf.nn.conv2d(drop2, wc3, strides=[1,1,1,1], padding='SAME') + bc3 )

        tf.summary.histogram('wc3-gram', wc3)
        tf.summary.histogram('bc3-gram', bc3)

    with tf.name_scope('pool_3'):
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

    with tf.name_scope('dropout_conv3'):
        drop3 = tf.nn.dropout(pool3, keep_prob)

    # Conv 4 Layer
    with tf.name_scope('conv_4'):
        wc4 = tf.Variable( tf.truncated_normal(shape=shapeconv4, stddev=0.1) )
        bc4 =  tf.Variable(np.zeros(shapeconv4[3]).astype(np.float32))

        conv4 = tf.nn.relu( tf.nn.conv2d(drop3, wc4, strides=[1,1,1,1], padding='SAME') + bc4 )

        tf.summary.histogram('wc4-gram', wc4)
        tf.summary.histogram('bc4-gram', bc4)

    with tf.name_scope('pool_4'):
        pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

    with tf.name_scope('dropout_conv4'):
        drop4 = tf.nn.dropout(pool4, keep_prob)

    #Flatten tensors
    fcshape = np.int32([-1, nwin/8 * marray.size * shapeconv4[3]])

    with tf.name_scope('flattening'):
        flat4 = tf.reshape(drop4, fcshape)

    # FC 1 Layer
    with tf.name_scope('fc_1'):
        wfc1 = tf.Variable( tf.truncated_normal(shape=[fcshape[1],fc1_nhidden], stddev=0.1) )
        bfc1 =  tf.Variable(np.zeros(fc1_nhidden).astype(np.float32))

        fc1 = tf.nn.relu(tf.matmul(flat4, wfc1) + bfc1)

        tf.summary.histogram('wfc1-gram', wfc1)
        tf.summary.histogram('bfc1-gram', bfc1)

    with tf.name_scope('dropout_fc1'):
        dropfc1 = tf.nn.dropout(fc1, keep_prob)

    # FC 2 Layer
    with tf.name_scope('fc_2'):
        wfc2 = tf.Variable( tf.truncated_normal(shape=[fc1_nhidden,fc2_nhidden], stddev=0.1) )
        bfc2 =  tf.Variable(np.zeros(fc2_nhidden).astype(np.float32))

        fc2 = tf.nn.relu(tf.matmul(dropfc1, wfc2) + bfc2)

        tf.summary.histogram('wfc2-gram', wfc2)
        tf.summary.histogram('bfc2-gram', bfc2)

    with tf.name_scope('dropout_fc2'):
        dropfc2 = tf.nn.dropout(fc2, keep_prob)

    # Logits Layer
    with tf.name_scope('logits'):
        wl = tf.Variable(tf.truncated_normal(shape=[fc2_nhidden,nclass], stddev=0.1))
        bl = tf.Variable(np.zeros(nclass).astype(np.float32))

        logits = tf.matmul(dropfc2, wl) + bl

    return logits


def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate, momentum):
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)

    # Create the gradient descent optimizer with the given learning rate.
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)

    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    correct1 = tf.nn.in_top_k(logits, labels, 1)
    correct5 = tf.nn.in_top_k(logits, labels, 5)

    eval1 = tf.reduce_mean(tf.cast(correct1, tf.int32))
    eval5 = tf.reduce_mean(tf.cast(correct5, tf.int32))

    tf.summary.scalar('top-1', eval1)
    tf.summary.scalar('top-5', eval5)

    return eval1, eval5


