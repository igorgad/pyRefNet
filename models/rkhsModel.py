

import tensorflow as tf
import numpy as np
import models.ITL as ITL

name = 'color-rkhs'
# TODO - encapsulate network params into a netparam dict
##### NETWORK PARAMS #####
N = 256     # VBR signal length
nwin = 32   # Number of windows
nsigs = 2   # Amount of signals
OR = 4      # Overlap Ratio
batch_size = 16
lr = 0.0001

trefClass = np.array(range(-80,80)).astype(np.int32)
sigma = 10

kp = 0.7

medfiltersize = 8
medinit = 1/medfiltersize * np.ones((1, medfiltersize, 1, 1), dtype=np.float32)

shapeconv2 = [9, 9, 3, 48]
shapeconv3 = [9, 9, 48, 24]
shapeconv4 = [5, 5, 24, 8]

fc1_nhidden = trefClass.size * 2
nclass = len(trefClass)

medconvtrain = True

hptext = {'model_name': name, 'N': N, 'nwin': nwin, 'lr': lr, 'medconvtrain': medconvtrain, 'batch_size': batch_size,  'sigma': sigma, 'medfiltersize': medfiltersize, 'shapeconv2': shapeconv2, 'shapeconv3': shapeconv3, 'shapeconv4': shapeconv4, 'fc1_nhidden': fc1_nhidden, 'nclass': nclass}
##########################


def inference(ins, keep_prob):

    # Conv 1 Layer (Mean Filter)
    with tf.name_scope('conv_1'):
        wc1x = tf.Variable(medinit, trainable=medconvtrain)
        wc1y = tf.Variable(medinit, trainable=medconvtrain)
        bc1x = tf.Variable(0.0, trainable=medconvtrain)
        bc1y = tf.Variable(0.0, trainable=medconvtrain)

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
    with tf.name_scope('rkhs'):
        Sigma = tf.Variable(np.float32(sigma), trainable=medconvtrain)

        hs = ITL.gspace_layer(conv1, Sigma)

        tf.summary.image('rkhs_color', hs)
        tf.summary.scalar('sigma', Sigma)

        # [hxx, hyy, hxy] = tf.unstack(hs, axis=3)
        # tf.summary.image('rkhs_xx', tf.expand_dims(hxx,axis=3))
        # tf.summary.image('rkhs_yy', tf.expand_dims(hyy,axis=3))
        # tf.summary.image('rkhs_xy', tf.expand_dims(hxy,axis=3))

    # Conv 2 Layer
    with tf.name_scope('conv_2'):
        wc2 = tf.Variable( tf.truncated_normal(shape=shapeconv2, stddev=0.1) )
        bc2 =  tf.Variable(np.zeros(shapeconv2[3]).astype(np.float32))

        conv2 = tf.nn.relu( tf.nn.conv2d(hs, wc2, strides=[1,1,1,1], padding='SAME') + bc2 )
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # drop2 = tf.nn.dropout(pool2, keep_prob)

        tf.summary.histogram('wc2-gram', wc2)
        tf.summary.histogram('bc2-gram', bc2)

    # Conv 3 Layer
    with tf.name_scope('conv_3'):
        wc3 = tf.Variable( tf.truncated_normal(shape=shapeconv3, stddev=0.1) )
        bc3 =  tf.Variable(np.zeros(shapeconv3[3]).astype(np.float32))

        conv3 = tf.nn.relu( tf.nn.conv2d(pool2, wc3, strides=[1,1,1,1], padding='SAME') + bc3 )
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # drop3 = tf.nn.dropout(pool3, keep_prob)

        tf.summary.histogram('wc3-gram', wc3)
        tf.summary.histogram('bc3-gram', bc3)

    # Conv 4 Layer
    with tf.name_scope('conv_4'):
        wc4 = tf.Variable( tf.truncated_normal(shape=shapeconv4, stddev=0.1) )
        bc4 =  tf.Variable(np.zeros(shapeconv4[3]).astype(np.float32))

        conv4 = tf.nn.relu( tf.nn.conv2d(pool3, wc4, strides=[1,1,1,1], padding='SAME') + bc4 )
        pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # drop4 = tf.nn.dropout(pool4, keep_prob)

        tf.summary.histogram('wc4-gram', wc4)
        tf.summary.histogram('bc4-gram', bc4)

    #Flatten tensors
    with tf.name_scope('flattening'):
        flat4 = tf.layers.flatten(pool4)

    # FC 1 Layer
    with tf.name_scope('fc_1'):
        wfc1 = tf.Variable( tf.truncated_normal(shape=[flat4.get_shape().as_list()[-1], fc1_nhidden], stddev=0.1) )
        bfc1 =  tf.Variable(np.zeros(fc1_nhidden).astype(np.float32))

        fc1 = tf.nn.relu(tf.matmul(flat4, wfc1) + bfc1)
        dropfc1 = tf.nn.dropout(fc1, keep_prob)

        tf.summary.histogram('wfc1-gram', wfc1)
        tf.summary.histogram('bfc1-gram', bfc1)

    # Logits Layer
    with tf.name_scope('logits'):
        wl = tf.Variable(tf.truncated_normal(shape=[fc1_nhidden,nclass], stddev=0.1))
        bl = tf.Variable(np.zeros(nclass).astype(np.float32))

        logits = tf.matmul(dropfc1, wl) + bl

        tf.summary.histogram('w-logits', wl)
        tf.summary.histogram('b-logits', bl)

    # tf.summary.image('logits', tf.expand_dims(tf.expand_dims(logits, axis=0), axis=3))
    return logits


def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss):
    global_step = tf.Variable(0, name='global_step', trainable=False)

    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    correct1 = tf.nn.in_top_k(logits, labels, 1)
    correct5 = tf.nn.in_top_k(logits, labels, 5)

    eval1 = tf.reduce_mean(tf.cast(correct1, tf.float32))
    eval5 = tf.reduce_mean(tf.cast(correct5, tf.float32))

    return eval1, eval5, correct1, correct5


