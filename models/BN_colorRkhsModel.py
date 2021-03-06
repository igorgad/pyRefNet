

import tensorflow as tf
import numpy as np
import models.ITL as ITL

name = 'BN-color-rkhs'
# TODO - encapsulate network params into a netparam dict
##### NETWORK PARAMS #####
N = 256     # VBR signal length
nwin = 64   # Number of windows
nsigs = 2   # Amount of signals
OR = 4      # Frame Overlap Ratio
batch_size = 48
lr = 0.0001

trefClass = np.array(range(-80,80)).astype(np.int32)
sigma = 10

kp = 0.6

medfiltersize = 8
medinit = 1/medfiltersize * np.ones((1, medfiltersize, 1, 1), dtype=np.float32)

shapeconv2 = [9, 9, 3, 8]
shapeconv3 = [5, 5, 8, 16]
shapeconv4 = [5, 5, 16, 32]

fc1_nhidden = 4096
fc2_nhidden = 1024
nclass = len(trefClass)

medconvtrain = False

hptext = {'model_name': name, 'N': N, 'nwin': nwin, 'lr': lr, 'kp': kp, 'medconvtrain': medconvtrain,
          'batch_size': batch_size,  'sigma': sigma, 'medfiltersize': medfiltersize,
          'shapeconv2': shapeconv2, 'shapeconv3': shapeconv3, 'shapeconv4': shapeconv4, 'fc1_nhidden': fc1_nhidden, 'fc2_nhidden': fc2_nhidden, 'nclass': nclass}


##########################

def activation(inp):
    return tf.nn.leaky_relu(inp)

xavier_init =  tf.contrib.layers.xavier_initializer(uniform=True)
xavier_init_conv2d =  tf.contrib.layers.xavier_initializer_conv2d(uniform=True)

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
        Sigma = tf.Variable(np.float32(sigma), trainable=False)

        hs = ITL.gspace_color_layer(conv1, Sigma)

        tf.summary.image('rkhs_color', hs)

    # Conv 2 Layer
    with tf.name_scope('conv_2'):
        wc2 = tf.Variable(xavier_init_conv2d(shapeconv2))

        conv2 = tf.nn.conv2d(hs, wc2, strides=[1,1,1,1], padding='SAME')
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=False)
        conv2 = activation(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        p2feat = tf.unstack(conv2, axis=3)
        tf.summary.image('conv2_feat', tf.expand_dims(p2feat[0], axis=3))
        tf.summary.histogram('wc2-gram', wc2)

    # Conv 3 Layer
    with tf.name_scope('conv_3'):
        wc3 = tf.Variable(xavier_init_conv2d(shapeconv3))

        conv3 = tf.nn.conv2d(conv2, wc3, strides=[1,1,1,1], padding='SAME')
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=False)
        conv3 = activation(conv3)
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        p3feat = tf.unstack(conv3, axis=3)
        tf.summary.image('conv3_feat', tf.expand_dims(p3feat[0], axis=3))
        tf.summary.histogram('wc3-gram', wc3)

    # Conv 4 Layer
    with tf.name_scope('conv_4'):
        wc4 = tf.Variable(xavier_init_conv2d(shapeconv4))

        conv4 = tf.nn.conv2d(conv3, wc4, strides=[1,1,1,1], padding='SAME')
        conv4 = tf.layers.batch_normalization(conv4, center=True, scale=False)
        conv4 = activation(conv4)
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        p4feat = tf.unstack(conv4, axis=3)
        tf.summary.image('conv4_feat', tf.expand_dims(p4feat[0], axis=3))
        tf.summary.histogram('wc4-gram', wc4)

    #Flatten tensors
    with tf.name_scope('flattening'):
        flat4 = tf.layers.flatten(conv3)

    # FC 1 Layer
    with tf.name_scope('fc_1'):
        wfc1 = tf.Variable(xavier_init([flat4.get_shape().as_list()[-1], fc1_nhidden]))
        bfc1 =  tf.Variable(np.zeros(fc1_nhidden).astype(np.float32))

        fc1 = activation(tf.matmul(flat4, wfc1) + bfc1)
        dropfc1 = tf.nn.dropout(fc1, keep_prob)

        tf.summary.histogram('wfc1-gram', wfc1)
        tf.summary.histogram('bfc1-gram', bfc1)

    # FC 2 Layer
    with tf.name_scope('fc_2'):
        wfc2 = tf.Variable(xavier_init([fc1_nhidden, fc2_nhidden]))
        bfc2 = tf.Variable(np.zeros(fc2_nhidden).astype(np.float32))

        fc2 = activation(tf.matmul(dropfc1, wfc2) + bfc2)
        dropfc2 = tf.nn.dropout(fc2, keep_prob)

        tf.summary.histogram('wfc2-gram', wfc2)
        tf.summary.histogram('bfc2-gram', bfc2)

    # Logits Layer
    with tf.name_scope('logits'):
        wl = tf.Variable(xavier_init([fc2_nhidden,nclass]))
        bl = tf.Variable(np.zeros(nclass).astype(np.float32))

        logits = tf.matmul(dropfc2, wl) + bl

        tf.summary.histogram('w-logits', wl)
        tf.summary.histogram('b-logits', bl)

    # tf.summary.image('logits', tf.expand_dims(tf.expand_dims(logits, axis=0), axis=3))
    return logits


def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, global_step):
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    correct1 = tf.nn.in_top_k(logits, labels, 1)
    correct5 = tf.nn.in_top_k(logits, labels, 5)

    eval1 = tf.reduce_mean(tf.cast(correct1, tf.float32))
    eval5 = tf.reduce_mean(tf.cast(correct5, tf.float32))

    return eval1, eval5, correct1, correct5


