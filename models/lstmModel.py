

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

name = 'lstm'
# TODO - encapsulate network params into a netparam dict
##### NETWORK PARAMS #####
N = 256     # VBR signal length - time steps
nwin = 64   # Number of windows - number of inputs
nsigs = 2   # Amount of signals

trefClass = np.array(range(-80,80)).astype(np.int32)

lstm_units=128
fc1_nhidden = trefClass.size * 2
nclass = len(trefClass)

hptext = {'model_name': name, 'lstm_units': lstm_units, 'fc1_hidden': fc1_nhidden}
##########################


def inference(ins, keep_prob): # (bs, nw, N, ns)

    ins = tf.reshape(tf.transpose(ins, [0, 2, 1, 3]), [tf.shape(ins)[0], N, nwin * nsigs]) #(bs, N, nw * ns)

    # Conv 1 Layer (Mean Filter)
    with tf.name_scope('lstm'):
        input = tf.unstack(ins, N, axis=1) # [N](bs, nw * ns)

        lstm_fw_cell = rnn.BasicLSTMCell(lstm_units, forget_bias=1)
        lstm_bw_cell = rnn.BasicLSTMCell(lstm_units, forget_bias=1)

        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input, dtype=tf.float32) # [N](bs, 2 * lstm_units)

        outputs = tf.stack(outputs, axis=1) # (bs, N, 2 * lstm_units)

    #Flatten tensors
    with tf.name_scope('flattening'):
        flatp = tf.layers.flatten(outputs)

    # FC 1 Layer
    with tf.name_scope('fc_1'):
        wfc1 = tf.Variable( tf.truncated_normal(shape=[flatp.get_shape().as_list()[-1], fc1_nhidden], stddev=0.1) )
        bfc1 =  tf.Variable(np.zeros(fc1_nhidden).astype(np.float32))

        fc1 = tf.nn.relu(tf.matmul(flatp, wfc1) + bfc1)
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


def training(loss, learning_rate, momentum):

    global_step = tf.Variable(0, name='global_step', trainable=False)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    correct1 = tf.nn.in_top_k(logits, labels, 1)
    correct5 = tf.nn.in_top_k(logits, labels, 5)

    eval1 = tf.reduce_mean(tf.cast(correct1, tf.float32))
    eval5 = tf.reduce_mean(tf.cast(correct5, tf.float32))

    return eval1, eval5, correct1, correct5


