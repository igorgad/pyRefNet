
import tensorflow as tf
import numpy as np
import models.ITL as ITL

name = 'CONVFC-COVLOSS-'
# TODO - encapsulate network params into a netparam dict
##### NETWORK PARAMS #####
N = 256     # VBR signal length
nwin = 32    # Number of windows
nsigs = 2   # Amount of signals
OR = 4      # Frame Overlap Ratio
batch_size = 1
lr = 0.0001

mbdelay = 88200 // 1024 + 2
trefClass = np.array(range(-mbdelay ,mbdelay)).astype(np.int32)
# sigma = 10

kp = 0.5

medfiltersize = 4
medinit = 1/ medfiltersize * np.ones((1, medfiltersize, 1, 1), dtype=np.float32)

# shapeconv2 = [9, 9, 8, 16] # 256
# shapeconv3 = [5, 5, 16, 32] # 128
# shapeconv4 = [5, 5, 32, 64] # 64 -> 32

fc1_nhidden = len(trefClass) * 2
# fc2_nhidden = 1024
nclass = len(trefClass)

medconvtrain = True

hptext = {'model_name': name, 'N': N, 'nwin': nwin, 'lr': lr, 'kp': kp, 'medconvtrain': medconvtrain,
          'batch_size': batch_size,
          # 'sigma': sigma,
          'medfiltersize': medfiltersize,
          # 'shapeconv2': shapeconv2,
          # 'shapeconv3': shapeconv3,
          # 'shapeconv4': shapeconv4,
          'fc1_nhidden': fc1_nhidden,
          'nclass': nclass}

##########################

fc_x = None
fc_y = None


def activation(inp):
    return tf.nn.relu(inp)


def pooling(inp):
    # return tf.layers.average_pooling2d(inp, pool_size=[2,2], strides=[2, 2], padding='SAME')
    return tf.layers.max_pooling2d(inp, pool_size=[2, 2], strides=[2, 2], padding='SAME')


xavier_init = tf.contrib.layers.xavier_initializer(uniform=True)
xavier_init_conv2d = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)


def inference(ins, keep_prob):
    global fc_x, fc_y

    ins = tf.transpose(ins, [1, 0, 2, 3])
    ins.set_shape([nwin, 1, N, nsigs])

    # Conv 1 Layer (Mean Filter)
    with tf.name_scope('conv_1'):
        [insx, insy] = tf.unstack(ins, axis=3)
        insx = tf.expand_dims(insx, axis=3)
        insy = tf.expand_dims(insy, axis=3)

        dwc1 = xavier_init_conv2d([1, 8, 1, 4])  # tf.Variable(medinit, trainable=medconvtrain)
        dbc1 = tf.Variable(0.0, trainable=medconvtrain)
        conv1x = tf.nn.relu(tf.nn.conv2d(insx, dwc1, strides=[1, 1, 1, 1], padding='SAME') + dbc1)
        conv1y = tf.nn.relu(tf.nn.conv2d(insy, dwc1, strides=[1, 1, 1, 1], padding='SAME') + dbc1)

        dwc2 = xavier_init_conv2d([1, 4, 4, 8])  # tf.Variable(medinit, trainable=medconvtrain)
        dbc2 = tf.Variable(0.0, trainable=medconvtrain)
        conv2x = tf.nn.relu(tf.nn.conv2d(conv1x, dwc2, strides=[1, 1, 1, 1], padding='SAME') + dbc2)
        conv2y = tf.nn.relu(tf.nn.conv2d(conv1y, dwc2, strides=[1, 1, 1, 1], padding='SAME') + dbc2)

        flat_x = tf.layers.flatten(conv2x)
        flat_y = tf.layers.flatten(conv2y)

        fcw = xavier_init([flat_x.get_shape().as_list()[-1], N])
        fcb = tf.Variable(np.zeros(N).astype(np.float32))

        fc_x = tf.matmul(flat_x, fcw) + fcb
        fc_y = tf.matmul(flat_y, fcw) + fcb

        tf.summary.histogram('wc1-gram', dwc1)
        tf.summary.histogram('bc1-gram', dbc1)
        tf.summary.histogram('wc2-gram', dwc2)
        tf.summary.histogram('bc2-gram', dbc2)
        tf.summary.histogram('fcw-gram', fcw)
        tf.summary.histogram('fcb-gram', fcb)

    # Normalized Cross Correntropy Layer
    with tf.name_scope('gram'):
        # Sigma = tf.Variable(np.float32(sigma), trainable=False)

        hs = ITL.gram(fc_x, fc_y)
        hs = tf.expand_dims(hs, axis=-1)

        himg = tf.unstack(hs, axis=3)
        tf.summary.image('gram', tf.expand_dims(himg[0], axis=3))
        # tf.summary.scalar('sigma', Sigma)

    # Conv 2 Layer
    #     with tf.name_scope('conv_2'):
    #         wc2 = tf.Variable(xavier_init_conv2d(shapeconv2))

    #         conv2 = tf.nn.conv2d(hs, wc2, strides=[1,1,1,1], padding='SAME')
    #         conv2 = tf.layers.batch_normalization(conv2, center=True, scale=False)
    #         # conv2 = activation(conv2)
    #         conv2 = pooling(conv2)

    #         p2feat = tf.unstack(conv2, axis=3)
    #         tf.summary.image('conv2_feat', tf.expand_dims(p2feat[0], axis=3))
    #         tf.summary.histogram('wc2-gram', wc2)
    #
    # # Conv 3 Layer
    # with tf.name_scope('conv_3'):
    #     wc3 = tf.Variable(xavier_init_conv2d(shapeconv3))
    #
    #     conv3 = tf.nn.conv2d(conv2, wc3, strides=[1,1,1,1], padding='SAME')
    #     conv3 = tf.layers.batch_normalization(conv3, center=True, scale=False)
    #     conv3 = activation(conv3)
    #     conv3 = pooling(conv3)
    #
    #     p3feat = tf.unstack(conv3, axis=3)
    #     tf.summary.image('conv3_feat', tf.expand_dims(p3feat[0], axis=3))
    #     tf.summary.histogram('wc3-gram', wc3)
    #
    # # Conv 4 Layer
    # with tf.name_scope('conv_4'):
    #     wc4 = tf.Variable(xavier_init_conv2d(shapeconv4))
    #
    #     conv4 = tf.nn.conv2d(conv3, wc4, strides=[1,1,1,1], padding='SAME')
    #     conv4 = tf.layers.batch_normalization(conv4, center=True, scale=False)
    #     conv4 = activation(conv4)
    #     conv4 = pooling(conv4)
    #
    #     p4feat = tf.unstack(conv4, axis=3)
    #     tf.summary.image('conv4_feat', tf.expand_dims(p4feat[0], axis=3))
    #     tf.summary.histogram('wc4-gram', wc4)

    # Flatten tensors
    with tf.name_scope('flattening'):
        flat4 = tf.layers.flatten(hs)

    # FC 1 Layer
    with tf.name_scope('fc_1'):
        wfc1 = tf.Variable(xavier_init([flat4.get_shape().as_list()[-1], fc1_nhidden]))
        bfc1 = tf.Variable(np.zeros(fc1_nhidden).astype(np.float32))

        fc1 = activation(tf.matmul(flat4, wfc1) + bfc1)
        fc1 = tf.nn.dropout(fc1, keep_prob)

        tf.summary.histogram('wfc1-gram', wfc1)
        tf.summary.histogram('bfc1-gram', bfc1)

    # FC 2 Layer
    #     with tf.name_scope('fc_2'):
    #         wfc2 = tf.Variable(xavier_init([fc1_nhidden, fc2_nhidden]))
    #         bfc2 = tf.Variable(np.zeros(fc2_nhidden).astype(np.float32))

    #         fc2 = activation(tf.matmul(dropfc1, wfc2) + bfc2)
    #         dropfc2 = tf.nn.dropout(fc2, keep_prob)

    #         tf.summary.histogram('wfc2-gram', wfc2)
    #         tf.summary.histogram('bfc2-gram', bfc2)

    # Logits Layer
    with tf.name_scope('logits'):
        wl = tf.Variable(xavier_init([fc1_nhidden, nclass]))
        bl = tf.Variable(np.zeros(nclass).astype(np.float32))

        logits = tf.matmul(fc1, wl) + bl

        tf.summary.histogram('w-logits', wl)
        tf.summary.histogram('b-logits', bl)

    # tf.summary.image('logits', tf.expand_dims(tf.expand_dims(logits, axis=0), axis=3))
    return logits, hs


def loss(logits, labels):
    global fc_x, fc_y

    fc_x -= tf.reduce_mean(fc_x, 0)
    fc_y -= tf.reduce_mean(fc_y, 0)

    correlation_matrix = tf.matmul(fc_x, fc_y, transpose_a=True) / (tf.to_float(tf.shape(fc_x)[0]) - 1.0)
    source_covariance = tf.matmul(fc_x, fc_x, transpose_a=True) / (tf.to_float(tf.shape(fc_x)[0]) - 1.0)

    # source_covariance = source_covariance + tf.ones_like(source_covariance) * tf.reduce_min(source_covariance)
    # root_source_covariance = tf.cholesky(source_covariance)
    # inv_root_source_covariance = tf.matrix_inverse(root_source_covariance)

    canonical_correlation = tf.matmul(source_covariance, correlation_matrix)

    loss, u, v = tf.svd(canonical_correlation)
    return -loss

    # labels = tf.to_int64(labels)
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    # return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, global_step):
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    labels = tf.tile(labels, [tf.shape(logits)[0]])
    correct1 = tf.nn.in_top_k(logits, labels, 1)
    correct5 = tf.nn.in_top_k(logits, labels, 5)

    eval1 = tf.reduce_mean(tf.cast(correct1, tf.float32))
    eval5 = tf.reduce_mean(tf.cast(correct5, tf.float32))

    return eval1, eval5, correct1, correct5
