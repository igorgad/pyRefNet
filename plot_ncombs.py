
import os
import models.ITL as ITL
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

features = {
        'comb/id': tf.FixedLenFeature([], tf.int64),
        'comb/class': tf.FixedLenFeature([], tf.int64),
        'comb/inst1': tf.FixedLenFeature([], tf.string),
        'comb/inst2': tf.FixedLenFeature([], tf.string),
        'comb/type1': tf.FixedLenFeature([], tf.string),
        'comb/type2': tf.FixedLenFeature([], tf.string),
        'comb/sig1': tf.FixedLenFeature([], tf.string),
        'comb/sig2': tf.FixedLenFeature([], tf.string),
        'comb/lab1': tf.FixedLenFeature([], tf.string),
        'comb/lab2': tf.FixedLenFeature([], tf.string),
        'comb/ref': tf.FixedLenFeature([], tf.int64),
        'comb/label': tf.FixedLenFeature([], tf.int64),
    }

initdiscard = 300
N = 256
nwin = 64
OR = 4
ncombs = 192401 // 50
np.random.seed(0)
testIds = np.random.randint(0,ncombs, [int(ncombs * 0.2)])
trainIds = np.setdiff1d(np.array(range(0,ncombs)), testIds)

def filter_train_examples(tf_example):
    parsed_features = tf.parse_single_example(tf_example, features)
    id = parsed_features['comb/id']
    return tf.reduce_any(tf.equal(id,trainIds))


def filter_test_examples(tf_example):
    parsed_features = tf.parse_single_example(tf_example, features)
    id = parsed_features['comb/id']
    return tf.reduce_any(tf.equal(id,testIds))


def filter_perclass_examples(tf_example, selected_class):
    parsed_features = tf.parse_single_example(tf_example, features)
    cls = parsed_features['comb/class']
    return tf.reduce_any(tf.equal(cls,selected_class))


def filter_perwindow_examples(tf_example, N, nwin, OR):
    parsed_features = tf.parse_single_example(tf_example, features)

    sig1 = tf.reshape(tf.decode_raw(parsed_features['comb/sig1'], tf.float32), [-1])
    sig2 = tf.reshape(tf.decode_raw(parsed_features['comb/sig2'], tf.float32), [-1])

    nw1 = 1 + OR * tf.shape(sig1)[0] // N
    nw2 = 1 + OR * tf.shape(sig2)[0] // N

    return tf.logical_and(tf.less_equal(nwin,nw1), tf.less_equal(nwin,nw2))


with tf.device('/cpu:0'):
    datasetfile  = '/home/pepeu/workspace/Dataset/SME_bitrate_medleydb_xpan10_split8_blocksize1024.tfrecord'

    tfdataset = tf.data.TFRecordDataset(datasetfile, compression_type='GZIP', buffer_size=4096)
    # tfdataset = tfdataset.filter(lambda ex: filter_perclass_examples(ex, [3,4,5]))
    # tfdataset = tfdataset.filter(lambda ex: filter_perwindow_examples(ex, N, nwin, OR))

    iter = tfdataset.repeat(40).make_one_shot_iterator()
    ne = iter.get_next()

    parsed_features = tf.parse_single_example(ne, features)

    label = tf.cast(parsed_features['comb/label'], tf.int32)
    type1 = tf.cast(parsed_features['comb/type1'], tf.string)
    type2 = tf.cast(parsed_features['comb/type2'], tf.string)
    cls = tf.cast(parsed_features['comb/class'], tf.int32)


sess = tf.Session()
vcls = []
count = 0

# sess.run(iterator.initializer)
while True:
    try:
        vcls.append(sess.run([type1, type2, cls]))
        count += 1

    except tf.errors.OutOfRangeError:
        print('found ' + str(count) + ' combinations...')
        break

# vcls = vcls[:count//50]

vcc = [[os.fsdecode(vcls[j][i]) for i in range(2)] for j in range(len(vcls))]
clss = [i[2] for i in vcls]

slb = [sorted(i) for i in vcc]
vstr = [''.join([c[0], ' x ', c[1]]) for c in slb]

mycolors = list([('b', 'b', 'b', 'b', 'b', 'r', 'b', 'b', 'b', 'b', 'r', 'b', 'b', 'b', 'r', 'b', 'b', 'r', 'b', 'r', 'r')])

import pandas
from collections import Counter
letter_counts = Counter(vstr)
df = pandas.DataFrame.from_dict(letter_counts, orient='index')
df = df.sort_index(axis=0)

ax = df.plot(kind='bar', legend=False, stacked=True, color=mycolors)
ax.set_xlabel('combinations per type')
ax.set_ylabel('number of combinations')

fig = ax.get_figure()
fig.set_tight_layout(True)

#
# count = 0
#
# while True:
#     try:
#         sess.run(train_element)
#         count += 1
#
#     except tf.errors.OutOfRangeError:
#         print('found ' + str(count) + ' training combinations...')
#         break
#
# count = 0
#
# while True:
#     try:
#         sess.run(test_element)
#         count += 1
#
#     except tf.errors.OutOfRangeError:
#         print('found ' + str(count) + ' testing combinations...')
#         break