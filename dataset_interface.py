
import tensorflow as tf

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


def parse_features_and_decode(tf_example):
    parsed_features =  tf.parse_single_example(tf_example, features)

    parsed_features['comb/sig1'] = tf.reshape(tf.decode_raw(parsed_features['comb/sig1'], tf.float32), [-1])
    parsed_features['comb/sig2'] = tf.reshape(tf.decode_raw(parsed_features['comb/sig2'], tf.float32), [-1])
    parsed_features['comb/lab1'] = tf.reshape(tf.decode_raw(parsed_features['comb/lab1'], tf.float32), [-1])
    parsed_features['comb/lab2'] = tf.reshape(tf.decode_raw(parsed_features['comb/lab2'], tf.float32), [-1])

    return parsed_features

def filter_split_dataset_from_ids(parsed_features, ids):
    id = parsed_features['comb/id']
    return tf.reduce_any(tf.equal(id,ids))


def filter_perclass_examples(parsed_features, selected_class):
    cls = parsed_features['comb/class']
    return tf.reduce_any(tf.equal(cls,selected_class))


def filter_perwindow_examples(parsed_features, N, nwin, OR):
    sig1 = parsed_features['comb/sig1']
    sig2 = parsed_features['comb/sig2']

    nw1 = 1 + OR * tf.shape(sig1)[0] // N
    nw2 = 1 + OR * tf.shape(sig2)[0] // N

    return tf.logical_and(tf.less_equal(nwin,nw1), tf.less_equal(nwin,nw2))


def clean_from_activation_signal(parsed_features):
    sig1 = parsed_features['comb/sig1']
    sig2 = parsed_features['comb/sig2']
    lab1 = parsed_features['comb/lab1']
    lab2 = parsed_features['comb/lab2']

    sigsize = tf.minimum(tf.shape(sig1)[0], tf.shape(sig2)[0])
    sig1 = sig1[:sigsize]
    sig2 = sig2[:sigsize]
    lab1 = lab1[:sigsize]
    lab2 = lab2[:sigsize]

    sigfil1 = tf.reshape(tf.gather(sig1, tf.where(tf.logical_and(lab1 >= 0.5, lab2 >= 0.5)), axis=0), [-1])
    sigfil2 = tf.reshape(tf.gather(sig2, tf.where(tf.logical_and(lab1 >= 0.5, lab2 >= 0.5)), axis=0), [-1])

    parsed_features['comb/sig1'] = sigfil1
    parsed_features['comb/sig2'] = sigfil2

    return parsed_features


def replace_label_of_unselected_class(parsed_features, selected_class):
    label = tf.cast(parsed_features['comb/label'], tf.int32)
    cls   = tf.cast(parsed_features['comb/class'], tf.int32)
    is_from_selected_class = tf.reduce_any(tf.equal(cls, selected_class))  # Check if combination is from class of interest
    parsed_features['comb/label'] = tf.cond(is_from_selected_class, lambda: label, lambda: 0)
    return parsed_features


def prepare_input_with_random_sampling(parsed_features, N, nwin, OR):
    sig1 = parsed_features['comb/sig1']
    sig2 = parsed_features['comb/sig2']

    sigmat1 = tf.contrib.signal.frame(sig1, N, N // OR, pad_end=True, pad_value=0, axis=-1)
    sigmat2 = tf.contrib.signal.frame(sig2, N, N // OR, pad_end=True, pad_value=0, axis=-1)

    # Random window sampling
    wins = tf.random_uniform((nwin,), maxval=tf.shape(sigmat1)[0], dtype=tf.int32)
    sigmat1 = tf.gather(sigmat1, wins, axis=0)
    sigmat2 = tf.gather(sigmat2, wins, axis=0)

    sigmat1 = tf.squeeze(tf.image.per_image_standardization(tf.expand_dims(sigmat1, axis=2)), axis=2)
    sigmat2 = tf.squeeze(tf.image.per_image_standardization(tf.expand_dims(sigmat2, axis=2)), axis=2)

    parsed_features['example/input'] = tf.stack((sigmat1, sigmat2), axis=2)

    return parsed_features


def prepare_input_with_all_windows(parsed_features, N, OR):
    sig1 = parsed_features['comb/sig1']
    sig2 = parsed_features['comb/sig2']

    sigmat1 = tf.contrib.signal.frame(sig1, N, N // OR, pad_end=True, pad_value=0, axis=-1)
    sigmat2 = tf.contrib.signal.frame(sig2, N, N // OR, pad_end=True, pad_value=0, axis=-1)

    sigmat1 = tf.squeeze(tf.image.per_image_standardization(tf.expand_dims(sigmat1, axis=2)), axis=2)
    sigmat2 = tf.squeeze(tf.image.per_image_standardization(tf.expand_dims(sigmat2, axis=2)), axis=2)

    parsed_features['example/input'] = tf.stack((sigmat1, sigmat2), axis=2)

    return parsed_features


def parse_example(parsed_features):
    label = tf.cast(parsed_features['comb/label'], tf.int32)
    ins = parsed_features['example/input']
    type1 = parsed_features['comb/type1']
    type2 = parsed_features['comb/type2']

    return ins, label, tf.string_join([type1, ' x ', type2])


def add_defaul_dataset_pipeline(trainParams, modelParams, iterator_handle):
    with tf.name_scope('dataset') as scope:
        with tf.device('/cpu:0'):
            train_ids = trainParams.train_ids
            eval_ids = trainParams.eval_ids
            datasetfile = trainParams.datasetfile
            classes = trainParams.selected_class

            N = modelParams.N
            nwin = modelParams.nwin
            batch_size = modelParams.batch_size
            OR = modelParams.OR

            tfdataset = tf.data.TFRecordDataset(datasetfile)
            tfdataset = tfdataset.map(parse_features_and_decode, num_parallel_calls=4)

            tfdataset = tfdataset.filter(lambda feat: filter_perclass_examples(feat, classes))
            # tfdataset = tfdataset.map(lambda feat: replace_label_of_unselected_class(feat, classes))
            tfdataset = tfdataset.map(clean_from_activation_signal, num_parallel_calls=4)
            tfdataset = tfdataset.filter(lambda feat: filter_perwindow_examples(feat, N, nwin, OR))
            tfdataset = tfdataset.map(lambda feat: prepare_input_with_random_sampling(feat, N, nwin, OR), num_parallel_calls=4)
            # tfdataset = tfdataset.map(lambda feat: prepare_input_with_all_windows(feat, N, OR))
            train_dataset = tfdataset.filter(lambda feat: filter_split_dataset_from_ids(feat, train_ids)).map(parse_example)
            test_dataset = tfdataset.filter(lambda feat: filter_split_dataset_from_ids(feat, eval_ids)).map(parse_example)

            train_dataset = train_dataset.shuffle(4096, reshuffle_each_iteration=True)
            test_dataset = test_dataset.shuffle(4096, reshuffle_each_iteration=True)

            train_dataset = train_dataset.prefetch(buffer_size=4096)
            test_dataset = test_dataset.prefetch(buffer_size=4096)

            train_dataset = train_dataset.batch(batch_size)
            test_dataset = test_dataset.batch(batch_size)

            iterator = tf.data.Iterator.from_string_handle(iterator_handle, train_dataset.output_types, train_dataset.output_shapes)
            next_element = iterator.get_next()

            train_iterator = train_dataset.make_initializable_iterator()
            test_iterator = test_dataset.make_initializable_iterator()

    return next_element, train_iterator, test_iterator

