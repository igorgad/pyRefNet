
import tensorflow as tf

features = {
        'comb/id': tf.FixedLenFeature([], tf.int64),
        'comb/class': tf.FixedLenFeature([], tf.int64),
        'comb/genre': tf.FixedLenFeature([], tf.string),
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

    parsed_features['comb/label'] = tf.cast(parsed_features['comb/label'], tf.int32)
    parsed_features['comb/ref']   = tf.cast(parsed_features['comb/ref'], tf.int32)
    parsed_features['comb/class'] = tf.cast(parsed_features['comb/class'], tf.int32)
    parsed_features['comb/id']    = tf.cast(parsed_features['comb/id'], tf.int32)

    return parsed_features


def split_train_test(parsed_features, train_rate):
    parsed_features['comb/is_train'] = tf.gather(tf.random_uniform([1], maxval=100, dtype=tf.int32) < tf.cast(train_rate * 100, tf.int32), 0)
    return parsed_features

def grab_train_examples(parsed_features):
    return parsed_features['comb/is_train']

def grab_test_examples(parsed_features):
    return ~parsed_features['comb/is_train']


def turn_into_autotest(parsed_features):
    ref = parsed_features['comb/label'] - (88200//1152) + 1

    def delay_positive():
        parsed_features['comb/sig2'] = tf.concat([tf.zeros(tf.abs(ref)), parsed_features['comb/sig1'][:-tf.abs(ref)]], axis=0)
        parsed_features['comb/lab2'] = tf.concat([tf.zeros(tf.abs(ref)), parsed_features['comb/lab1'][:-tf.abs(ref)]], axis=0)
        return parsed_features

    def delay_negative():
        parsed_features['comb/sig2'] = tf.concat([parsed_features['comb/sig1'][tf.abs(ref):], tf.zeros(tf.abs(ref))], axis=0)
        parsed_features['comb/lab2'] = tf.concat([parsed_features['comb/lab1'][tf.abs(ref):], tf.zeros(tf.abs(ref))], axis=0)
        return parsed_features

    parsed_features = tf.cond(tf.greater_equal(ref, 0), delay_positive, delay_negative)

    return parsed_features


def use_activation_signal_instead_of_bit_rate_signal(parsed_features):
    parsed_features['comb/sig1'] = parsed_features['comb/lab1']
    parsed_features['comb/sig2'] = parsed_features['comb/lab2']
    return parsed_features


def filter_split_dataset_from_ids(parsed_features, ids):
    id = parsed_features['comb/id']
    return tf.reduce_any(tf.equal(id,ids))


def filter_combinations_with_voice(parsed_features):
    type1 = parsed_features['comb/type1']
    type2 = parsed_features['comb/type2']
    return tf.logical_not(tf.logical_or(tf.equal(type1, 'voice'), tf.equal(type2, 'voice')))


def filter_perclass(parsed_features, selected_class):
    cls = parsed_features['comb/class']
    return tf.reduce_any(tf.equal(cls,selected_class))


def filter_sigsize_leq_N(parsed_features, N):
    sig1 = parsed_features['comb/sig1']
    sig2 = parsed_features['comb/sig2']
    sigsize = tf.minimum(tf.shape(sig1)[0], tf.shape(sig2)[0])
    return tf.greater_equal(sigsize,N)


def filter_perwindow(parsed_features, N, nwin, OR):
    sig1 = parsed_features['comb/sig1']
    sig2 = parsed_features['comb/sig2']

    nw1 = 1 + OR * tf.shape(sig1)[0] // N
    nw2 = 1 + OR * tf.shape(sig2)[0] // N

    return tf.reduce_all([tf.less_equal(nwin,nw1), tf.less_equal(nwin,nw2)])


def clean_from_activation_signal(parsed_features, activation_treshold):
    sig1 = parsed_features['comb/sig1']
    sig2 = parsed_features['comb/sig2']
    lab1 = parsed_features['comb/lab1']
    lab2 = parsed_features['comb/lab2']

    sigsize = tf.reduce_min([tf.shape(sig1)[0], tf.shape(sig2)[0], tf.shape(lab1)[0], tf.shape(lab2)[0]])
    sig1 = sig1[:sigsize]
    sig2 = sig2[:sigsize]
    lab1 = lab1[:sigsize]
    lab2 = lab2[:sigsize]

    sigfil1 = tf.reshape(tf.gather(sig1, tf.where(tf.logical_and(lab1 >= activation_treshold, lab2 >= activation_treshold)), axis=0), [-1])
    sigfil2 = tf.reshape(tf.gather(sig2, tf.where(tf.logical_and(lab1 >= activation_treshold, lab2 >= activation_treshold)), axis=0), [-1])

    m1, v1 = tf.nn.moments(sigfil1, axes=[0])
    m2, v2 = tf.nn.moments(sigfil2, axes=[0])

    parsed_features['comb/sig1'] = (sigfil1 - m1) / tf.sqrt(v1)
    parsed_features['comb/sig2'] = (sigfil2 - m2) / tf.sqrt(v2)

    return parsed_features


def replace_label_of_unselected_class(parsed_features, selected_class):
    label = tf.cast(parsed_features['comb/label'], tf.int32)
    cls   = tf.cast(parsed_features['comb/class'], tf.int32)
    is_from_selected_class = tf.reduce_any(tf.equal(cls, selected_class))  # Check if combination is from class of interest
    parsed_features['comb/label'] = tf.cond(is_from_selected_class, lambda: label, lambda: 0)
    return parsed_features


def normalize_blocks(sigmat):
    mean, var = tf.nn.moments(sigmat, axes=[0])
    signorm = (sigmat - mean) / tf.sqrt(var)
    return signorm


def prepare_input_with_random_sampling(parsed_features, N, nwin, OR):
    sig1 = parsed_features['comb/sig1']
    sig2 = parsed_features['comb/sig2']

    sigmat1 = tf.contrib.signal.frame(sig1, N, N // OR, pad_end=True, pad_value=0, axis=-1)
    sigmat2 = tf.contrib.signal.frame(sig2, N, N // OR, pad_end=True, pad_value=0, axis=-1)

    # Random window sampling
    wins = tf.random_uniform((nwin,), maxval=tf.minimum(tf.shape(sigmat1)[0], tf.shape(sigmat2)[0]), dtype=tf.int32)
    sigmat1 = tf.gather(sigmat1, wins, axis=0)
    sigmat2 = tf.gather(sigmat2, wins, axis=0)

    sigmat1 = normalize_blocks(sigmat1)
    sigmat2 = normalize_blocks(sigmat2)

    sigmat1 = tf.where(tf.is_nan(sigmat1), tf.zeros_like(sigmat1), sigmat1)
    sigmat2 = tf.where(tf.is_nan(sigmat2), tf.zeros_like(sigmat2), sigmat2)

    parsed_features['example/input'] = tf.stack((sigmat2, sigmat1), axis=2)

    return parsed_features


def prepare_input_with_all_windows(parsed_features, N, OR):
    sig1 = parsed_features['comb/sig1']
    sig2 = parsed_features['comb/sig2']

    sigmat1 = tf.contrib.signal.frame(sig1, N, N // OR, pad_end=True, pad_value=0, axis=-1)
    sigmat2 = tf.contrib.signal.frame(sig2, N, N // OR, pad_end=True, pad_value=0, axis=-1)

    sigmat1 = normalize_blocks(sigmat1)
    sigmat2 = normalize_blocks(sigmat2)

    parsed_features['example/input'] = tf.stack((sigmat2, sigmat1), axis=2)

    return parsed_features


def parse_example(parsed_features):
    label = tf.cast(parsed_features['comb/label'], tf.int32)
    ins = parsed_features['example/input']
    type1 = parsed_features['comb/type1']
    type2 = parsed_features['comb/type2']
    genre = parsed_features['comb/genre']

    return ins, label, tf.string_join([type1, ' x ', type2]), genre


def add_defaul_dataset_pipeline(trainParams, modelParams, iterator_handle):
    with tf.name_scope('dataset') as scope:
        with tf.device('/cpu:0'):
            datasetfile = trainParams.dataset_file
            # classes = trainParams.selected_class

            N = modelParams.N
            nwin = modelParams.nwin
            batch_size = modelParams.batch_size
            OR = modelParams.OR

            tfdataset = tf.data.TFRecordDataset(datasetfile)
            tfdataset = tfdataset.map(parse_features_and_decode, num_parallel_calls=4)
            # tfdataset = tfdataset.map(turn_into_autotest, num_parallel_calls=4) #USE FOR DEBUG ONLY
            # tfdataset = tfdataset.map(use_activation_signal_instead_of_bit_rate_signal) #USE FOR DEBUG ONLY

            # tfdataset = tfdataset.filter(lambda feat: filter_perclass(feat, classes))
            tfdataset = tfdataset.filter(filter_combinations_with_voice)
            # tfdataset = tfdataset.map(lambda feat: replace_label_of_unselected_class(feat, classes))
            tfdataset = tfdataset.map(lambda feat: clean_from_activation_signal(feat, 0.8), num_parallel_calls=4)
            tfdataset = tfdataset.filter(lambda feat: filter_sigsize_leq_N(feat, N))
            # tfdataset = tfdataset.filter(lambda feat: filter_perwindow(feat, N, nwin, OR))
            tfdataset = tfdataset.map(lambda feat: prepare_input_with_random_sampling(feat, N, nwin, OR), num_parallel_calls=4)
            # tfdataset = tfdataset.map(lambda feat: prepare_input_with_all_windows(feat, N, OR))

            tfdataset = tfdataset.map(lambda feat: split_train_test(feat, trainParams.train_test_rate))
            train_dataset = tfdataset.filter(grab_train_examples).map(parse_example)
            test_dataset = tfdataset.filter(grab_test_examples).map(parse_example)

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

