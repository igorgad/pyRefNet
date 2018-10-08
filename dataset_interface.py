import tensorflow as tf
import numpy as np

features = {
    'comb/id': tf.FixedLenFeature([], tf.int64),
    'comb/class': tf.FixedLenFeature([], tf.int64),
    'comb/genre': tf.FixedLenFeature([], tf.string),
    'comb/inst1': tf.FixedLenFeature([], tf.string),
    'comb/inst2': tf.FixedLenFeature([], tf.string),
    'comb/type1': tf.FixedLenFeature([], tf.string),
    'comb/type2': tf.FixedLenFeature([], tf.string),
    'comb/file1': tf.FixedLenFeature([], tf.string),
    'comb/file2': tf.FixedLenFeature([], tf.string),
    'comb/sig1': tf.FixedLenFeature([], tf.string),
    'comb/sig2': tf.FixedLenFeature([], tf.string),
    'comb/lab1': tf.FixedLenFeature([], tf.string),
    'comb/lab2': tf.FixedLenFeature([], tf.string),
    'comb/sig1_sample_delay': tf.FixedLenFeature([], tf.int64),
    'comb/sig2_sample_delay': tf.FixedLenFeature([], tf.int64),
    'comb/ref': tf.FixedLenFeature([], tf.int64),
    'comb/label': tf.FixedLenFeature([], tf.int64),
    'comb/istrain': tf.FixedLenFeature([], tf.int64),
}


def parse_features_and_decode(tf_example):
    parsed_features = tf.parse_single_example(tf_example, features)

    parsed_features['comb/sig1'] = tf.reshape(tf.decode_raw(parsed_features['comb/sig1'], tf.float32), [-1])
    parsed_features['comb/sig2'] = tf.reshape(tf.decode_raw(parsed_features['comb/sig2'], tf.float32), [-1])
    parsed_features['comb/lab1'] = tf.reshape(tf.decode_raw(parsed_features['comb/lab1'], tf.float32), [-1])
    parsed_features['comb/lab2'] = tf.reshape(tf.decode_raw(parsed_features['comb/lab2'], tf.float32), [-1])

    parsed_features['comb/label'] = tf.cast(parsed_features['comb/label'], tf.int32)
    parsed_features['comb/ref'] = tf.cast(parsed_features['comb/ref'], tf.int32)
    parsed_features['comb/class'] = tf.cast(parsed_features['comb/class'], tf.int32)
    parsed_features['comb/id'] = tf.cast(parsed_features['comb/id'], tf.int32)

    parsed_features['comb/sig1_sample_delay'] = tf.cast(parsed_features['comb/sig1_sample_delay'], tf.int32)
    parsed_features['comb/sig2_sample_delay'] = tf.cast(parsed_features['comb/sig2_sample_delay'], tf.int32)

    parsed_features['comb/istrain'] = tf.cast(parsed_features['comb/istrain'], tf.bool)

    return parsed_features


def split_train_test(parsed_features, train_rate):
    parsed_features['comb/istrain'] = tf.gather(tf.random_uniform([1], maxval=100, dtype=tf.int32) < tf.cast(train_rate * 100, tf.int32), 0)
    return parsed_features


def grab_train_examples(parsed_features):
    return parsed_features['comb/istrain']


def grab_test_examples(parsed_features):
    return ~parsed_features['comb/istrain']


def turn_into_autotest(parsed_features):
    ref = parsed_features['comb/ref']

    def delay_positive():
        parsed_features['comb/sig2'] = tf.concat([tf.zeros(tf.abs(ref)), parsed_features['comb/sig1'][:-tf.abs(ref)]], axis=0)
        parsed_features['comb/lab2'] = tf.concat([tf.zeros(tf.abs(ref)), parsed_features['comb/lab1'][:-tf.abs(ref)]], axis=0)
        return parsed_features

    def delay_negative():
        parsed_features['comb/sig2'] = tf.concat([parsed_features['comb/sig1'][tf.abs(ref):], tf.zeros(tf.abs(ref))], axis=0)
        parsed_features['comb/lab2'] = tf.concat([parsed_features['comb/lab1'][tf.abs(ref):], tf.zeros(tf.abs(ref))], axis=0)
        return parsed_features

    parsed_features = tf.cond(tf.less_equal(ref, 0), delay_positive, delay_negative)

    return parsed_features


def add_noiose(parsed_features):
    parsed_features['comb/sig1'] = parsed_features['comb/sig1'] + tf.random_normal(tf.shape(parsed_features['comb/sig1']), stddev=1)
    parsed_features['comb/sig2'] = parsed_features['comb/sig2'] + tf.random_normal(tf.shape(parsed_features['comb/sig2']), stddev=1)
    return parsed_features


def use_activation_signal_instead_of_bit_rate_signal(parsed_features):
    parsed_features['comb/sig1'] = parsed_features['comb/lab1']
    parsed_features['comb/sig2'] = parsed_features['comb/lab2']
    return parsed_features


def filter_split_dataset_from_ids(parsed_features, ids):
    id = parsed_features['comb/id']
    return tf.reduce_any(tf.equal(id, ids))


def filter_combinations_with_voice(parsed_features):
    type1 = parsed_features['comb/type1']
    type2 = parsed_features['comb/type2']
    return tf.logical_not(tf.logical_or(tf.equal(type1, 'voice'), tf.equal(type2, 'voice')))


def filter_combinations_with_different_types(parsed_features):
    type1 = parsed_features['comb/type1']
    type2 = parsed_features['comb/type2']
    return tf.equal(type1, type2)


def filter_combinations_with_rap_genre(parsed_features):
    genre = parsed_features['comb/genre']
    return tf.not_equal(genre, 'rap')


def filter_perclass(parsed_features, selected_class):
    cls = parsed_features['comb/class']
    return tf.reduce_any(tf.equal(cls, selected_class))


def filter_sigsize_leq_N(parsed_features, N):
    sig1 = parsed_features['comb/sig1']
    sig2 = parsed_features['comb/sig2']
    sigsize = tf.minimum(tf.shape(sig1)[0], tf.shape(sig2)[0])
    return tf.greater_equal(sigsize, N)


def filter_perwindow(parsed_features, N, nwin, OR):
    sig1 = parsed_features['comb/sig1']
    sig2 = parsed_features['comb/sig2']

    nw1 = 1 + OR * tf.shape(sig1)[0] // N
    nw2 = 1 + OR * tf.shape(sig2)[0] // N

    return tf.reduce_all([tf.less_equal(nwin + 2, nw1), tf.less_equal(nwin + 2, nw2)])


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

    parsed_features['comb/sig1'] = sigfil1
    parsed_features['comb/sig2'] = sigfil2

    return parsed_features


def read_audio_files(audiodir, parsed_features):
    wav1 = tf.read_file(tf.string_join([audiodir, parsed_features['comb/file1']], separator='/'))
    wav1 = tf.contrib.ffmpeg.decode_audio(wav1, file_format='wav', samples_per_second=44100, channel_count=1)
    wav1 = tf.concat([tf.zeros([parsed_features['comb/sig1_sample_delay'], 1]), wav1], axis=0)

    wav2 = tf.read_file(tf.string_join([audiodir, parsed_features['comb/file2']], separator='/'))
    wav2 = tf.contrib.ffmpeg.decode_audio(wav2, file_format='wav', samples_per_second=44100, channel_count=1)
    wav2 = tf.concat([tf.zeros([parsed_features['comb/sig2_sample_delay'], 1]), wav2], axis=0)

    parsed_features['sig1/samples'] = tf.divide(wav1, 1.5 * tf.reduce_max(wav1))
    parsed_features['sig2/samples'] = tf.divide(wav2, 1.5 * tf.reduce_max(wav2))
    return parsed_features


def replace_label_of_unselected_class(parsed_features, selected_class):
    label = tf.cast(parsed_features['comb/label'], tf.int32)
    cls = tf.cast(parsed_features['comb/class'], tf.int32)
    is_from_selected_class = tf.reduce_any(tf.equal(cls, selected_class))  # Check if combination is from class of interest
    parsed_features['comb/label'] = tf.cond(is_from_selected_class, lambda: label, lambda: 0)
    return parsed_features


def normalize_minmax(sigmat):
    max = tf.reduce_max(sigmat, axis=1, keep_dims=True)
    min = tf.reduce_min(sigmat, axis=1, keep_dims=True)
    signorm = tf.divide(tf.subtract(sigmat, min), tf.subtract(max, min))
    return signorm


def normalize_blocks(sigmat):
    mean, var = tf.nn.moments(sigmat, axes=[1], keep_dims=True)
    signorm = tf.divide(tf.subtract(sigmat, mean), tf.sqrt(var))
    return signorm


def gkernel(x, y, s):
    return tf.divide(1.0,tf.sqrt(tf.multiply(tf.multiply(2.0,np.pi),s))) * tf.exp( tf.divide(tf.multiply(-1.0,tf.pow(tf.subtract(x,y), 2.0)),tf.multiply(2.0,tf.pow(s, 2.0))) )


def get_most_correlated_blocks(sigmat1, sigmat2, nblocks):
    # cy = tf.map_fn(lambda i: gkernel(tf.expand_dims(tf.gather(sigmat1, i, axis=1), dim=1), sigmat2, 0.01), tf.range(tf.shape(sigmat1)[1]), dtype=tf.float32)
    cy = tf.map_fn(lambda i: tf.multiply(tf.expand_dims(tf.gather(sigmat1, i, axis=1), dim=1), sigmat2), tf.range(tf.shape(sigmat1)[1]), dtype=tf.float32)
    cy = tf.transpose(cy, [1, 0, 2])
    cy = tf.reduce_mean(cy, axis=[1, 2])
    # cy = tf.reduce_max(cy) - cy
    topwin = tf.nn.top_k(cy, nblocks)
    return topwin.indices


def prepare_input_with_most_correlated_blocks(parsed_features, N, nwin, OR):
    sig1 = parsed_features['comb/sig1']
    sig2 = parsed_features['comb/sig2']

    sigmat1 = tf.contrib.signal.frame(sig1, N, N // OR, pad_end=False, pad_value=0, axis=-1)
    sigmat2 = tf.contrib.signal.frame(sig2, N, N // OR, pad_end=False, pad_value=0, axis=-1)

    sigmat1 = normalize_blocks(sigmat1)
    sigmat2 = normalize_blocks(sigmat2)

    wins = get_most_correlated_blocks(sigmat1, sigmat2, nwin)
    sigmat1 = tf.gather(sigmat1, wins, axis=0)
    sigmat2 = tf.gather(sigmat2, wins, axis=0)

    sigmat1 = tf.where(tf.is_nan(sigmat1), tf.zeros_like(sigmat1), sigmat1)
    sigmat2 = tf.where(tf.is_nan(sigmat2), tf.zeros_like(sigmat2), sigmat2)

    parsed_features['example/input'] = tf.stack((sigmat1, sigmat2), axis=2)

    return parsed_features


def prepare_input_with_sequential_sampling(parsed_features, N, nwin, OR):
    sig1 = parsed_features['comb/sig1']
    sig2 = parsed_features['comb/sig2']

    sigmat1 = tf.contrib.signal.frame(sig1, N, N // OR, pad_end=False, pad_value=0, axis=-1)
    sigmat2 = tf.contrib.signal.frame(sig2, N, N // OR, pad_end=False, pad_value=0, axis=-1)

    init_id = tf.squeeze(tf.random_uniform([1,], maxval=tf.subtract(tf.shape(sigmat1)[0], nwin), dtype=tf.int32, seed=0))
    wins = tf.range(init_id, init_id + nwin)
    sigmat1 = tf.gather(sigmat1, wins, axis=0)
    sigmat2 = tf.gather(sigmat2, wins, axis=0)

    sigmat1 = normalize_blocks(sigmat1)
    sigmat2 = normalize_blocks(sigmat2)

    sigmat1 = tf.where(tf.is_nan(sigmat1), tf.zeros_like(sigmat1), sigmat1)
    sigmat2 = tf.where(tf.is_nan(sigmat2), tf.zeros_like(sigmat2), sigmat2)

    parsed_features['example/input'] = tf.stack((sigmat1, sigmat2), axis=2)
    return parsed_features


def prepare_input_with_random_sampling(parsed_features, N, nwin, OR):
    sig1 = parsed_features['comb/sig1']
    sig2 = parsed_features['comb/sig2']

    sigmat1 = tf.contrib.signal.frame(sig1, N, N // OR, pad_end=False, pad_value=0, axis=-1)
    sigmat2 = tf.contrib.signal.frame(sig2, N, N // OR, pad_end=False, pad_value=0, axis=-1)

    # Random window sampling
    wins = tf.random_uniform((nwin,), maxval=tf.minimum(tf.shape(sigmat1)[0], tf.shape(sigmat2)[0]), dtype=tf.int32, seed=0)
    sigmat1 = tf.gather(sigmat1, wins, axis=0)
    sigmat2 = tf.gather(sigmat2, wins, axis=0)

    sigmat1 = normalize_blocks(sigmat1)
    sigmat2 = normalize_blocks(sigmat2)

    sigmat1 = tf.where(tf.is_nan(sigmat1), tf.zeros_like(sigmat1), sigmat1)
    sigmat2 = tf.where(tf.is_nan(sigmat2), tf.zeros_like(sigmat2), sigmat2)

    parsed_features['example/input'] = tf.stack((sigmat1, sigmat2), axis=2)
    return parsed_features


def prepare_input_with_all_windows(parsed_features, N, OR):
    sig1 = parsed_features['comb/sig1']
    sig2 = parsed_features['comb/sig2']

    sigmat1 = tf.contrib.signal.frame(sig1, N, N // OR, pad_end=False, pad_value=0, axis=-1)
    sigmat2 = tf.contrib.signal.frame(sig2, N, N // OR, pad_end=False, pad_value=0, axis=-1)

    sigmat1 = normalize_blocks(sigmat1)
    sigmat2 = normalize_blocks(sigmat2)

    parsed_features['example/input'] = tf.stack((sigmat1, sigmat2), axis=2)
    return parsed_features


def parse_example(parsed_features):
    label = parsed_features['comb/label']
    ins = parsed_features['example/input']
    inst1 = tf.regex_replace(parsed_features['comb/inst1'], ' ', '_')
    inst2 = tf.regex_replace(parsed_features['comb/inst2'], ' ', '_')
    type1 = parsed_features['comb/type1']
    type2 = parsed_features['comb/type2']
    file1 = parsed_features['comb/file1']
    file2 = parsed_features['comb/file2']
    genre = parsed_features['comb/genre']
    id = parsed_features['comb/id']

    return ins, label, tf.string_join([type1, ' x ', type2]), tf.string_join([inst1, ' x ', inst2]), genre, id, tf.string_join([file1, ' x ', file2])


def add_defaul_dataset_pipeline(trainParams, modelParams, iterator_handle):
    with tf.name_scope('dataset') as scope:
        datasetfile = trainParams.dataset_file
        # classes = trainParams.selected_class
        # np.random.seed(3)
        # ncombs = 192401
        # eval_ids = np.random.randint(0, ncombs, [np.int32(np.floor(ncombs * 0.35))])
        # train_ids = np.setdiff1d(np.array(range(0, ncombs)), eval_ids)

        N = modelParams.N
        nwin = modelParams.nwin
        batch_size = modelParams.batch_size
        OR = modelParams.OR

        tfdataset = tf.data.TFRecordDataset(datasetfile, compression_type='GZIP', buffer_size=4096)
        tfdataset = tfdataset.map(parse_features_and_decode, num_parallel_calls=2)
        # tfdataset = tfdataset.map(turn_into_autotest, num_parallel_calls=2) #USE FOR DEBUG ONLY
        # tfdataset = tfdataset.map(use_activation_signal_instead_of_bit_rate_signal) #USE FOR DEBUG ONLY

        # tfdataset = tfdataset.filter(lambda feat: filter_perclass(feat, classes))
        # tfdataset = tfdataset.filter(filter_combinations_with_voice)
        # tfdataset = tfdataset.filter(filter_combinations_with_rap_genre)
        # tfdataset = tfdataset.filter(filter_combinations_with_different_types)
        # tfdataset = tfdataset.map(lambda feat: replace_label_of_unselected_class(feat, classes))
        # tfdataset = tfdataset.map(lambda feat: clean_from_activation_signal(feat, 0.5), num_parallel_calls=2)
        tfdataset = tfdataset.filter(lambda feat: filter_sigsize_leq_N(feat, N))
        # tfdataset = tfdataset.filter(lambda feat: filter_perwindow(feat, N, nwin, OR))
        tfdataset = tfdataset.map(lambda feat: prepare_input_with_random_sampling(feat, N, nwin, OR), num_parallel_calls=2)
        # tfdataset = tfdataset.map(lambda feat: prepare_input_with_sequential_sampling(feat, N, nwin, OR), num_parallel_calls=2)
        # tfdataset = tfdataset.map(lambda feat: prepare_input_with_all_windows(feat, N, OR))

        # train_dataset = tfdataset.filter(lambda feat: filter_split_dataset_from_ids(feat, train_ids)).map(parse_example, num_parallel_calls=6)
        # test_dataset = tfdataset.filter(lambda feat: filter_split_dataset_from_ids(feat, eval_ids)).map(parse_example, num_parallel_calls=6)

        # tfdataset = tfdataset.map(lambda feat: split_train_test(feat, trainParams.train_test_rate), num_parallel_calls=6)
        train_dataset = tfdataset.filter(grab_train_examples).map(parse_example, num_parallel_calls=2)
        test_dataset = tfdataset.filter(grab_test_examples).map(parse_example, num_parallel_calls=2)

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


def add_augmented_dataset_pipeline(trainParams, modelParams, iterator_handle):
    with tf.name_scope('dataset') as scope:
        datasetfile = trainParams.dataset_file

        N = modelParams.N
        nwin = modelParams.nwin
        batch_size = modelParams.batch_size
        OR = modelParams.OR

        tfdataset = tf.data.TFRecordDataset(datasetfile, compression_type='GZIP', buffer_size=4096)
        tfdataset = tfdataset.map(parse_features_and_decode, num_parallel_calls=2)

        tfdataset = tfdataset.filter(filter_combinations_with_voice)
        tfdataset = tfdataset.map(lambda feat: clean_from_activation_signal(feat, 0.5), num_parallel_calls=2)
        tfdataset = tfdataset.filter(lambda feat: filter_sigsize_leq_N(feat, N))

        real_tr_dataset = tfdataset.filter(grab_train_examples)
        auto_tr_dataset = real_tr_dataset.map(turn_into_autotest, num_parallel_calls=4).map(add_noiose, num_parallel_calls=4)
        zip_dataset = tf.data.Dataset.zip((auto_tr_dataset, real_tr_dataset))

        train_dataset = zip_dataset.flat_map(lambda x0, x1: tf.data.Dataset.from_tensors(x0).concatenate(tf.data.Dataset.from_tensors(x1)))
        test_dataset = tfdataset.filter(grab_test_examples)

        train_dataset = train_dataset.filter(lambda feat: filter_sigsize_leq_N(feat, N))
        test_dataset = test_dataset.filter(lambda feat: filter_sigsize_leq_N(feat, N))

        train_dataset = train_dataset.map(lambda feat: prepare_input_with_random_sampling(feat, N, nwin, OR), num_parallel_calls=2)
        test_dataset = test_dataset.map(lambda feat: prepare_input_with_random_sampling(feat, N, nwin, OR), num_parallel_calls=2)

        train_dataset = train_dataset.map(parse_example, num_parallel_calls=2)
        test_dataset = test_dataset.map(parse_example, num_parallel_calls=2)

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