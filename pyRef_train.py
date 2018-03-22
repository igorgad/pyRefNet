
import os
import time
import tensorflow as tf
import numpy as np
import tfplot
import models.rkhsModel as model #Chose model here!!!

from prettytable import PrettyTable

features = {
        'comb/id': tf.FixedLenFeature([], tf.int64),
        'comb/class': tf.FixedLenFeature([], tf.int64),
        'comb/inst1': tf.FixedLenFeature([], tf.string),
        'comb/inst2': tf.FixedLenFeature([], tf.string),
        'comb/type1': tf.FixedLenFeature([], tf.string),
        'comb/type2': tf.FixedLenFeature([], tf.string),
        'comb/sig1' : tf.FixedLenFeature([], tf.string),
        'comb/sig2' : tf.FixedLenFeature([], tf.string),
        'comb/lab1' : tf.FixedLenFeature([], tf.string),
        'comb/lab2' : tf.FixedLenFeature([], tf.string),
        'comb/ref'  : tf.FixedLenFeature([], tf.int64),
        'comb/label': tf.FixedLenFeature([], tf.int64),
    }


def filter_split_examples(tf_example, ids):
    parsed_features = tf.parse_single_example(tf_example, features)
    id = parsed_features['comb/id']
    return tf.reduce_any(tf.equal(id,ids))


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


def slice_examples(tf_example, N, nwin, OR, selected_class):
    parsed_features = tf.parse_single_example(tf_example, features)

    label = tf.cast(parsed_features['comb/label'], tf.int32)
    type1 = tf.cast(parsed_features['comb/type1'], tf.string)
    type2 = tf.cast(parsed_features['comb/type2'], tf.string)
    cls = tf.cast(parsed_features['comb/class'], tf.int32)

    sig1 = tf.reshape(tf.decode_raw(parsed_features['comb/sig1'], tf.float32), [-1])
    sig2 = tf.reshape(tf.decode_raw(parsed_features['comb/sig2'], tf.float32), [-1])
    lab1 = tf.reshape(tf.decode_raw(parsed_features['comb/lab1'], tf.float32), [-1])
    lab2 = tf.reshape(tf.decode_raw(parsed_features['comb/lab2'], tf.float32), [-1])

    sigmat1 = tf.contrib.signal.frame(sig1, N, N // OR, pad_end=True, pad_value=0, axis=-1)
    sigmat2 = tf.contrib.signal.frame(sig2, N, N // OR, pad_end=True, pad_value=0, axis=-1)
    labmat1 = tf.contrib.signal.frame(lab1, N, N // OR, pad_end=True, pad_value=0, axis=-1)
    labmat2 = tf.contrib.signal.frame(lab2, N, N // OR, pad_end=True, pad_value=0, axis=-1)

    # Perform random sample of windows
    wins = tf.random_uniform((nwin,), maxval=tf.shape(sigmat1)[0], dtype=tf.int32)

    sigmat1 = tf.gather(sigmat1, wins, axis=0)
    sigmat2 = tf.gather(sigmat2, wins, axis=0)
    labmean1 = tf.reduce_mean(tf.gather(labmat1, wins, axis=0), axis=1)
    labmean2 = tf.reduce_mean(tf.gather(labmat2, wins, axis=0), axis=1)

    ins = tf.stack((sigmat1, sigmat2), axis=2)

    at_least_one_window_active = tf.reduce_any(tf.logical_and(labmean1 > 0.5, labmean2 > 0.5))  # Check if there is at least one window pair (x_w and y_w) with the instruments active
    is_from_selected_class = tf.reduce_any(tf.equal(cls, selected_class))  # Check if combination is from combination class of interest

    label = tf.cond(tf.logical_and(at_least_one_window_active, is_from_selected_class), lambda: label, lambda: 0)

    return ins, label, tf.string_join([type1, ' x ', type2])


def add_data_pipeline(batch_size, train_ids, eval_ids, handle, datasetfile, classes):

    with tf.name_scope('dataset') as scope:
        tfdataset = tf.data.TFRecordDataset(datasetfile)
        # tfdataset = tfdataset.filter(lambda ex: filter_perclass_examples(ex, classes))
        tfdataset = tfdataset.filter(lambda ex: filter_perwindow_examples(ex, model.N, model.nwin, model.OR))

        train_dataset = tfdataset.filter(lambda ex: filter_split_examples(ex, train_ids))
        test_dataset  = tfdataset.filter(lambda ex: filter_split_examples(ex, eval_ids))

        train_dataset = train_dataset.map(lambda ex: slice_examples(ex, model.N, model.nwin, model.OR, classes), num_parallel_calls=4)
        test_dataset  = test_dataset.map(lambda ex: slice_examples(ex, model.N, model.nwin, model.OR, classes), num_parallel_calls=4)

        train_dataset = train_dataset.shuffle(2048, reshuffle_each_iteration=True)
        test_dataset = test_dataset.shuffle(2048, reshuffle_each_iteration=True)

        train_dataset = train_dataset.prefetch(buffer_size=512)
        test_dataset  = test_dataset.prefetch(buffer_size=512)

        train_dataset = train_dataset.batch(batch_size)
        test_dataset  = test_dataset.batch(batch_size)

        iterator = tf.data.Iterator.from_string_handle(
            handle, train_dataset.output_types, train_dataset.output_shapes)
        next_element = iterator.get_next()

        train_iterator = train_dataset.make_initializable_iterator()
        test_iterator = test_dataset.make_initializable_iterator()

    return next_element, train_iterator, test_iterator


def add_summaries(loss, eval1, eval5):
    with tf.name_scope('summ') as scope:
        avg_loss, avg_loss_op = tf.contrib.metrics.streaming_mean(loss)
        avg_top1, avg_top1_op = tf.contrib.metrics.streaming_mean(eval1)
        avg_top5, avg_top5_op = tf.contrib.metrics.streaming_mean(eval5)

        reset_op = tf.local_variables_initializer()

        tf.summary.scalar('avg_loss', avg_loss)
        tf.summary.scalar('avg_top1', avg_top1)
        tf.summary.scalar('avg_top5', avg_top5)

    return avg_loss_op, avg_top1_op, avg_top5_op, reset_op


def add_tables(correct1, correct5, typecombs):
    top1_typecomb_table = tf.contrib.lookup.MutableHashTable(tf.string, tf.float32, 0)
    top5_typecomb_table = tf.contrib.lookup.MutableHashTable(tf.string, tf.float32, 0)

    update_top1_typecomb_table = top1_typecomb_table.insert(typecombs, tf.divide(tf.add(top1_typecomb_table.lookup(typecombs), tf.cast(correct1, tf.float32)), 2))
    update_top5_typecomb_table = top5_typecomb_table.insert(typecombs, tf.divide(tf.add(top5_typecomb_table.lookup(typecombs), tf.cast(correct5, tf.float32)), 2))

    export_top1_typecomb_table = top1_typecomb_table.export()
    export_top5_typecomb_table = top5_typecomb_table.export()

    update_ops = [update_top1_typecomb_table, update_top5_typecomb_table]
    export_ops = [export_top1_typecomb_table, export_top5_typecomb_table]

    return update_ops, export_ops


def create_stats_figure(labels, probs):
    fig, ax = tfplot.subplots()
    ax.bar(np.arange(probs.size), probs, 0.35)
    ax.set_xticks(np.arange(labels.size))
    ax.set_xticklabels(labels, rotation=-90, ha='center')
    fig.set_tight_layout(True)
    return fig

# TODO: ideally these tables should be cleaned after each epoch
def add_comb_stats(correct1, correct5, typecombs, table_selector):
    with tf.name_scope('combStats') as scope:
        train_update_ops, train_export_ops = add_tables(correct1, correct5, typecombs)
        test_update_ops, test_export_ops = add_tables(correct1, correct5, typecombs)

        update_stats, export_stats = tf.cond(tf.equal(table_selector, 0), lambda: [train_update_ops, train_export_ops], lambda: [test_update_ops, test_export_ops])

        top1_plot_op = tfplot.plot(create_stats_figure, [export_stats[0][0], export_stats[0][1]])
        top5_plot_op = tfplot.plot(create_stats_figure, [export_stats[1][0], export_stats[1][1]])

        tf.summary.image('top1_typecomb', tf.expand_dims(top1_plot_op, 0), max_outputs=1)
        tf.summary.image('top5_typecomb', tf.expand_dims(top5_plot_op, 0), max_outputs=1)

    return update_stats, export_stats

def add_hyperparameters_textsum(trainParams):
    table = PrettyTable(['hparams', 'value'])
    hpd = dict(model.hptext)
    hpd.update(trainParams.hptext)

    for key, val in hpd.items():
        table.add_row([key, val])

    print (table)
    return tf.summary.text('hyperparameters', tf.convert_to_tensor(table.get_html_string(format=True)))

def run_training(trainParams):

    with tf.Graph().as_default():

        keepp_pl = tf.placeholder(tf.float32)
        queue_selector = tf.placeholder(tf.int32)
        dataset_handle = tf.placeholder(tf.string, shape=[])

        examples, train_iterator, test_iterator = add_data_pipeline(model.batch_size, trainParams.trainIds, trainParams.evalIds, dataset_handle,
                                                                    trainParams.datasetfile, trainParams.combSets)
        ins = examples[0]
        lbs = examples[1]
        typecombs = examples[2]

        logits = model.inference(ins, keepp_pl)
        loss   = model.loss(logits, lbs)
        train_op = model.training(loss)
        eval_top1, eval_top5, correct1, correct5 = model.evaluation(logits, lbs)

        avg_loss_op, avg_top1_op, avg_top5_op, reset_op = add_summaries(loss, eval_top1, eval_top5)
        update_stats, export_stats = add_comb_stats(correct1, correct5, typecombs, queue_selector)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        train_writer = tf.summary.FileWriter(trainParams.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(trainParams.log_dir + '/test')

        training_handle = sess.run(train_iterator.string_handle())
        testing_handle = sess.run(test_iterator.string_handle())

        hparams_op = add_hyperparameters_textsum(trainParams)
        _, hp_str = sess.run([init, hparams_op])
        train_writer.add_summary(hp_str, 0)
        train_writer.flush()

        train_btch = 0
        sum_step = 0
        eval_btch = 0

        try:

            # Start the training loop.
            for epoch in range(trainParams.numEpochs):

                # Train
                sess.run(reset_op)
                sess.run(train_iterator.initializer)

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, loss_value, top1_value, top5_value, __ = sess.run([train_op, avg_loss_op, avg_top1_op, avg_top5_op, update_stats],
                                                                     feed_dict={dataset_handle: training_handle, queue_selector: 0, keepp_pl: model.kp}, options=run_options,
                                                                     run_metadata=run_metadata)

                train_writer.add_run_metadata(run_metadata, 'stats_epoch %d' % epoch)
                train_writer.flush()

                while True:
                    try:
                        start_time = time.time()

                        # Log training runtime statistics
                        if np.mod(train_btch + 1, trainParams.sum_interval) == 0:

                            summary_str, _, loss_value, top1_value, top5_value, __ = sess.run([summary, train_op, avg_loss_op, avg_top1_op, avg_top5_op, update_stats],
                                                                                          feed_dict={dataset_handle: training_handle, queue_selector: 0, keepp_pl: model.kp})

                            train_writer.add_summary(summary_str, sum_step )
                            train_writer.flush()

                            sum_step += 1
                            sess.run([reset_op])
                        else:
                            _, loss_value, top1_value, top5_value, __ = sess.run([train_op, avg_loss_op, avg_top1_op, avg_top5_op, update_stats],
                                                                              feed_dict={dataset_handle: training_handle, queue_selector: 0, keepp_pl: model.kp})

                        duration = time.time() - start_time
                        train_btch += 1

                        print ('%s_run_%d: TRAIN epoch %d, step %d. %0.2f hz loss: %0.04f top1 %0.04f top5 %0.04f' %
                               (trainParams.runName, trainParams.n + 1, epoch, train_btch, model.batch_size/duration, loss_value, top1_value, top5_value) )

                    except tf.errors.OutOfRangeError:
                        break

                # Evaluate
                sess.run([reset_op])
                sess.run(test_iterator.initializer)
                while True:
                    try:
                        start_time = time.time()

                        loss_value, top1_value, top5_value, _ = sess.run([avg_loss_op, avg_top1_op, avg_top5_op, update_stats],
                                                                                      feed_dict={dataset_handle: testing_handle, queue_selector: 1, keepp_pl: 1})

                        duration = time.time() - start_time
                        print('%s_run_%d: TEST epoch %d,step %d. %0.2f hz. loss: %0.04f. top1 %0.04f. top5 %0.04f' %
                              (trainParams.runName, trainParams.n + 1, epoch, eval_btch, model.batch_size / duration, loss_value, top1_value, top5_value))

                        eval_btch += 1

                    except tf.errors.OutOfRangeError:
                        break

                sess.run(test_iterator.initializer)
                summary_str, loss_value, top1_value, top5_value, _ = sess.run([summary, avg_loss_op, avg_top1_op, avg_top5_op, update_stats],
                                                                              feed_dict={dataset_handle: testing_handle, queue_selector: 1, keepp_pl: 1})
                test_writer.add_summary(summary_str, sum_step - 1 )
                test_writer.flush()

                # Save a checkpoint
                if (epoch + 1) % 10 == 0 or (epoch + 1) == trainParams.numEpochs:
                    checkpoint_file = os.path.join(trainParams.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=epoch)

        finally:
            print ('\n\ncleaning...')
            train_stats = sess.run(export_stats, {queue_selector: 0})
            test_stats = sess.run(export_stats, {queue_selector: 1})
            sess.close()
            np.save(trainParams.log_dir + '/combstats', [train_stats, test_stats])
            return [train_stats, test_stats]


def runExperiment(trainParams):
    if tf.gfile.Exists(trainParams.log_dir):
        tf.gfile.DeleteRecursively(trainParams.log_dir)

    tf.gfile.MakeDirs(trainParams.log_dir)

    return run_training(trainParams)

