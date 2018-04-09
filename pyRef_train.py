
import os
import time
import tensorflow as tf
import numpy as np
import tfplot
import models.rkhsModel as model #Chose model here!!!
import dataset_interface

from prettytable import PrettyTable

def add_summaries(loss, eval1, eval5):
    with tf.name_scope('summ') as scope:
        with tf.device('/cpu:0'):
            avg_loss, avg_loss_op = tf.contrib.metrics.streaming_mean(loss)
            avg_top1, avg_top1_op = tf.contrib.metrics.streaming_mean(eval1)
            avg_top5, avg_top5_op = tf.contrib.metrics.streaming_mean(eval5)

            reset_op = tf.local_variables_initializer()

            tf.summary.scalar('avg_loss', avg_loss)
            tf.summary.scalar('avg_top1', avg_top1)
            tf.summary.scalar('avg_top5', avg_top5)

    return avg_loss_op, avg_top1_op, avg_top5_op, reset_op


def sort_labels(labels):
    labels = np.array([os.fsdecode(i) for i in labels])
    vcc = [[i.replace(' ', ''), j.replace(' ', '')] for i, j in [lb.split('x') for lb in labels]]
    slb = [sorted(i) for i in vcc]
    sorted_labels = np.array([''.join([c[0], ' x ', c[1]]) for c in slb], np.chararray)
    return sorted_labels


def add_tables(correct1, correct5, typecombs):
    top1_typecomb_table = tf.contrib.lookup.MutableHashTable(tf.string, tf.float32, 0)
    top5_typecomb_table = tf.contrib.lookup.MutableHashTable(tf.string, tf.float32, 0)

    sorted_typecombs = tf.py_func(sort_labels, [typecombs], tf.string)

    update_top1_typecomb_table = top1_typecomb_table.insert(sorted_typecombs, tf.divide(tf.add(top1_typecomb_table.lookup(sorted_typecombs), tf.cast(correct1, tf.float32)), 2))
    update_top5_typecomb_table = top5_typecomb_table.insert(sorted_typecombs, tf.divide(tf.add(top5_typecomb_table.lookup(sorted_typecombs), tf.cast(correct5, tf.float32)), 2))

    export_top1_typecomb_table = top1_typecomb_table.export()
    export_top5_typecomb_table = top5_typecomb_table.export()

    update_ops = [update_top1_typecomb_table, update_top5_typecomb_table]
    export_ops = [export_top1_typecomb_table, export_top5_typecomb_table]

    return update_ops, export_ops


def create_stats_figure(labels, probs):
    labels = np.array([os.fsdecode(i) for i in labels])
    fig, ax = tfplot.subplots()
    ax.bar(np.arange(probs.size), probs, 0.35)
    ax.set_xticks(np.arange(labels.size))
    ax.set_xticklabels(labels, rotation=+90, ha='center')
    fig.set_tight_layout(True)
    return fig

# TODO: ideally these tables should be cleaned after each epoch
def add_comb_stats(correct1, correct5, typecombs, table_selector):
    with tf.name_scope('combStats') as scope:
        with tf.device('/cpu:0'):
            train_update_ops, train_export_ops = add_tables(correct1, correct5, typecombs)
            test_update_ops, test_export_ops = add_tables(correct1, correct5, typecombs)

            update_stats, export_stats = tf.cond(tf.equal(table_selector, 0), lambda: [train_update_ops, train_export_ops], lambda: [test_update_ops, test_export_ops])

            top1_plot_op = tfplot.plot(create_stats_figure, [export_stats[0][0], export_stats[0][1]])
            top5_plot_op = tfplot.plot(create_stats_figure, [export_stats[1][0], export_stats[1][1]])

            tf.summary.image('top1_typecomb', tf.expand_dims(top1_plot_op, 0), max_outputs=1)
            tf.summary.image('top5_typecomb', tf.expand_dims(top5_plot_op, 0), max_outputs=1)

    return update_stats, export_stats


def create_confusion_image(confusion_data):
    fig, ax = tfplot.subplots()
    ax.imshow(confusion_data)
    fig.set_size_inches(8, 8)
    return fig


def add_confusion_matrix(logits, labels):
    with tf.name_scope('confusionMatrix') as scope:
        with tf.device('/cpu:0'):
            predictions = tf.argmax(logits,1)
            confusion = tf.confusion_matrix(labels=labels, predictions=predictions)

            confusion_img = tfplot.plot(create_confusion_image, [confusion])
            tf.summary.image('confusion_matrix', tf.expand_dims(confusion_img,0), max_outputs=1)


def add_hyperparameters_textsum(trainParams):
    table = PrettyTable(['hparams', 'value'])
    hpd = dict(model.hptext)
    hpd.update(trainParams.hptext)

    for key, val in hpd.items():
        table.add_row([key, val])

    print (table)
    return tf.summary.text('hyperparameters', tf.convert_to_tensor(table.get_html_string(format=True)))


def start_training(trainParams):

    with tf.Graph().as_default():

        keepp_pl = tf.placeholder(tf.float32)
        queue_selector = tf.placeholder(tf.int32)
        dataset_handle = tf.placeholder(tf.string, shape=[])

        global_step = tf.Variable(0, name='global_step', trainable=False)

        examples, train_iterator, test_iterator = dataset_interface.add_defaul_dataset_pipeline(trainParams, model, dataset_handle)

        ins = examples[0]
        lbs = examples[1]
        typecombs = examples[2]

        logits = model.inference(ins, keepp_pl)
        loss   = model.loss(logits, lbs)
        train_op = model.training(loss, global_step)
        eval_top1, eval_top5, correct1, correct5 = model.evaluation(logits, lbs)

        avg_loss_op, avg_top1_op, avg_top5_op, reset_op = add_summaries(loss, eval_top1, eval_top5)
        update_stats, export_stats = add_comb_stats(correct1, correct5, typecombs, queue_selector)
        add_confusion_matrix(logits, lbs)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        sess = tf.Session(config=config)
        train_writer = tf.summary.FileWriter(trainParams.log_path_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(trainParams.log_path_dir + '/test')

        training_handle = sess.run(train_iterator.string_handle())
        testing_handle = sess.run(test_iterator.string_handle())

        hparams_op = add_hyperparameters_textsum(trainParams)

        # Initialize or load graph from checkpoint
        if not trainParams.restore_from_dir:
            tf.gfile.MakeDirs(trainParams.log_path_dir)
            _, hp_str = sess.run([init, hparams_op])
            train_writer.add_summary(hp_str, 0)
            train_writer.flush()
        else:
            ckpt = tf.train.get_checkpoint_state(trainParams.restore_from_dir[0])
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            print ('loaded graph from dir %s' % trainParams.restore_from_dir[0])

        gstep = 0
        try:
            # Start the training loop.
            while gstep < trainParams.num_steps:
                # Train
                sess.run(reset_op)
                sess.run(train_iterator.initializer)

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                _, loss_value, top1_value, top5_value, __, gstep = sess.run([train_op, avg_loss_op, avg_top1_op, avg_top5_op, update_stats, global_step],
                                                                     feed_dict={dataset_handle: training_handle, queue_selector: 0, keepp_pl: model.kp}, options=run_options,
                                                                     run_metadata=run_metadata)

                train_writer.add_run_metadata(run_metadata, 'stats_epoch %d' % gstep)
                train_writer.flush()

                print('%s: TRAIN step %d. %0.2f hz loss: %0.04f top1 %0.04f top5 %0.04f' %
                      (trainParams.run_name, gstep, 0.0, loss_value, top1_value, top5_value))

                while True:
                    try:
                        start_time = time.time()

                        # Log training runtime statistics
                        if np.mod(gstep + 1, trainParams.summary_interval) == 0:
                            summary_str, _, loss_value, top1_value, top5_value, __, gstep = sess.run([summary, train_op, avg_loss_op, avg_top1_op, avg_top5_op, update_stats, global_step],
                                                                                                     feed_dict={dataset_handle: training_handle, queue_selector: 0, keepp_pl: model.kp})

                            train_writer.add_summary(summary_str, gstep )
                            train_writer.flush()

                            sess.run([reset_op])
                        else:
                            _, loss_value, top1_value, top5_value, __, gstep = sess.run([train_op, avg_loss_op, avg_top1_op, avg_top5_op, update_stats, global_step],
                                                                              feed_dict={dataset_handle: training_handle, queue_selector: 0, keepp_pl: model.kp})

                        duration = time.time() - start_time

                        print ('%s: TRAIN step %d. %0.2f hz loss: %0.04f top1 %0.04f top5 %0.04f' %
                               (trainParams.run_name, gstep, model.batch_size/duration, loss_value, top1_value, top5_value) )

                    except tf.errors.OutOfRangeError:
                        break

                # Evaluate
                sess.run([reset_op])
                sess.run(test_iterator.initializer)
                while True:
                    try:
                        start_time = time.time()

                        loss_value, top1_value, top5_value, _, gstep = sess.run([avg_loss_op, avg_top1_op, avg_top5_op, update_stats, global_step],
                                                                                      feed_dict={dataset_handle: testing_handle, queue_selector: 1, keepp_pl: 1})

                        duration = time.time() - start_time
                        print('%s: TEST step %d. %0.2f hz. loss: %0.04f. top1 %0.04f. top5 %0.04f' %
                              (trainParams.run_name, gstep, model.batch_size / duration, loss_value, top1_value, top5_value))

                    except tf.errors.OutOfRangeError:
                        break

                sess.run(test_iterator.initializer)
                summary_str, loss_value, top1_value, top5_value, _ , gstep = sess.run([summary, avg_loss_op, avg_top1_op, avg_top5_op, update_stats, global_step],
                                                                              feed_dict={dataset_handle: testing_handle, queue_selector: 1, keepp_pl: 1})
                test_writer.add_summary(summary_str, gstep)
                test_writer.flush()

                # Save a checkpoint
                checkpoint_file = os.path.join(trainParams.log_path_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=gstep)

        except Exception as e:
            print('finishing...' + str(e))
            sess.close()
            return 0
