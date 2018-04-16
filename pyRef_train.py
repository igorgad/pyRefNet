
import os
import time
import tensorflow as tf
import numpy as np
import tfplot
import models.rkhsModel as model #Chose model here!!!
import dataset_interface
import stats
from tensorflow.python import debug as tf_debug

from prettytable import PrettyTable

def add_hyperparameters_textsum(trainParams):
    table = PrettyTable(['hparams', 'value'])
    hpd = dict(model.hptext)
    hpd.update(trainParams.hptext)

    for key, val in hpd.items():
        table.add_row([key, val])

    print (table)
    return tf.summary.text('hyperparameters', tf.convert_to_tensor(table.get_html_string(format=True)))


def estimator_model_fn(features, labels, mode, params):

    ins = features['ins']
    lbs = labels
    typecombs = features['typecomb']
    genres = features['genre']

    keep_prob = 1.0
    train_test_selector = 1
    if tf.estimator.ModeKeys.TRAIN:
        keep_prob = params['keep_prob']
        train_test_selector = 0


    logits = model.inference(ins, keep_prob)
    loss = model.loss(logits, lbs)
    optimizer = tf.train.AdamOptimizer(model.lr)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
    eval_top1, eval_top5, correct1, correct5 = model.evaluation(logits, lbs)
    update_comb_stats, export_comb_stats = stats.add_comb_stats(correct1, correct5, typecombs, train_test_selector)
    update_genre_stats = stats.add_genre_stats(correct1, correct5, genres, train_test_selector)
    stats.add_confusion_matrix(logits, lbs)
    predictions = {'classes': tf.argmax(logits, axis=1), 'probabilities': tf.nn.softmax(logits)}

    acc = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    eval_metrics = {'accuracy': acc } #, 'top1_accuracy': avg_top1_op, 'top5_accuracy': avg_top5_op } #, 'combStats': update_comb_stats, 'genreStats': update_genre_stats}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions,
                                          export_outputs={ 'classify': tf.estimator.export.PredictOutput(predictions) })

    if tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op, eval_metric_ops=eval_metrics)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metrics)


def start_training_with_estimator_api(trainParams):

    pyref_estimator = tf.estimator.Estimator(
        model_fn=estimator_model_fn,
        model_dir=trainParams.log_path_dir,
        params={
            'keep_prob': model.kp,
        })

    pyref_estimator.train(input_fn=lambda: dataset_interface.get_train_input_fn(trainParams, model), steps=100000)


def start_training(trainParams):

    with tf.Graph().as_default():

        keepp_pl = tf.placeholder(tf.float32)
        train_test_selector = tf.placeholder(tf.int32)
        dataset_handle = tf.placeholder(tf.string, shape=[])

        global_step = tf.Variable(0, name='global_step', trainable=False)

        examples, train_iterator, test_iterator = dataset_interface.add_defaul_dataset_pipeline(trainParams, model, dataset_handle)

        ins = examples[0]
        lbs = examples[1]['ref']
        typecombs = examples[1]['typecomb']
        genres = examples[1]['genre']

        logits = model.inference(ins, keepp_pl)
        loss   = model.loss(logits, lbs)
        train_op = model.training(loss, global_step)
        eval_top1, eval_top5, correct1, correct5 = model.evaluation(logits, lbs)

        avg_loss_op, avg_top1_op, avg_top5_op, reset_op = stats.add_summaries(loss, eval_top1, eval_top5)
        update_comb_stats = stats.add_comb_stats(correct1, correct5, typecombs, train_test_selector)
        update_genre_stats = stats.add_genre_stats(correct1, correct5, genres, train_test_selector)
        stats.add_confusion_matrix(logits, lbs)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        sess = tf.Session(config=config)
        if trainParams.debug:
            sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')

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

                if trainParams.trace:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    _, loss_value, top1_value, top5_value, __, ___, gstep = sess.run([train_op, avg_loss_op, avg_top1_op, avg_top5_op, update_comb_stats, update_genre_stats, global_step],
                                                                         feed_dict={dataset_handle: training_handle, train_test_selector: 0, keepp_pl: model.kp}, options=run_options,
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
                            summary_str, _, loss_value, top1_value, top5_value, __, ___, gstep = sess.run([summary, train_op, avg_loss_op, avg_top1_op, avg_top5_op,
                                                                                                           update_comb_stats, update_genre_stats, global_step],
                                                                                                     feed_dict={dataset_handle: training_handle, train_test_selector: 0, keepp_pl: model.kp})

                            train_writer.add_summary(summary_str, gstep )
                            train_writer.flush()

                            sess.run([reset_op])
                        else:
                            _, loss_value, top1_value, top5_value, __, ___, gstep = sess.run([train_op, avg_loss_op, avg_top1_op, avg_top5_op,
                                                                                              update_comb_stats, update_genre_stats, global_step],
                                                                              feed_dict={dataset_handle: training_handle, train_test_selector: 0, keepp_pl: model.kp})

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

                        loss_value, top1_value, top5_value, _, ___, gstep = sess.run([avg_loss_op, avg_top1_op, avg_top5_op, update_comb_stats, update_genre_stats, global_step],
                                                                                      feed_dict={dataset_handle: testing_handle, train_test_selector: 1, keepp_pl: 1})

                        duration = time.time() - start_time
                        print('%s: TEST step %d. %0.2f hz. loss: %0.04f. top1 %0.04f. top5 %0.04f' %
                              (trainParams.run_name, gstep, model.batch_size / duration, loss_value, top1_value, top5_value))

                    except tf.errors.OutOfRangeError:
                        break

                sess.run(test_iterator.initializer)
                summary_str, loss_value, top1_value, top5_value, _ , ___, gstep = sess.run([summary, avg_loss_op, avg_top1_op, avg_top5_op,
                                                                                            update_comb_stats, update_genre_stats, global_step],
                                                                              feed_dict={dataset_handle: testing_handle, train_test_selector: 1, keepp_pl: 1})
                test_writer.add_summary(summary_str, gstep)
                test_writer.flush()

                # Save a checkpoint
                checkpoint_file = os.path.join(trainParams.log_path_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=gstep)

        except Exception as e:
            print('finishing...' + str(e))
            sess.close()
            return 0
