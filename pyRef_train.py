
import os
import time

import tensorflow as tf
import numpy as np

import models.rkhsModel as model


mmap = None # FIX
def np_get_batch(batch_ids, batch_size):
    bcan = np.random.choice(batch_ids, batch_size)
    ins_feed = np.transpose(np.array([mmap[itm][0] for itm in bcan]), [0,2,3,1]) # NEWS: Set [bsize,nwin,N,nsigs]
    lbs_feed = np.array([mmap[itm][1] for itm in bcan]) + 80

    return ins_feed.astype(np.float32), lbs_feed.astype(np.int32)


def tf_get_batch(batch_ids, batch_size):
    ins_ex, lbs_ex = tf.py_func(np_get_batch, [batch_ids, batch_size], [tf.float32, tf.int32])

    return ins_ex, lbs_ex

def add_queues(batch_size, train_ids, eval_ids, selector_pl):
    with tf.name_scope('queues') as scope:
        q_train = tf.FIFOQueue(3, dtypes=[tf.float32, tf.int32], shapes=[[batch_size, model.nwin, model.N, model.nsigs], [batch_size]])
        q_eval = tf.FIFOQueue(3, dtypes=[tf.float32, tf.int32], shapes=[[batch_size, model.nwin, model.N, model.nsigs], [batch_size]])

        q_train_op = q_train.enqueue(tf_get_batch(train_ids, batch_size))
        q_eval_op = q_train.enqueue(tf_get_batch(eval_ids, batch_size))

        qr_train = tf.train.QueueRunner(q_train, [q_train_op] * 3)
        qr_eval = tf.train.QueueRunner(q_eval, [q_eval_op] * 3)

        tf.train.add_queue_runner(qr_train)
        tf.train.add_queue_runner(qr_eval)

        q = tf.QueueBase.from_list(selector_pl, [q_train, q_eval])

        ins, lbs = q.dequeue()

    return ins, lbs,

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


def run_training(trainParams):

    with tf.Graph().as_default():

        keepp_pl = tf.placeholder(tf.float32)
        queue_selector = tf.placeholder(tf.int32)

        ins, lbs = add_queues(trainParams.batch_size, trainParams.trainIds, trainParams.evalIds, queue_selector)

        logits = model.inference(ins, keepp_pl)
        loss = model.loss(logits, lbs)
        train_op = model.training(loss, trainParams.lr, trainParams.momentum)
        eval_top1, eval_top5 = model.evaluation(logits, lbs)
        avg_loss_op, avg_top1_op, avg_top5_op, reset_op = add_summaries(loss, eval_top1, eval_top5)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        train_writer = tf.summary.FileWriter(trainParams.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(trainParams.log_dir + '/test')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run([init, reset_op])

        ntrain = trainParams.trainIds.size
        neval  = trainParams.evalIds.size

        nsteps_train = np.int32(np.floor(ntrain / trainParams.batch_size))
        nsteps_eval = np.int32(np.floor(neval / trainParams.batch_size))


        # Start the training loop.
        for epoch in range(trainParams.numEpochs):
            # Train
            for bthc in range(nsteps_train):
                sum_step = trainParams.sumPerEpoch * epoch + bthc // np.ceil(nsteps_train / trainParams.sumPerEpoch)

                keep_prob = 0.7 #Dynamic control of dropout rate

                start_time = time.time()

                # Log training runtime statistics. One per epoch (last step)
                if np.mod(bthc, np.ceil(nsteps_train / trainParams.sumPerEpoch)) == 1:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary_str, _, loss_value, top1_value, top5_value = sess.run([summary, train_op, avg_loss_op, avg_top1_op, avg_top5_op],
                                                                                  feed_dict={queue_selector: 0, keepp_pl: keep_prob}, options=run_options, run_metadata=run_metadata)

                    train_writer.add_run_metadata(run_metadata, 'epoch %d' % sum_step )
                    train_writer.add_summary(summary_str, sum_step )
                    train_writer.flush()

                    # sess.run([reset_op])
                else:
                     _, loss_value, top1_value, top5_value = sess.run([train_op, avg_loss_op, avg_top1_op, avg_top5_op],
                                                                      feed_dict={queue_selector: 0, keepp_pl: keep_prob})

                duration = time.time() - start_time
                print ('%s_run_%d: TRAIN epoch %d, %d/%d. %0.2f hz loss: %0.04f top1 %0.04f top5 %0.04f' %
                       (trainParams.runName, trainParams.n + 1, epoch, bthc, nsteps_train - 1, 1.0/duration, loss_value, top1_value, top5_value) )

            # Evaluate
            for bthc in range(nsteps_eval):
                batch_ids = np.random.choice(trainParams.evalIds, trainParams.batch_size)
                sum_step = trainParams.sumPerEpoch * epoch + bthc // np.ceil(nsteps_eval / trainParams.sumPerEpoch)

                keep_prob = 1.0  # Dynamic control of dropout rate

                start_time = time.time()

                # Log testing runtime statistics. One per epoch (last step)
                if np.mod(bthc, np.ceil(nsteps_eval / trainParams.sumPerEpoch)) == 1:
                    summary_str, loss_value, top1_value, top5_value = sess.run([summary, avg_loss_op, avg_top1_op, avg_top5_op],
                                                                               feed_dict={queue_selector: 1, keepp_pl: keep_prob})
                    test_writer.add_summary(summary_str, sum_step )
                    test_writer.flush()

                    # sess.run([reset_op])
                else:
                    loss_value, top1_value, top5_value = sess.run([avg_loss_op, avg_top1_op, avg_top5_op],
                                                                  feed_dict={queue_selector: 1, keepp_pl: keep_prob})


                duration = time.time() - start_time
                print('%s_run_%d: TEST epoch %d, %d/%d. %0.2f hz. loss: %0.04f. top1 %0.04f. top5 %0.04f' %
                      (trainParams.runName, trainParams.n + 1, epoch, bthc, nsteps_eval - 1, 1.0/duration, loss_value, top1_value, top5_value))

            # Save a checkpoint
            if (epoch + 1) % 10 == 0 or (epoch + 1) == trainParams.numEpochs:
                checkpoint_file = os.path.join(trainParams.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=epoch)

        coord.request_stop()
        coord.join(threads)


def runExperiment(trainParams):
    if tf.gfile.Exists(trainParams.log_dir):
        tf.gfile.DeleteRecursively(trainParams.log_dir)

    tf.gfile.MakeDirs(trainParams.log_dir)

    return run_training(trainParams)

