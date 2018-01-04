
import os
import time

import tensorflow as tf
import numpy as np
import models.rkhsModel as model


def placeholder_inputs(batch_size):
    ins_pl = tf.placeholder(tf.float32, shape=[batch_size, model.nwin, model.N, model.nsigs])
    lbs_pl = tf.placeholder(tf.int32, shape=[batch_size])
    keepp_pl = tf.placeholder(tf.float32)

    return ins_pl, lbs_pl, keepp_pl


def fill_feed_dict(mmap, batch_ids, keep_prob, ins_pl, lbs_pl, keepp_pl):
    # TODO: Optimize dataset access function.
    ins_feed = np.transpose(np.array([mmap[itm][0] for itm in batch_ids]), [0,2,3,1]) # NEWS: Set [bsize,nwin,N,nsigs]
    lbs_feed = np.array([mmap[itm][1] for itm in batch_ids]) + 80

    feed_dict = { ins_pl: ins_feed.astype(np.float32), lbs_pl: lbs_feed.astype(np.int32), keepp_pl: keep_prob }

    return feed_dict


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

        ins_pl, lbs_pl, keepp_pl = placeholder_inputs( trainParams.batch_size )

        logits = model.inference(ins_pl, keepp_pl)
        loss = model.loss(logits, lbs_pl)
        train_op = model.training(loss, trainParams.lr, trainParams.momentum)
        eval_top1, eval_top5 = model.evaluation(logits, lbs_pl)
        avg_loss_op, avg_top1_op, avg_top5_op, reset_op = add_summaries(loss, eval_top1, eval_top5)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        train_writer = tf.summary.FileWriter(trainParams.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(trainParams.log_dir + '/test')

        sess.run([init, reset_op])

        ntrain = trainParams.trainIds.size
        neval  = trainParams.evalIds.size

        nsteps_train = np.int32(np.floor(ntrain / trainParams.batch_size))
        nsteps_eval = np.int32(np.floor(neval / trainParams.batch_size))


        # Start the training loop.
        for epoch in range(trainParams.numEpochs):
            # Train
            for bthc in range(nsteps_train):
                batch_ids = np.random.choice(trainParams.trainIds, trainParams.batch_size)
                sum_step = trainParams.sumPerEpoch * epoch + bthc // np.ceil(nsteps_train / trainParams.sumPerEpoch)

                keep_prob = 0.7 #Dynamic control of dropout rate

                start_time = time.time()

                feed_dict = fill_feed_dict(trainParams.mmap, batch_ids, keep_prob, ins_pl, lbs_pl, keepp_pl)

                # Log training runtime statistics. One per epoch (last step)
                if np.mod(bthc, np.ceil(nsteps_train / trainParams.sumPerEpoch)) == 1:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary_str, _, loss_value, top1_value, top5_value = sess.run([summary, train_op, avg_loss_op, avg_top1_op, avg_top5_op],
                                                                                  feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

                    train_writer.add_run_metadata(run_metadata, 'epoch %d' % sum_step )
                    train_writer.add_summary(summary_str, sum_step )
                    train_writer.flush()

                    sess.run([reset_op])
                else:
                     _, loss_value, top1_value, top5_value = sess.run([train_op, avg_loss_op, avg_top1_op, avg_top5_op], feed_dict=feed_dict)

                duration = time.time() - start_time
                print ('%s_run_%d: TRAIN epoch %d, %d/%d. %0.2f hz loss: %0.04f top1 %0.04f top5 %0.04f' %
                       (trainParams.runName, trainParams.n + 1, epoch, bthc, nsteps_train - 1, 1.0/duration, loss_value, top1_value, top5_value) )

            # Evaluate
            for bthc in range(nsteps_eval):
                batch_ids = np.random.choice(trainParams.evalIds, trainParams.batch_size)
                sum_step = trainParams.sumPerEpoch * epoch + bthc // np.ceil(nsteps_eval / trainParams.sumPerEpoch)

                keep_prob = 1.0  # Dynamic control of dropout rate

                start_time = time.time()

                feed_dict = fill_feed_dict(trainParams.mmap, batch_ids, keep_prob, ins_pl, lbs_pl, keepp_pl)

                # Log testing runtime statistics. One per epoch (last step)
                if np.mod(bthc, np.ceil(nsteps_eval / trainParams.sumPerEpoch)) == 1:
                    summary_str, loss_value, top1_value, top5_value = sess.run([summary, avg_loss_op, avg_top1_op, avg_top5_op], feed_dict=feed_dict)
                    test_writer.add_summary(summary_str, sum_step )
                    test_writer.flush()

                    sess.run([reset_op])
                else:
                    loss_value, top1_value, top5_value = sess.run([avg_loss_op, avg_top1_op, avg_top5_op], feed_dict=feed_dict)


                duration = time.time() - start_time
                print('%s_run_%d: TEST epoch %d, %d/%d. %0.2f hz. loss: %0.04f. top1 %0.04f. top5 %0.04f' %
                      (trainParams.runName, trainParams.n + 1, epoch, bthc, nsteps_eval - 1, 1.0/duration, loss_value, top1_value, top5_value))

            # Save a checkpoint
            if (epoch + 1) % 10 == 0 or (epoch + 1) == trainParams.numEpochs:
                checkpoint_file = os.path.join(trainParams.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=epoch)


def runExperiment(trainParams):
    if tf.gfile.Exists(trainParams.log_dir):
        tf.gfile.DeleteRecursively(trainParams.log_dir)

    tf.gfile.MakeDirs(trainParams.log_dir)

    return run_training(trainParams)

