
import os
import time

import tensorflow as tf
import numpy as np
import pyRef


def placeholder_inputs(batch_size):
    ins_pl = tf.placeholder(tf.float32, shape=[batch_size, pyRef.nwin, pyRef.N, pyRef.nsigs])
    lbs_pl = tf.placeholder(tf.int32, shape=[batch_size])
    keepp_pl = tf.placeholder(tf.float32)

    return ins_pl, lbs_pl, keepp_pl


def fill_feed_dict(mmap, batch_ids, keep_prob, ins_pl, lbs_pl, keepp_pl):
    # TODO: Optimize dataset access function.

    ins_feed = np.transpose(np.array([mmap[itm][0] for itm in batch_ids]), [0,2,3,1]) # NEWS: Set [bsize,nwin,N,nsigs]
    lbs_feed = np.array([mmap[itm][1] for itm in batch_ids]) + 80

    feed_dict = { ins_pl: ins_feed.astype(np.float32), lbs_pl: lbs_feed.astype(np.int32), keepp_pl: keep_prob }

    return feed_dict


def run_training(trainParams):

    with tf.Graph().as_default():

        ins_pl, lbs_pl, keepp_pl = placeholder_inputs( trainParams.batch_size )

        logits = pyRef.inference(ins_pl, keepp_pl)
        loss = pyRef.loss(logits, lbs_pl)
        train_op = pyRef.training(loss, trainParams.lr, trainParams.momentum)
        eval_top1, eval_top5 = pyRef.evaluation(logits, lbs_pl)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        summary_writer = tf.summary.FileWriter(trainParams.log_dir, sess.graph)

        sess.run(init)

        ntrain = trainParams.trainIds.size
        neval  = trainParams.evalIds.size

        nsteps_train = np.int32(np.floor(ntrain / trainParams.batch_size))
        nsteps_eval = np.int32(np.floor(neval / trainParams.batch_size))

        # Start the training loop.
        for epoch in range(trainParams.numEpochs):
            # Train
            for bthc in range(nsteps_train):

                batch_ids = np.random.choice(trainParams.trainIds,trainParams.batch_size)
                keep_prob = 0.5 #Dynamic control of dropout rate

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                start_time = time.time()

                feed_dict = fill_feed_dict(trainParams.mmap, batch_ids, keep_prob, ins_pl, lbs_pl, keepp_pl)
                summary_str, _, loss_value, top1_value, top5_value = sess.run([summary, train_op, loss, eval_top1, eval_top5],
                                                                              feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

                duration = time.time() - start_time

                summary_writer.add_run_metadata(run_metadata, 'step %d' % (epoch*nsteps_train + bthc))
                summary_writer.add_summary(summary_str, epoch*nsteps_train + bthc)
                summary_writer.flush()
                print ('TRAIN epoch %d, %d/%d. %0.2f hz loss: %0.04f top1 %0.04f top5 %0.04f' % (epoch, bthc, nsteps_train, 1.0/duration, loss_value, top1_value, top5_value) )

            # Evaluate
            for bthc in range(nsteps_eval):
                batch_ids = np.random.choice(trainParams.evalIds, trainParams.batch_size)
                keep_prob = 1.0  # Dynamic control of dropout rate

                start_time = time.time()

                feed_dict = fill_feed_dict(trainParams.mmap, batch_ids, keep_prob, ins_pl, lbs_pl, keepp_pl)
                summary_str, loss_value, top1_value, top5_value = sess.run([summary, loss, eval_top1, eval_top5], feed_dict=feed_dict)

                duration = time.time() - start_time

                summary_writer.add_summary(summary_str, epoch*nsteps_eval + bthc)
                summary_writer.flush()
                print('TEST epoch %d, %d/%d. %0.2f hz. loss: %0.04f. top1 %0.04f. top5 %0.04f' % ( epoch, bthc, nsteps_eval, 1.0/duration, loss_value, top1_value, top5_value))
            

            # Save a checkpoint
            if (epoch + 1) % 10 == 0 or (epoch + 1) == trainParams.max_steps:
                checkpoint_file = os.path.join(trainParams.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=epoch)


def runExperiment(trainParams):
    if tf.gfile.Exists(trainParams.log_dir):
        tf.gfile.DeleteRecursively(trainParams.log_dir)

    tf.gfile.MakeDirs(trainParams.log_dir)

    return run_training(trainParams)

