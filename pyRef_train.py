
import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import models.rkhsModel as model


mmap = None  # FIX
def np_get_batch(batch_ids, batch_size):
    bcan = np.random.choice(batch_ids, batch_size)

    ins_feed = np.array([mmap[itm][0] for itm in bcan]).transpose([0,2,3,1])
    lbs_feed = np.array([mmap[itm][1] for itm in bcan]) + 80
    instcomb = np.array([mmap[itm][2] for itm in bcan])
    typecomb = np.array([mmap[itm][3] for itm in bcan])

    return ins_feed.astype(np.float32), lbs_feed.astype(np.int32), instcomb.astype(np.chararray), typecomb.astype(np.chararray)


def tf_get_batch(batch_ids, batch_size):
    ins, lbs, instcombs, typecombs = tf.py_func(np_get_batch, [batch_ids, batch_size], [tf.float32, tf.int32, tf.string, tf.string])

    return ins, lbs, instcombs, typecombs


def add_queues(batch_size, train_ids, eval_ids, selector_pl):
    with tf.name_scope('queues') as scope:
        q_train = tf.FIFOQueue(3, dtypes=[tf.float32, tf.int32, tf.string, tf.string], shapes=[[batch_size, model.nwin, model.N, model.nsigs], [batch_size], [batch_size], [batch_size]])
        q_eval = tf.FIFOQueue(8, dtypes=[tf.float32, tf.int32, tf.string, tf.string], shapes=[[batch_size, model.nwin, model.N, model.nsigs], [batch_size], [batch_size], [batch_size]])

        q_train_op = q_train.enqueue(tf_get_batch(train_ids, batch_size))
        q_eval_op = q_eval.enqueue(tf_get_batch(eval_ids, batch_size))

        qr_train = tf.train.QueueRunner(q_train, [q_train_op] * 3)
        qr_eval = tf.train.QueueRunner(q_eval, [q_eval_op] * 8)

        tf.train.add_queue_runner(qr_train)
        tf.train.add_queue_runner(qr_eval)

        q = tf.QueueBase.from_list(selector_pl, [q_train, q_eval])

        ins, lbs, instcombs, typecombs = q.dequeue()

    return ins, lbs, instcombs, typecombs


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


def create_comb_tables(correct1, correct5, instcombs, typecombs):
    top1_instcomb = tf.gather(instcombs, tf.where(correct1))
    top1_typecomb = tf.gather(typecombs, tf.where(correct1))
    top5_instcomb = tf.gather(instcombs, tf.where(correct5))
    top5_typecomb = tf.gather(typecombs, tf.where(correct5))

    top1_instcomb_table = tf.contrib.lookup.MutableHashTable(tf.string, tf.float32, 0)
    top1_typecomb_table = tf.contrib.lookup.MutableHashTable(tf.string, tf.float32, 0)
    top5_instcomb_table = tf.contrib.lookup.MutableHashTable(tf.string, tf.float32, 0)
    top5_typecomb_table = tf.contrib.lookup.MutableHashTable(tf.string, tf.float32, 0)

    update_top1_instcomb_table = top1_instcomb_table.insert(top1_instcomb, tf.divide(tf.add(top1_instcomb_table.lookup(top1_instcomb), 1), 2))
    update_top1_typecomb_table = top1_typecomb_table.insert(top1_typecomb, tf.divide(tf.add(top1_typecomb_table.lookup(top1_typecomb), 1), 2))
    update_top5_instcomb_table = top5_instcomb_table.insert(top5_instcomb, tf.divide(tf.add(top5_instcomb_table.lookup(top5_instcomb), 1), 2))
    update_top5_typecomb_table = top5_typecomb_table.insert(top5_typecomb, tf.divide(tf.add(top5_typecomb_table.lookup(top5_typecomb), 1), 2))

    export_top1_instcomb_table = top1_instcomb_table.export()
    export_top1_typecomb_table = top1_typecomb_table.export()
    export_top5_instcomb_table = top5_instcomb_table.export()
    export_top5_typecomb_table = top5_typecomb_table.export()

    update_ops = [update_top1_instcomb_table, update_top1_typecomb_table, update_top5_instcomb_table, update_top5_typecomb_table]
    export_ops = [export_top1_instcomb_table, export_top1_typecomb_table, export_top5_instcomb_table, export_top5_typecomb_table]

    return update_ops, export_ops


# TODO: ideally these tables should be cleaned after each epoch
def add_comb_stats(correct1, correct5, instcombs, typecombs, table_selector):
    with tf.name_scope('combStats') as scope:
        train_update_ops, train_export_ops = create_comb_tables(correct1, correct5, instcombs, typecombs)
        test_update_ops, test_export_ops = create_comb_tables(correct1, correct5, instcombs, typecombs)

        update_stats, export_stats = tf.cond(tf.equal(table_selector, 0), lambda: [train_update_ops, train_export_ops], lambda: [test_update_ops, test_export_ops])

    return update_stats, export_stats


def run_training(trainParams):

    with tf.Graph().as_default():

        keepp_pl = tf.placeholder(tf.float32)
        queue_selector = tf.placeholder(tf.int32)

        ins, lbs, instcombs, typecombs = add_queues(trainParams.batch_size, trainParams.trainIds, trainParams.evalIds, queue_selector)

        logits = model.inference(ins, keepp_pl)
        loss   = model.loss(logits, lbs)
        train_op = model.training(loss, trainParams.lr, trainParams.momentum)
        eval_top1, eval_top5, correct1, correct5 = model.evaluation(logits, lbs)

        avg_loss_op, avg_top1_op, avg_top5_op, reset_op = add_summaries(loss, eval_top1, eval_top5)
        update_stats, export_stats = add_comb_stats(correct1, correct5, instcombs, typecombs, queue_selector)

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

        try:
            # Start the training loop.
            for epoch in range(trainParams.numEpochs):
                # Train
                for bthc in range(nsteps_train):
                    sum_step = trainParams.sumPerEpoch * epoch + bthc // np.ceil(nsteps_train / trainParams.sumPerEpoch)

                    keep_prob = 0.7 #Dynamic control of dropout rate

                    start_time = time.time()

                    # Log training runtime statistics
                    if np.mod(bthc + 1, np.ceil(nsteps_train / trainParams.sumPerEpoch)) == 0:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary_str, _, loss_value, top1_value, top5_value, __ = sess.run([summary, train_op, avg_loss_op, avg_top1_op, avg_top5_op, update_stats],
                                                                                      feed_dict={queue_selector: 0, keepp_pl: keep_prob}, options=run_options, run_metadata=run_metadata)

                        train_writer.add_run_metadata(run_metadata, 'epoch %d' % sum_step )
                        train_writer.add_summary(summary_str, sum_step )
                        train_writer.flush()

                        sess.run([reset_op])
                    else:
                        _, loss_value, top1_value, top5_value, __ = sess.run([train_op, avg_loss_op, avg_top1_op, avg_top5_op, update_stats],
                                                                          feed_dict={queue_selector: 0, keepp_pl: keep_prob})


                    duration = time.time() - start_time
                    print ('%s_run_%d: TRAIN epoch %d, %d/%d. %0.2f hz loss: %0.04f top1 %0.04f top5 %0.04f' %
                           (trainParams.runName, trainParams.n + 1, epoch, bthc, nsteps_train - 1, trainParams.batch_size/duration, loss_value, top1_value, top5_value) )

                # Evaluate
                for bthc in range(nsteps_eval):
                    sum_step = trainParams.sumPerEpoch * epoch + bthc // np.ceil(nsteps_eval / trainParams.sumPerEpoch)

                    keep_prob = 1.0  # Dynamic control of dropout rate

                    start_time = time.time()

                    # Log testing runtime statistics
                    if np.mod(bthc + 1, np.ceil(nsteps_eval / trainParams.sumPerEpoch)) == 0:
                        summary_str, loss_value, top1_value, top5_value, _ = sess.run([summary, avg_loss_op, avg_top1_op, avg_top5_op, update_stats],
                                                                                   feed_dict={queue_selector: 1, keepp_pl: keep_prob})
                        test_writer.add_summary(summary_str, sum_step )
                        test_writer.flush()

                        sess.run([reset_op])
                    else:
                        loss_value, top1_value, top5_value, _ = sess.run([avg_loss_op, avg_top1_op, avg_top5_op, update_stats],
                                                                      feed_dict={queue_selector: 1, keepp_pl: keep_prob})


                    duration = time.time() - start_time
                    print('%s_run_%d: TEST epoch %d, %d/%d. %0.2f hz. loss: %0.04f. top1 %0.04f. top5 %0.04f' %
                          (trainParams.runName, trainParams.n + 1, epoch, bthc, nsteps_eval - 1, trainParams.batch_size/duration, loss_value, top1_value, top5_value))


                # Save a checkpoint
                if (epoch + 1) % 10 == 0 or (epoch + 1) == trainParams.numEpochs:
                    checkpoint_file = os.path.join(trainParams.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=epoch)

        finally:
            print ('\n\ncleaning...')

            train_stats = np.array(sess.run([export_stats], {queue_selector: 0}))
            test_stats  = np.array(sess.run([export_stats], {queue_selector: 1}))

            coord.request_stop()
            coord.join(threads)
            sess.close()

            f, trainsubplt = plt.subplots(2, 2)
            trainsubplt[0,0].bar(np.arange(train_stats[0,0,1].size), train_stats[0,0,1], 0.35)
            trainsubplt[0,0].set_xticks(np.arange(train_stats[0,0,0].size))
            trainsubplt[0,0].set_xticklabels(train_stats[0,0,0], rotation=-90,  ha='center')
            trainsubplt[0,0].set_title('TOP1 instcomb - training')
            trainsubplt[0, 1].bar(np.arange(train_stats[0,2,1].size), train_stats[0, 2, 1], 0.35)
            trainsubplt[0, 1].set_xticks(np.arange(train_stats[0,2,0].size))
            trainsubplt[0, 1].set_xticklabels(train_stats[0, 2, 0], rotation=-90, ha='center')
            trainsubplt[0, 1].set_title('TOP5 instcomb - training')
            trainsubplt[1, 0].bar(np.arange(train_stats[0,1,1].size), train_stats[0, 1, 1], 0.35)
            trainsubplt[1, 0].set_xticks(np.arange(train_stats[0,1,0].size))
            trainsubplt[1, 0].set_xticklabels(train_stats[0, 1, 0], rotation=-90, ha='center')
            trainsubplt[1, 0].set_title('TOP1 typecomb - training')
            trainsubplt[1, 1].bar(np.arange(train_stats[0,3,1].size), train_stats[0, 3, 1], 0.35)
            trainsubplt[1, 1].set_xticks(np.arange(train_stats[0,3,0].size))
            trainsubplt[1, 1].set_xticklabels(train_stats[0, 3, 0], rotation=-90, ha='center')
            trainsubplt[1, 1].set_title('TOP5 typecomb - training')

            f2, testsubplt = plt.subplots(2, 2)
            testsubplt[0, 0].bar(np.arange(test_stats[0,0,1].size), test_stats[0, 0, 1], 0.35)
            testsubplt[0, 0].set_xticks(np.arange(test_stats[0,0,0].size), test_stats[0, 0, 0])
            testsubplt[0, 0].set_title('TOP1 instcomb - testing')
            testsubplt[0, 1].bar(np.arange(test_stats[0,2,1].size), test_stats[0, 2, 1], 0.35)
            testsubplt[0, 1].set_xticks(np.arange(test_stats[0,2,0].size), test_stats[0, 2, 0])
            testsubplt[0, 1].set_title('TOP5 instcomb - testing')
            testsubplt[1, 0].bar(np.arange(test_stats[0,1,1].size), test_stats[0, 1, 1], 0.35)
            testsubplt[1, 0].set_xticks(np.arange(test_stats[0,1,0].size), test_stats[0, 1, 0])
            testsubplt[1, 0].set_title('TOP1 typecomb - testing')
            testsubplt[1, 1].bar(np.arange(test_stats[0,3,1].size), test_stats[0, 3, 1], 0.35)
            testsubplt[1, 1].set_xticks(np.arange(test_stats[0,3,0].size), test_stats[0, 3, 0])
            testsubplt[1, 1].set_title('TOP5 typecomb - testing')

            return train_stats, test_stats


def runExperiment(trainParams):
    if tf.gfile.Exists(trainParams.log_dir):
        tf.gfile.DeleteRecursively(trainParams.log_dir)

    tf.gfile.MakeDirs(trainParams.log_dir)

    return run_training(trainParams)

