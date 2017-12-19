import os
import sys
import time

import tensorflow as tf
import numpy as np
import pyRef



def placeholder_inputs(batch_size):
    ins_pl = tf.placeholder(tf.float32, shape=[pyRef.batch_size, pyRef.N, pyRef.nwin, pyRef.nsigs])
    lbs_pl = tf.placeholder(tf.int32, shape=[pyRef.batch_size])
    keepp_pl = tf.placeholder(tf.float32, shape=[1])

    return ins_pl, lbs_pl, keepp_pl


def fill_feed_dict(data_set, keepp_val, ins__pl, lbs_pl, keepp_pl):
    ins_feed, lbs_feed = data_set.next_batch(trainParams.batch_size, trainParams.fake_data) # TODO - remember to fix this
    feed_dict = { ins__pl: ins_feed, lbs_pl: lbs_feed, keepp_pl: keepp_val }

    return feed_dict


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // trainParams.batch_size
    num_examples = steps_per_epoch * trainParams.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def run_training(trainParams):
    """Train MNIST for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on MNIST.
    data_sets = input_data.read_data_sets(trainParams.input_data_dir, trainParams.fake_data)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder = placeholder_inputs(
            trainParams.batch_size)

        # Build a Graph that computes predictions from the inference model.
        logits = mnist.inference(images_placeholder,
                                 trainParams.hidden1,
                                 trainParams.hidden2)

        # Add to the Graph the Ops for loss calculation.
        loss = mnist.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = mnist.training(loss, trainParams.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(trainParams.log_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        for step in xrange(trainParams.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == trainParams.max_steps:
                checkpoint_file = os.path.join(trainParams.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.test)


def runExperiment(trainParams):
    if tf.gfile.Exists(trainParams.log_dir):
        tf.gfile.DeleteRecursively(trainParams.log_dir)
    tf.gfile.MakeDirs(trainParams.log_dir)

    run_training(trainParams)

