
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

    ins = features
    lbs = labels['label']
    typecombs = labels['typecomb']
    genres = labels['genre']

    keep_prob = 1.0
    train_test_selector = 1
    if tf.estimator.ModeKeys.TRAIN:
        keep_prob = params['keep_prob']
        train_test_selector = 0


    logits = model.inference(ins, keep_prob)
    loss = model.loss(logits, lbs)
    optimizer = tf.train.AdamOptimizer(model.lr)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())

    correct1 = tf.nn.in_top_k(logits, lbs, 1)
    correct5 = tf.nn.in_top_k(logits, lbs, 5)

    update_comb_stats = stats.add_comb_stats(correct1, correct5, typecombs, train_test_selector)
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

