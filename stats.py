
import os
import tensorflow as tf
import numpy as np
import tfplot


def _add_onehot_to_hash_table_average(onehot, string_ref):
    table = tf.contrib.lookup.MutableHashTable(tf.string, tf.float32, 0)
    update_table_op = table.insert(string_ref, tf.divide(tf.add(table.lookup(string_ref), tf.cast(onehot, tf.float32)), 2))
    return update_table_op, table.export()


def _create_bar_stats_figure(labels, probs):
    labels = np.array([os.fsdecode(i) for i in labels])
    fig, ax = tfplot.subplots()
    ax.bar(np.arange(probs.size), probs, 0.35)
    ax.set_xticks(np.arange(labels.size))
    ax.set_xticklabels(labels, rotation=+90, ha='center')
    fig.set_tight_layout(True)
    return fig


def _sort_typecomb_labels(labels):
    labels = np.array([os.fsdecode(i) for i in labels])
    vcc = [[i.replace(' ', ''), j.replace(' ', '')] for i, j in [lb.split('x') for lb in labels]]
    slb = [sorted(i) for i in vcc]
    sorted_labels = np.array([''.join([c[0], ' x ', c[1]]) for c in slb], np.chararray)
    return sorted_labels


def _create_confusion_image(confusion_data):
    fig, ax = tfplot.subplots()
    ax.imshow(confusion_data)
    fig.set_size_inches(8, 8)
    return fig


def add_genre_stats(correct1, correct5, genre, train_test_table_selector):
    with tf.name_scope('genreStats') as scope:
        with tf.device('/cpu:0'):
            update_train_top1_table, export_train_top1_table = _add_onehot_to_hash_table_average(correct1, genre)
            update_train_top5_table, export_train_top5_table = _add_onehot_to_hash_table_average(correct5, genre)
            update_test_top1_table, export_test_top1_table   = _add_onehot_to_hash_table_average(correct1, genre)
            update_test_top5_table, export_test_top5_table   = _add_onehot_to_hash_table_average(correct5, genre)

            update_top1_table, export_top1_table = tf.cond(tf.equal(train_test_table_selector, 0),
                                                           lambda: [update_train_top1_table, export_train_top1_table], lambda: [update_test_top1_table, export_test_top1_table])
            update_top5_table, export_top5_table = tf.cond(tf.equal(train_test_table_selector, 0),
                                                           lambda: [update_train_top5_table, export_train_top5_table], lambda: [update_test_top5_table, export_test_top5_table])

            top1_plot_op = tfplot.plot(_create_bar_stats_figure, [export_top1_table[0], export_top1_table[1]])
            top5_plot_op = tfplot.plot(_create_bar_stats_figure, [export_top5_table[0], export_top5_table[1]])

            tf.summary.image('top1_genre', tf.expand_dims(top1_plot_op, 0), max_outputs=1)
            tf.summary.image('top5_genre', tf.expand_dims(top5_plot_op, 0), max_outputs=1)

    return [update_top1_table, update_top5_table]


def add_comb_stats(correct1, correct5, typecombs, train_test_table_selector):
    with tf.name_scope('combStats') as scope:
        with tf.device('/cpu:0'):
            sorted_typecomb_labels = tf.py_func(_sort_typecomb_labels, [typecombs], tf.string)

            update_train_top1_table, export_train_top1_table = _add_onehot_to_hash_table_average(correct1, sorted_typecomb_labels)
            update_train_top5_table, export_train_top5_table = _add_onehot_to_hash_table_average(correct5, sorted_typecomb_labels)
            update_test_top1_table, export_test_top1_table = _add_onehot_to_hash_table_average(correct1, sorted_typecomb_labels)
            update_test_top5_table, export_test_top5_table = _add_onehot_to_hash_table_average(correct5, sorted_typecomb_labels)

            update_top1_table, export_top1_table = tf.cond(tf.equal(train_test_table_selector, 0),
                                                           lambda: [update_train_top1_table, export_train_top1_table], lambda: [update_test_top1_table, export_test_top1_table])
            update_top5_table, export_top5_table = tf.cond(tf.equal(train_test_table_selector, 0),
                                                           lambda: [update_train_top5_table, export_train_top5_table], lambda: [update_test_top5_table, export_test_top5_table])

            top1_plot_op = tfplot.plot(_create_bar_stats_figure, [export_top1_table[0], export_top1_table[1]])
            top5_plot_op = tfplot.plot(_create_bar_stats_figure, [export_top5_table[0], export_top5_table[1]])

            tf.summary.image('top1_typecomb', tf.expand_dims(top1_plot_op, 0), max_outputs=1)
            tf.summary.image('top5_typecomb', tf.expand_dims(top5_plot_op, 0), max_outputs=1)

    return [update_top1_table, update_top5_table]


def add_confusion_matrix(logits, labels):
    with tf.name_scope('confusionMatrix') as scope:
        with tf.device('/cpu:0'):
            predictions = tf.argmax(logits,1)
            confusion = tf.confusion_matrix(labels=labels, predictions=predictions)

            confusion_img = tfplot.plot(_create_confusion_image, [confusion])
            tf.summary.image('confusion_matrix', tf.expand_dims(confusion_img,0), max_outputs=1)


def add_summaries(loss, eval1, eval5):
    with tf.name_scope('summ') as scope:
        with tf.device('/cpu:0'):
            avg_loss, avg_loss_op = tf.contrib.metrics.streaming_mean(loss)
            avg_top1, avg_top1_op = tf.contrib.metrics.streaming_mean(eval1)
            avg_top5, avg_top5_op = tf.contrib.metrics.streaming_mean(eval5)

            # vars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            # reset_op = tf.variables_initializer(vars)
            reset_op = tf.local_variables_initializer()

            tf.summary.scalar('avg_loss', avg_loss)
            tf.summary.scalar('avg_top1', avg_top1)
            tf.summary.scalar('avg_top5', avg_top5)

    return avg_loss_op, avg_top1_op, avg_top5_op, reset_op