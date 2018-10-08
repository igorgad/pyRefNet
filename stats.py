
import os
import tensorflow as tf
import numpy as np
import tfplot


def _add_onehot_to_hash_table_average(onehot, string_ref):
    table = tf.contrib.lookup.MutableHashTable(tf.string, tf.float32, 0)

    unique_ref, uref_idx = tf.unique(string_ref)
    unique_prob = tf.map_fn(lambda idx: tf.reduce_mean(tf.cast(tf.gather(onehot, tf.where(tf.equal(idx, uref_idx))), tf.float32)), tf.range(tf.size(unique_ref)), dtype=tf.float32)

    update_table_op = table.insert(unique_ref, tf.divide(tf.add(table.lookup(unique_ref), unique_prob), 2.0))
    return update_table_op, table.export()


def _create_bar_stats_figure(labels, probs):
    labels = np.array([os.fsdecode(i) for i in labels])
    fig, ax = tfplot.subplots()
    ax.bar(np.arange(probs.size), probs, 0.35)
    ax.set_xticks(np.arange(labels.size))
    ax.set_xticklabels(labels, rotation=+90, ha='center')
    fig.set_tight_layout(True)
    return fig


def _sort_labels(labels):
    labels = np.array([os.fsdecode(i) for i in labels])
    vcc = [[i.replace(' ', ''), j.replace(' ', '')] for i, j in [lb.split(' x ') for lb in labels]]
    slb = [sorted(i) for i in vcc]
    sorted_labels = np.array([''.join([c[0], ' x ', c[1]]) for c in slb], np.chararray)
    return sorted_labels


def _create_confusion_image(confusion_data):
    fig, ax = tfplot.subplots()
    ax.imshow(confusion_data)
    fig.set_size_inches(8, 8)
    return fig


def _create_wrong_example_plot(wis, whs, winst, wtypes, wids):
    winst = [os.fsdecode(i) for i in winst]
    wtypes = [os.fsdecode(i) for i in wtypes]

    mb = np.random.randint(0,wis.shape[0])

    fig, [ax1, ax2] = tfplot.subplots(2,1)
    ax1.plot(wis[mb, 0, :, 0])
    ax1.plot(wis[mb, 0, :, 1])
    ax1.set_title(winst[mb] + ' | ' + wtypes[mb] + ' | id: ' + str(wids[mb]))
    ax2.imshow(whs[mb, :, :, 0])
    fig.set_size_inches(8, 8)
    return fig


def collect_wrong_examples(correct, ins, rkhs, instcombs, typecombs, ids, files):
    sorted_inst = tf.py_func(_sort_labels, [instcombs], tf.string)
    sorted_type = tf.py_func(_sort_labels, [typecombs], tf.string)

    whs = tf.squeeze(tf.gather(rkhs, tf.where(tf.logical_not(correct))), axis=1)
    wis = tf.squeeze(tf.gather(ins, tf.where(tf.logical_not(correct))), axis=1)
    winst = tf.squeeze(tf.gather(sorted_inst, tf.where(tf.logical_not(correct))), axis=1)
    wtypes = tf.squeeze(tf.gather(sorted_type, tf.where(tf.logical_not(correct))), axis=1)
    wids = tf.squeeze(tf.gather(ids, tf.where(tf.logical_not(correct))), axis=1)
    wfiles = tf.squeeze(tf.gather(files, tf.where(tf.logical_not(correct))), axis=1)

    fig = tfplot.plot(_create_wrong_example_plot, [wis, whs, winst, wtypes, wids])

    tf.summary.image('wrong_examples', tf.expand_dims(fig, 0), max_outputs=1)
    tf.summary.text('wrong_ids', tf.reduce_join(tf.as_string(wids), axis=0, separator=','))
    tf.summary.text('wrong_files', wfiles)
    return


def add_genre_stats(correct1, correct5, genre, train_test_table_selector):
    with tf.name_scope('genreStats') as scope:
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

        reset_tables = tf.local_variables_initializer()

    return [update_top1_table, update_top5_table], reset_tables


def add_inst_stats(correct1, correct5, instcombs, train_test_table_selector):
    with tf.name_scope('instStats') as scope:
        sorted_labels = tf.py_func(_sort_labels, [instcombs], tf.string)

        update_train_top1_table, export_train_top1_table = _add_onehot_to_hash_table_average(correct1, sorted_labels)
        update_train_top5_table, export_train_top5_table = _add_onehot_to_hash_table_average(correct5, sorted_labels)
        update_test_top1_table, export_test_top1_table = _add_onehot_to_hash_table_average(correct1, sorted_labels)
        update_test_top5_table, export_test_top5_table = _add_onehot_to_hash_table_average(correct5, sorted_labels)

        update_top1_table, export_top1_table = tf.cond(tf.equal(train_test_table_selector, 0),
                                                       lambda: [update_train_top1_table, export_train_top1_table], lambda: [update_test_top1_table, export_test_top1_table])
        update_top5_table, export_top5_table = tf.cond(tf.equal(train_test_table_selector, 0),
                                                       lambda: [update_train_top5_table, export_train_top5_table], lambda: [update_test_top5_table, export_test_top5_table])

        # top1_plot_op = tfplot.plot(_create_bar_stats_figure, [export_top1_table[0], export_top1_table[1]])
        # top5_plot_op = tfplot.plot(_create_bar_stats_figure, [export_top5_table[0], export_top5_table[1]])
        # tf.summary.image('top1_instcomb', tf.expand_dims(top1_plot_op, 0), max_outputs=1)
        # tf.summary.image('top5_instcomb', tf.expand_dims(top5_plot_op, 0), max_outputs=1)

        top_sorted = tf.nn.top_k(export_top1_table[1], k=tf.shape(export_top1_table[1])[0], sorted=True)
        sidx = top_sorted.indices

        tf.summary.text('top1_instcomb', tf.string_join([tf.gather(export_top1_table[0], sidx), tf.as_string(tf.gather(export_top1_table[1], sidx))], separator=' - '))
        tf.summary.text('top5_instcomb', tf.string_join([tf.gather(export_top5_table[0], sidx), tf.as_string(tf.gather(export_top5_table[1], sidx))], separator=' - '))

        reset_tables = tf.local_variables_initializer()

    return [update_top1_table, update_top5_table], reset_tables


def add_comb_stats(correct1, correct5, typecombs, train_test_table_selector):
    with tf.name_scope('combStats') as scope:
        sorted_typecomb_labels = tf.py_func(_sort_labels, [typecombs], tf.string)

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

        reset_tables = tf.local_variables_initializer()

    return [update_top1_table, update_top5_table], reset_tables


def add_confusion_matrix(logits, labels):
    with tf.name_scope('confusionMatrix') as scope:
        predictions = tf.argmax(logits,1)
        confusion = tf.confusion_matrix(labels=labels, predictions=predictions)

        confusion_img = tfplot.plot(_create_confusion_image, [confusion])
        tf.summary.image('confusion_matrix', tf.expand_dims(confusion_img,0), max_outputs=1)


def add_summaries(loss, eval1, eval5):
    with tf.name_scope('summ') as scope:
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