from __future__ import print_function

import tensorflow as tf
import numpy as np
import do_validate
import pdb

def run(FLAGS, sess, net, saver, data_train, data_test):
    loss = net.loss
    acc = net.acc

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./log/', sess.graph)

    ph_image, ph_label = net.placeholders()

    optimizer = tf.train.GradientDescentOptimizer(0.05)
    train_op = optimizer.minimize(loss)

    prev_epoch = data_train.epoch
    while (data_train.epoch < FLAGS.epoches):
        image, label = data_train.next_batch()
        image = image.astype(np.float32) / 255.0
        label = label.astype(np.int32)
        feed_dict = {
            ph_image: image,
            ph_label: label
        }
        _, acc_val, summary_str = sess.run([train_op, acc, summary_op], feed_dict=feed_dict)
        if data_train.iteration % 100 == 0:
            print("Train: %3.3f" % acc_val)
            summary_writer.add_summary(summary_str, data_train.iteration)
        if (prev_epoch != data_train.epoch):
            print('Epoch[%03d]:' % data_train.epoch, end=' ')
            do_validate.run(sess, net, data_test)
            saver.save(sess, "./log/model.ckpt", data_train.iteration)
        prev_epoch = data_train.epoch
    sess.close()