import tensorflow as tf
import numpy as np
import do_validate
import pdb

def run(FLAGS, net, data_train, data_test):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    loss = net.loss()
    acc = net.acc()
    ph_image, ph_label = net.placeholders()

    optimizer = tf.train.GradientDescentOptimizer(0.05)
    train_op = optimizer.minimize(loss)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess.run(init_op)
    prev_epoch = data_train.epoch
    while (data_train.epoch <= FLAGS.epoches):
        image, label = data_train.next_batch()
        image = image.astype(np.float32) / 255.0
        label = label.astype(np.int32)
        feed_dict = {
            ph_image: image,
            ph_label: label
        }
        _, acc_val = sess.run([train_op, acc], feed_dict=feed_dict)
        if (prev_epoch != data_train.epoch):
            do_validate.run(sess, acc, data_test, ph_image, ph_label)
        prev_epoch = data_train.epoch
    sess.close()