import tensorflow as tf
import numpy as np
import os

from data_reader import DataReader
from network import Net
import do_train
import do_validate

# parameters for app:
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", 100, "batch size for training")
tf.flags.DEFINE_integer("epoches", 20, "number of epoches")
tf.flags.DEFINE_integer("disp", 100, "how many iterations to display")
tf.flags.DEFINE_float("weight_decay", 0.001, "weight decay")
tf.flags.DEFINE_float("learning_rate", 0.05, "learning rate")
tf.flags.DEFINE_string("data_path", "./data/", "data path storing npy files")
tf.flags.DEFINE_string("log_path", "./log/", "log path storing checkpoints")
tf.flags.DEFINE_string("mode", "train", "train or test")

def main():
    train_data_reader = DataReader(FLAGS, dtype='train')
    test_data_reader = DataReader(FLAGS, dtype='test')

    with tf.Graph().as_default():
        net = Net(FLAGS)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        saver = tf.train.Saver()

        if FLAGS.mode == 'train':
            do_train.run(FLAGS, sess, net, saver, train_data_reader, test_data_reader)
        else:
            ckpt = tf.train.get_checkpoint_state(FLAGS.log_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored...")
            if FLAGS.mode == 'test':
                do_validate.run(sess, net, test_data_reader)
            else:
                do_train.run(FLAGS, sess, net, saver, train_data_reader, test_data_reader)


if __name__ == '__main__':
    main()