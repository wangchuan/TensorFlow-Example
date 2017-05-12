import tensorflow as tf
import numpy as np
import os

from data_reader import DataReader
from network import Net
import do_train
import do_validate

# parameters for app:
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "100", "batch size for training")
tf.flags.DEFINE_integer("epoches", "20", "number of epoches")
tf.flags.DEFINE_string("data_path", "./data/", "data path storing npy files")

def main():
    train_data_reader = DataReader(FLAGS, dtype='train')
    test_data_reader = DataReader(FLAGS, dtype='test')

    with tf.Graph().as_default():
        net = Net(FLAGS)
        do_train.run(FLAGS, net, train_data_reader, test_data_reader)

if __name__ == '__main__':
    main()