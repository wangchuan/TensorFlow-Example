import tensorflow as tf
import utils as utils

class Net:
    batch_size = 16
    ph_image = None
    ph_label = None
    layers = None

    def __init__(self, FLAGS):
        self.batch_size = FLAGS.batch_size
        self.ph_image = tf.placeholder(tf.float32, shape=(self.batch_size, 32, 32, 3), name='image')
        self.ph_label = tf.placeholder(tf.int32, shape=(self.batch_size), name='label')

    def placeholders(self):
        return self.ph_image, self.ph_label

    def inference(self, flag):
        with tf.variable_scope('layers', reuse=flag) as layer_scope:
            with tf.variable_scope('stage1') as scope:
                kernel = utils.weight_variable([5,5,3,32], name='weights')
                biases = utils.bias_variable([32], name='biases')
                conv1 = utils.conv2d(self.ph_image, kernel, biases)
                pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1')
                relu1 = tf.nn.relu(pool1, name=scope.name)
            with tf.variable_scope('stage2') as scope:
                kernel = utils.weight_variable([5,5,32,32], name='weights')
                biases = utils.bias_variable([32], name='biases')
                conv2 = utils.conv2d(relu1, kernel, biases)
                relu2 = tf.nn.relu(conv2, name='relu2')
                pool2 = tf.nn.avg_pool(relu2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=scope.name)
            with tf.variable_scope('stage3') as scope:
                kernel = utils.weight_variable([5,5,32,64], name='weights')
                biases = utils.bias_variable([64], name='biases')
                conv3 = utils.conv2d(pool2, kernel, biases)
                relu3 = tf.nn.relu(conv3, name='relu3')
                pool3 = tf.nn.avg_pool(relu3, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=scope.name)

            reshape = tf.reshape(pool3, [self.batch_size, -1])
            dim = reshape.get_shape()[1].value

            with tf.variable_scope('fc1') as scope:
                weights = utils.weight_variable([dim, 64], name='weights')
                biases = utils.bias_variable([64], name='biases')
                fc1 = tf.matmul(reshape, weights) + biases
            with tf.variable_scope('fc2') as scope:
                weights = utils.weight_variable([64, 10], name='weights')
                biases = utils.bias_variable([10], name='biases')
                fc2 = tf.matmul(fc1, weights) + biases
        return fc2

    def loss(self):
        label = tf.cast(self.ph_label, tf.int32)
        logits = self.inference(False)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label, name='xentropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return cross_entropy_mean

    def acc(self):
        label = tf.cast(self.ph_label, tf.int64)
        logits = self.inference(True)
        acc_mean = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), label), tf.float32))
        return acc_mean