import tensorflow as tf
import utils as utils

class Net:
    batch_size = 16
    ph_image = None
    ph_label = None
    ph_batch_size = None
    logits = None
    loss = None
    acc = None

    def __init__(self, FLAGS):
        self.batch_size = FLAGS.batch_size
        self.ph_image = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='image')
        self.ph_label = tf.placeholder(tf.int32, shape=(None), name='label')
        self.logits = self.inference(False)
        self.loss = self.compute_loss(FLAGS.weight_decay)
        self.acc = self.compute_acc()

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

            dim = 1
            for d in pool3.get_shape()[1:].as_list():
                dim *= d
            reshape = tf.reshape(pool3, [-1, dim])

            with tf.variable_scope('fc1') as scope:
                weights = utils.weight_variable([dim, 64], name='weights')
                biases = utils.bias_variable([64], name='biases')
                fc1 = tf.matmul(reshape, weights) + biases
            with tf.variable_scope('fc2') as scope:
                weights = utils.weight_variable([64, 10], name='weights')
                biases = utils.bias_variable([10], name='biases')
                fc2 = tf.matmul(fc1, weights) + biases
        return fc2

    def compute_loss(self, wd):
        label = tf.cast(self.ph_label, tf.int32)
        logits = self.logits
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label, name='xentropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        train_variables = tf.trainable_variables()
        for v in train_variables:
            utils.add_to_regularization(var=v)
        l2norm = tf.add_n(tf.get_collection('reg_loss'), name='l2norm')
        return cross_entropy_mean + wd * l2norm

    def compute_acc(self):
        label = tf.cast(self.ph_label, tf.int64)
        logits = self.logits
        acc_mean = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), label), tf.float32))
        return acc_mean