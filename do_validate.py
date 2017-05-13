import tensorflow as tf
import numpy as np

def run(sess, net, data_test):
    ph_image, ph_label = net.placeholders()
    acc = net.acc
    acc_total = 0.0
    n = 0
    data_test.reset()
    while not data_test.one_epoch_completed:
        image, label = data_test.next_batch()
        image = image.astype(np.float32) / 255.0
        label = label.astype(np.int32)
        feed_dict = {
            ph_image: image,
            ph_label: label
        }
        acc_val = sess.run(acc, feed_dict=feed_dict)
        acc_total += acc_val
        n += 1
    acc_total /= n
    print('Validation: %3.3f' % acc_total)

