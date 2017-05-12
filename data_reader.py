import numpy as np
import os

class DataReader:
    dummy = False
    data = None
    label = None
    data_path = None
    data_type = 'train'
    batch_size = 16
    start_index = 0
    _epoch = 0
    _one_epoch_completed = False

    def __init__(self, FLAGS, dtype):
        self.data_path = FLAGS.data_path
        self.batch_size = FLAGS.batch_size
        self.start_index = 0
        if dtype != 'train' and dtype != 'test':
            raise NameError('dtype not train or test')
        self.data_type = dtype
        self._load_data()

    def _load_data(self):
        data_filename = os.path.join(self.data_path, self.data_type + '_data.npy')
        label_filename = os.path.join(self.data_path, self.data_type + '_label.npy')

        if self.dummy:
            N, H, W, C = (1000, 32, 32, 3)
            self.data = np.ndarray([N, H, W, C], dtype=np.int8)
            self.label = np.ndarray(N, dtype=np.int32)
        else:
            self.data = np.load(data_filename)
            self.label = np.load(label_filename)

    def next_batch(self):
        s = self.start_index
        e = min(len(self.label), s + self.batch_size)

        batch = self.data[s:e]
        label = self.label[s:e]

        if e == len(self.label):
            self.start_index = 0
            self._epoch += 1
            self._one_epoch_completed = True
        else:
            self.start_index = e
        return batch, label

    def reset(self):
        self._epoch = 0
        self._one_epoch_completed = False

    @property
    def one_epoch_completed(self):
        return self._one_epoch_completed

    @property
    def epoch(self):
        return self._epoch

