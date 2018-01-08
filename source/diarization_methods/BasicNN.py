from diarization_methods.DiarizationBaseClass import DiarizationBaseClass
import tensorflow as tf
import numpy as np

from numpy import array, zeros


class BasicNN(DiarizationBaseClass):
    def neuron_layer(self, X, n_neurons, name, activation=None):
        with tf.name_scope(name):
            n_inputs = int(X.get_shape()[1])
            stddev = 2 / np.sqrt(n_inputs)
            init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
            # random stuff to initialize the neuron
            W = tf.Variable(init, name="kernel")
            b = tf.Variable(tf.zeros((1, n_neurons)), name="bias")
            Z = tf.matmul(X, W) + b
            return activation(Z) if activation is not None else Z

    def label_to_logit(self, label):
        """
        converts
        into
        :param label: [0, 3, 2, 1, ...]
        :return:     [[1, 0, 0, 0, ...],
                      [0, 0, 0, 1, ...],
                      [0, 0, 1, 0, ...],
                      [0, 1, 0, 0, ...]]
        """
        logit = zeros((4, len(label)))
        for i in range(len(label)):
            logit[int(label[i]), i] = 1
        return logit

    def run_on_data(self, data):
        print('run on data method')

    def init_diarization_method(self, params):
        self.layer_counts = params['layers']
        self.learning_rate = params['learning_rate']
        self.window_size = self.layer_counts[0]

        layer = self.input_neurons = tf.placeholder(tf.float32, shape=(1, self.layer_counts[0] * 2))

        for i in range(1, len(self.layer_counts)):
            layer = self.neuron_layer(layer, self.layer_counts[i], str('hidden%d' % i))

        self.output_layer = layer
        self.label = tf.placeholder(tf.float32)

        with tf.name_scope('loss'):
            xentropy = tf.square(self.label - self.output_layer)
            self.loss = tf.reduce_mean(xentropy, name='loss')

        with tf.name_scope('train'):
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        with tf.name_scope('eval'):
            correct = tf.nn.in_top_k(self.output_layer, tf.cast(self.label, tf.int32), 1)
            self.eval_op = tf.reduce_mean((tf.cast(correct, tf.float32)))

        self.saver = tf.train.Saver()

    def train_on_data(self, data, label):
        label = self.label_to_logit(label).T
        data = np.reshape(data, (1, -1))
        for i in range(len(label)):
            start = i * self.window_size * 2
            end = start + self.window_size * 2
            segment = data[:, start:end].reshape((1, -1))
            self.sess.run(self.train_op, feed_dict={self.input_neurons: segment, self.label: label[i, :]})

    def get_train_error(self, test_data, test_label):
        return self.eval_op.eval(feed_dict={self.input_neurons: test_data, self.label: test_label})

    def load(self, path):
        self.save.restore(self.sess, path)

    def save(self, path):
        self.saver.save(self.sess, path)
