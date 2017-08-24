#!/usr/bin/python
# -*- coding: utf-8 -*-

from data_util import *
import tensorflow as tf

class LMConfig(object):
    """Configuration of language model"""
    batch_size = 64
    num_steps = 20
    stride = 3

    embedding_dim = 64
    hidden_dim = 128
    num_layers = 2

    learning_rate = 0.05
    dropout = 0.2

class PTBInput(object):
    def __init__(self, config, data):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.vocab_size = config.vocab_size
        self.input_data, self.targets = ptb_producer(data, \
            self.vocab_size, batch_size, num_steps)
        self.batch_len = self.input_data.shape[0]
        self.cur_batch = 0

    def next_batch(self):
        x = self.input_data[self.cur_batch]
        y = self.targets[self.cur_batch]

        y_ = np.zeros((y.shape[0], self.vocab_size), dtype=np.bool)
        for i in range(y.shape[0]):
            y_[i][y[i]] = 1

        self.cur_batch = (self.cur_batch +1) % self.batch_len

        return x, y_


class PTBModel(object):
    def __init__(self, config, is_training=True):

        self.num_steps = config.num_steps
        self.vocab_size = config.vocab_size

        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        self.learning_rate = config.learning_rate
        self.dropout = config.dropout

        self.init_variables()
        self.model()
        self.cost()
        self.optimize()
        self.error()


    def init_variables(self):
        self._inputs = tf.placeholder(tf.int32, [None, self.num_steps])
        self._targets = tf.placeholder(tf.int32, [None, self.vocab_size])


    def get_input_embedding(self):
        # Embedding data
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [self.vocab_size,
                    self.embedding_dim], dtype=tf.float32)
            _inputs = tf.nn.embedding_lookup(embedding, self._inputs)

        return _inputs


    def model(self):
        def lstm_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)
            return tf.contrib.rnn.DropoutWrapper(cell,
                output_keep_prob=self.dropout)

        def gru_cell():
            cell = tf.contrib.rnn.GRUCell(self.hidden_dim)
            return tf.contrib.rnn.DropoutWrapper(cell,
                output_keep_prob=self.dropout)

        _inputs = self.get_input_embedding()

        cell = gru_cell()
        cells = [gru_cell() for _ in range(self.num_layers)]
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        _outputs, state = tf.nn.dynamic_rnn(cell=cell,
            inputs=_inputs, dtype=tf.float32)

        _outputs = tf.transpose(_outputs, [1, 0, 2])
        last = tf.gather(_outputs, int(_outputs.get_shape()[0]) - 1)

        logits = tf.layers.dense(last, self.vocab_size)
        prediction = tf.nn.softmax(logits)

        self._logits = logits
        self._pred = prediction

    def cost(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=self._logits, labels=self._targets)
        cost = tf.reduce_mean(cross_entropy)
        self.cost = cost

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optim = optimizer.minimize(self.cost)

    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self._targets, 1), tf.argmax(self._pred, 1))
        self.errors = tf.reduce_mean(tf.cast(mistakes, tf.float32))


config = LMConfig()

train_data, valid_data, test_data, words, word_to_id = \
    ptb_raw_data('simple-examples/data')
config.vocab_size = len(words)

input_train = PTBInput(config, train_data)

model = PTBModel(config)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(input_train.batch_len)
for epoch in range(5):
    for i in range(input_train.batch_len):
        x, y = input_train.next_batch()

        feed_dict_train = {model._inputs: x, model._targets: y}
        sess.run(model.optim, feed_dict=feed_dict_train)
        if i % 1000 == 0:
            cost = sess.run(model.cost, feed_dict_train)
            print(i, cost)
            pred = sess.run(model._pred, feed_dict={model._inputs: x, model._targets: y})
            word_ids = sess.run(tf.argmax(pred, 1))
            print(' '.join(words[w] for w in word_ids))
            true_ids = np.argmax(y, 1)
            print(' '.join(words[w] for w in true_ids))

sess.close()
