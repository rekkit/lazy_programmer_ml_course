import numpy as np
import tensorflow as tf

class hiddenLayer(object):
    with tf.variable_scope("hidden_layer"):
        def __init__(self, n_in, n_out, layer_id, activation_fn):
            self.n_in = n_in
            self.n_out = n_out
            self.layer_id = layer_id
            self.activation_fn = activation_fn

            # create initializer and initialize weights and biases
            initializer = tf.contrib.layers.xavier_initializer(
                uniform=False,
                dtype=tf.float32
            )

            self.w = tf.Variable(
                initializer((n_in, n_out)),
                name="w_%d" % self.layer_id
            )

            self.b = tf.Variable(
                np.zeros(self.n_out, dtype=np.float32),
                name="b_%d" % self.layer_id
            )

        def forwardLogits(self, x):
            """
            :param x: The input to the hidden layer.
            :return: The values after multiplying the input by the weights and adding the biases. These are the values
                     that are fed into the activation function.
            """
            return tf.matmul(x, self.w) + self.b

        def forward(self, x):
            """
            :param x: The input to the hidden layer.
            :return: The values after performing forward propagation in this layer.
            """
            return self.activation_fn(
                self.forwardLogits(x)
            )

class recurrentLayer(object):
    with tf.variable_scope("recurrent_layer"):
        def __init__(self, n_in, n_out, layer_id, activation_fn=tf.nn.relu):
            self.n_out = n_out
            self.n_in = n_in
            self.layer_id = layer_id
            self.activation_fn = activation_fn

            # initialize weights and biases
            initializer = tf.contrib.layers.xavier_initializer(
                uniform=False,
                dtype=tf.float32
            )

            self.wx = tf.Variable(
                initializer((self.n_in, self.n_out)),
                name="wx_%d" % self.layer_id
            )

            self.wh = tf.Variable(
                initializer((self.n_out, self.n_out)),
                name="wh_%d" % self.layer_id
            )

            self.bh = tf.Variable(
                np.zeros(self.n_out, dtype=np.float32),
                name="bh_%d" % self.layer_id
            )

            self.h0 = tf.Variable(
                np.zeros(self.n_out, dtype=np.float32),
                name="h0_%d" % self.layer_id
            )

        def recurrence(self, ht_1, xt):
            """
            This is the function by which recurrence is defined. It is the simplest possible recurrent unit.
            :param ht_1: The value of the hidden layer h at time step t-1.
            :param xt: The value of the input at time step t.
            :return: The value of the hidden layer h at time step t.
            """
            ht = self.activation_fn(
                tf.matmul(tf.reshape(xt, (1, self.n_in)), self.wx) + tf.matmul(tf.reshape(ht_1, (1, self.n_out)), self.wh) + self.bh
            )

            return tf.reshape(ht, (self.n_out,))

        def forward(self, x):
            """
            :param x: The input to the hidden layer.
            :return: The values after performing forward propagation in this layer.
            """
            return tf.scan(
                fn=self.recurrence,
                elems=x,
                initializer=self.h0
            )

class embeddingLayer(object):
    with tf.variable_scope("embedding_layer"):
        def __init__(self, vocabulary_size, embedding_space_dim, layer_id):
            self.vocabulary_size = vocabulary_size
            self.embedding_space_dim = embedding_space_dim

            # inititalize the embedding matrix
            initializer = tf.contrib.layers.xavier_initializer(
                uniform=False,
                dtype=tf.float32
            )

            self.we = tf.Variable(
                initializer((self.vocabulary_size, self.embedding_space_dim)),
                name="we_%d" % layer_id
            )

        def forward(self, x):
            return tf.nn.embedding_lookup(self.we, x)
