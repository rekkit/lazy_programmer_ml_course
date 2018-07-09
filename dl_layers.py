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

class ratedRecurrentLayer(recurrentLayer):
    with tf.variable_scope("rru_layer"):
        def __init__(self, n_in, n_out, layer_id, activation_fn=tf.nn.relu):
            recurrentLayer.__init__(self, n_in, n_out, layer_id, activation_fn)

            self.z0 = tf.Variable(
                np.ones(n_out, dtype=np.float32),
                name="z0_%d" % self.layer_id
            )

            self.z1 = tf.Variable(
                np.ones(n_out, dtype=np.float32),
                name="z1_%d" % self.layer_id
            )

        def recurrence(self, ht_1, xt):
            """
            This is the function by which recurrence is defined.
            :param ht_1: The value of the hidden layer h at time step t-1.
            :param xt: The value of the input at time step t.
            :return: The value of the hidden layer h at time step t.
            """
            ht = self.z0 * ht_1 + self.z1 * self.activation_fn(
                tf.matmul(tf.reshape(xt, (1, self.n_in)), self.wx) +
                tf.matmul(tf.reshape(ht_1, (1, self.n_out)), self.wh) + self.bh
            )

            return tf.reshape(ht, (self.n_out,))

class gatedRecurrentLayer(ratedRecurrentLayer):
    with tf.variable_scope("gru_layer"):
        def __init__(self, n_in, n_out, layer_id, activation_fn=tf.nn.relu):
            ratedRecurrentLayer.__init__(self, n_in, n_out, layer_id, activation_fn)

            self.r = tf.Variable(
                np.ones(shape=n_out, dtype=np.float32),
                name="r_%d" % layer_id
            )

        def recurrence(self, ht_1, xt):
            """
            This is the function by which recurrence is defined.
            :param ht_1: The value of the hidden layer h at time step t-1.
            :param xt: The value of the input at time step t.
            :return: The value of the hidden layer h at time step t.
            """
            ht = self.z0 * ht_1 + self.z1 * self.activation_fn(
                tf.matmul(tf.reshape(xt, (1, self.n_in)), self.wx) +
                tf.matmul(self.r * tf.reshape(ht_1, (1, self.n_out)), self.wh) + self.bh
            )

            return tf.reshape(ht, (self.n_out,))

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

class batchNormalizationLayer(object):
    def __init__(self, n_channels, axes=[0, 1, 2], beta=0.9):
        self.n_channels = n_channels
        self.mean = tf.Variable(np.zeros(self.n_channels), dtype=tf.float32, trainable=False)
        self.var = tf.Variable(np.ones(self.n_channels), dtype=tf.float32, trainable=False)
        self.offset = tf.Variable(np.zeros(self.n_channels), dtype=tf.float32)
        self.scale = tf.Variable(np.ones(self.n_channels), dtype=tf.float32)
        self.axes = axes
        self.beta = beta

    def forward(self, x, is_training):
        if is_training:
            # calculate the batch mean and variance
            batch_mean, batch_var = tf.nn.moments(
                x,
                axes=self.axes
            )

            # calculate the exponential weighted moving average of the mean and variance
            self.mean = self.beta * self.mean + (1 - self.beta) * batch_mean
            self.var = self.beta * self.var + (1 - self.beta) * batch_var

        # normalize the input
        return tf.nn.batch_normalization(
            x,
            self.mean,
            self.var,
            self.offset,
            self.scale,
            variance_epsilon=0.00001
        )

class convolutionalLayer(object):
    def __init__(
        self,
        filter_h,
        filter_w,
        maps_in,
        maps_out,
        layer_id,
        with_bias=True,
        activation_fn=tf.nn.relu,
        stride_horizontal=1,
        stride_vertical=1,
        padding="VALID"
    ):
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.maps_in = maps_in
        self.maps_out = maps_out
        self.layer_id = layer_id
        self.with_bias = with_bias
        self.activation_fn = activation_fn
        self.stride_horizontal = stride_horizontal
        self.stride_vertical = stride_vertical
        self.padding = padding

        # inititalize the embedding matrix
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=False,
            dtype=tf.float32
        )

        # print("filter_h: ", filter_h, "filter_w: ", filter_w, "maps_in: ", maps_in, "maps_out: ", maps_out)
        self.w = tf.Variable(
            initializer((self.filter_h, self.filter_w, self.maps_in, self.maps_out)),
            name="conv_w_%d" % self.layer_id
        )

        if self.with_bias:
            self.b = tf.Variable(
                np.zeros(self.maps_out),
                dtype=tf.float32,
                name="conv_b_%d" % self.layer_id
            )

    def forwardLogits(self, x):
        z = tf.nn.conv2d(
            x,
            filter=self.w,
            strides=[1, self.stride_horizontal, self.stride_vertical, 1],
            padding=self.padding
        )

        if self.with_bias:
            z = z + self.b

        return z

    def forward(self, x):
        z = self.forwardLogits(x)

        return self.activation_fn(z)

class resNetSubLayer(object):
    def __init__(
        self,
        filter_h,
        filter_w,
        maps_in,
        maps_out,
        layer_id,
        activation_fn,
        stride_horizontal,
        stride_vertical,
        axes=[0, 1, 2],
        beta=0.9
    ):
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.maps_in = maps_in
        self.maps_out = maps_out
        self.layer_id = layer_id
        self.activation_fn = activation_fn
        self.stride_horizontal = stride_horizontal
        self.stride_vertical = stride_vertical
        self.axes = axes
        self.beta = beta

        # convolutional layer
        self.conv_layer = convolutionalLayer(
            filter_h=self.filter_h,
            filter_w=self.filter_w,
            maps_in=self.maps_in,
            maps_out=self.maps_out,
            layer_id=self.layer_id,
            with_bias=False,
            stride_horizontal=self.stride_horizontal,
            stride_vertical=self.stride_vertical,
            padding="SAME"
        )

        # batch normalization layer
        self.batch_norm_layer = batchNormalizationLayer(
            n_channels=self.maps_out,
            axes=self.axes,
            beta=self.beta
        )

    def forwardLogits(self, x, is_training):
        # get the convolutional layer
        z = self.conv_layer.forwardLogits(x)

        # return after performing batch normalization
        return self.batch_norm_layer.forward(z, is_training)

    def forward(self, x, is_training):
        # get the output after applying convolution and batch normalization
        return self.activation_fn(
            self.forwardLogits(x, is_training)
        )

class resNetLayer(object):
    def __init__(
        self,
        filter_height,
        filter_width,
        maps_in,
        maps_out,
        layer_id,
        activation_fn=tf.nn.relu,
        stride_horizontal=[1, 1, 1],
        stride_vertical=[1, 1, 1],
        beta=[0.9, 0.9, 0.9]
    ):
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.maps_in = maps_in
        self.maps_out = maps_out
        self.layer_id = layer_id
        self.activation_fn = activation_fn
        self.stride_horizontal = stride_horizontal
        self.stride_vertical = stride_vertical
        self.beta = beta

        # we have 3 sub-layers in the resNet layer
        self.layer_1 = resNetSubLayer(
            filter_h=self.filter_height[0],
            filter_w=self.filter_width[0],
            maps_in=self.maps_in,
            maps_out=self.maps_out[0],
            layer_id=self.layer_id,
            activation_fn=self.activation_fn,
            stride_horizontal=self.stride_horizontal[0],
            stride_vertical=self.stride_vertical[0],
            beta=self.beta[0]
        )

        self.layer_2 = resNetSubLayer(
            filter_h=self.filter_height[1],
            filter_w=self.filter_width[1],
            maps_in=self.maps_out[0],
            maps_out=self.maps_out[1],
            layer_id=self.layer_id,
            activation_fn=self.activation_fn,
            stride_horizontal=self.stride_horizontal[1],
            stride_vertical=self.stride_vertical[1],
            beta=self.beta[1]
        )

        self.layer_3 = resNetSubLayer(
            filter_h=self.filter_height[2],
            filter_w=self.filter_width[2],
            maps_in=self.maps_out[1],
            maps_out=self.maps_in,
            layer_id=self.layer_id,
            activation_fn=self.activation_fn,
            stride_horizontal=self.stride_horizontal[2],
            stride_vertical=self.stride_vertical[2],
            beta=self.beta[2]
        )

    def forwardResidual(self, x, is_training):
        z = self.layer_1.forward(x, is_training)
        z = self.layer_2.forward(z, is_training)

        return self.layer_3.forwardLogits(z, is_training)

    def forwardLogits(self, x, is_training):
        return self.forwardResidual(x, is_training) + x

    def forward(self, x, is_training):
        return self.activation_fn(
            self.forwardLogits(x, is_training)
        )


