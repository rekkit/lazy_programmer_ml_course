import numbers
import numpy as np
import tensorflow as tf

class hiddenLayer(object):
    with tf.variable_scope("hidden_layer"):
        def __init__(self, n_out, layer_id, activation_fn, flatten_input=True):
            self.n_in = None
            self.n_out = n_out
            self.layer_id = layer_id
            self.activation_fn = activation_fn
            self.flatten_input = flatten_input
            self.input_dimensions = None
            
        def appendIn(self, input_dimensions):
            self.input_dimensions = input_dimensions

            if self.flatten_input:
                self.n_in = np.prod(self.input_dimensions)
            else:
                if len(input_dimensions) == 1:
                    self.n_in = self.input_dimensions[0]
                else:
                    raise ValueError("If flatten_input is False a one dimensional input needs to be passed.")

            # create initializer and initialize weights and biases
            initializer = tf.contrib.layers.xavier_initializer(
                uniform=False,
                dtype=tf.float32
            )

            self.w = tf.Variable(
                initializer((self.n_in, self.n_out)),
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
            x = tf.reshape(x, [-1, self.n_in])

            return tf.matmul(x, self.w) + self.b

        def forward(self, x):
            """
            :param x: The input to the hidden layer.
            :return: The values after performing forward propagation in this layer.
            """
            x = tf.reshape(x, [-1, self.n_in])

            return self.activation_fn(
                self.forwardLogits(x)
            )

        def getOutputDimensions(self):
            return (self.n_out, )

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

            # perform the matrix multiplication ahead of time so that we don't have to multiply each time step with w
            xw = tf.matmul(x, self.wx)

            return tf.scan(
                fn=self.recurrence,
                elems=xw,
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
        self.maps_in = None
        self.maps_out = maps_out
        self.layer_id = layer_id
        self.with_bias = with_bias
        self.activation_fn = activation_fn
        self.stride_horizontal = stride_horizontal
        self.stride_vertical = stride_vertical
        self.padding = padding
        self.input_dimensions = None
        self.w = None
        self.b = None

    def appendIn(self, input_dimensions):
        self.input_dimensions = input_dimensions
        self.maps_in = input_dimensions[2]

        # inititalize the embedding matrix
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=False,
            dtype=tf.float32
        )

        # print("filter_h: ", self.filter_h, "filter_w: ", self.filter_w, "maps_in: ", self.maps_in, "maps_out: ", self.maps_out)
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

    def getOutputDimensions(self):
        # sanity checks
        if not isinstance(self.input_dimensions, tuple) or not len(self.input_dimensions) == 3:
            raise TypeError("Please input a tuple of the form (input_height, input_width, n_channels)")

        output_h, output_w, n_channels = self.input_dimensions
        output_h = int((output_h - self.filter_w) / self.stride_vertical + 1) if self.padding == "VALID" else output_h
        output_w = int((output_w - self.filter_w) / self.stride_horizontal + 1) if self.padding == "VALID" else output_w

        return output_h, output_w, self.maps_out


class resNetSubLayer(object):
    def __init__(
        self,
        filter_h,
        filter_w,
        input_dimensions,
        maps_out,
        layer_id,
        activation_fn,
        stride_horizontal,
        stride_vertical,
        padding="SAME",
        axes=[0, 1, 2],
        beta=0.9
    ):
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.input_dimensions = input_dimensions
        self.maps_out = maps_out
        self.layer_id = layer_id
        self.activation_fn = activation_fn
        self.stride_horizontal = stride_horizontal
        self.stride_vertical = stride_vertical
        self.padding = padding
        self.axes = axes
        self.beta = beta

        # convolutional layer
        self.conv_layer = convolutionalLayer(
            filter_h=self.filter_h,
            filter_w=self.filter_w,
            maps_out=self.maps_out,
            layer_id=self.layer_id,
            with_bias=False,
            stride_horizontal=self.stride_horizontal,
            stride_vertical=self.stride_vertical,
            padding=self.padding
        )

        self.conv_layer.appendIn(self.input_dimensions)

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

    def getOutputDimensions(self):
        return self.conv_layer.getOutputDimensions()

class resNetLayer(object):
    def __init__(
        self,
        filter_height,
        filter_width,
        layer_id,
        activation_fn=tf.nn.tanh,
        downsample=False,
        horizontal_downsample_step=2,
        vertical_downsample_step=2,
        beta=0.9
    ):
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.maps_in = None
        self.maps_out = None
        self.layer_id = layer_id
        self.activation_fn = activation_fn
        self.downsample = downsample
        self.stride_horizontal = horizontal_downsample_step if self.downsample else 1
        self.stride_vertical = vertical_downsample_step if self.downsample else 1
        self.padding = "VALID" if self.downsample else "SAME"
        self.beta = beta

        # The three layers that make up the residual block
        self.layer_1 = None
        self.layer_2 = None
        self.layer_3 = None

    def appendIn(self, input_dimensions):
        self.input_dimensions = input_dimensions
        self.maps_in = input_dimensions[2]
        self.maps_out = 2 * self.maps_in if self.downsample else self.maps_in

        # we have 3 sub-layers in the resNet layer
        self.layer_1 = resNetSubLayer(
            filter_h=self.filter_height,
            filter_w=self.filter_width,
            input_dimensions=self.input_dimensions,
            maps_out=self.maps_out,
            layer_id=self.layer_id,
            activation_fn=self.activation_fn,
            stride_horizontal=self.stride_horizontal,
            stride_vertical=self.stride_vertical,
            padding=self.padding,
            beta=self.beta
        )

        self.layer_2 = resNetSubLayer(
            filter_h=self.filter_height,
            filter_w=self.filter_width,
            input_dimensions=self.layer_1.getOutputDimensions(),
            maps_out=self.maps_out,
            layer_id=self.layer_id,
            activation_fn=self.activation_fn,
            stride_horizontal=1,
            stride_vertical=1,
            padding="SAME",  # I know padding="SAME" is the default, I just want it to be explicit.
            beta=self.beta
        )

        self.layer_3 = resNetSubLayer(
            filter_h=self.filter_height,
            filter_w=self.filter_width,
            input_dimensions=self.layer_2.getOutputDimensions(),
            maps_out=self.maps_out,
            layer_id=self.layer_id,
            activation_fn=self.activation_fn,
            stride_horizontal=1,
            stride_vertical=1,
            padding="SAME",
            beta=self.beta
        )

    def forwardResidual(self, x, is_training):
        z = self.layer_1.forward(x, is_training)
        z = self.layer_2.forward(z, is_training)

        return self.layer_3.forwardLogits(z, is_training)

    def forwardLogits(self, x, is_training):
        if self.downsample:
            return self.forwardResidual(x, is_training)
        else:
            return self.forwardResidual(x, is_training) + x

    def forward(self, x, is_training):
        if self.downsample:
            return self.activation_fn(
                self.forwardResidual(x, is_training)
            )
        else:
            return self.activation_fn(
                self.forwardLogits(x, is_training)
            )

    def getOutputDimensions(self):
        return self.layer_3.getOutputDimensions()
