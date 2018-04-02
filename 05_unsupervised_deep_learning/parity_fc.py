# import standard libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly
import plotly.graph_objs as go
from sklearn.utils import shuffle

# load function for creating the parity pair data
import sys
sys.path.append("D:/Repos/lazy_programmer_ml_course/05_unsupervised_deep_learning")
from utilities import all_parity_pairs

class hiddenLayer(object):
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

class artificialNeuralNetwork(object):
    def __init__(self, hidden_layer_dimension, activation_fn):
        self.hidden_layer_dimensions = hidden_layer_dimension
        self.activation_fn = activation_fn
        self.layers = []

    def initializeLayers(self, n_features, n_classes):
        n_in = n_features

        # iterate through the hidden layer dimensions and create a list of hidden layer classes
        for i, n_out in enumerate(self.hidden_layer_dimensions):
            self.layers.append(
                hiddenLayer(n_in, n_out, i, self.activation_fn)
            )

            # the output dimension of layer i, is the input dimension of layer i+1
            n_in = n_out

        # add the final layer, which will give us an n-by-k matrix as the final result, where k is the number of classes
        # in the classification problem
        self.layers.append(hiddenLayer(n_in, n_classes, len(self.layers), self.activation_fn))

    def forwardLogits(self, x):
        z = x

        # iterate through all but the last layer to perform forward propagation
        for layer in self.layers[:-1]:
            z = layer.forward(z)

        # return logits
        return self.layers[-1].forwardLogits(z)

    def forwardProbabilities(self, x):
        return tf.nn.softmax(
            self.forwardLogits(x)
        )

    def predict(self, x):
        return tf.argmax(
            self.forwardProbabilities(x),
            axis=1
        )

    def returnPredictions(self, x):
        return self.session.run(
            self.predict(x),
            feed_dict={self.tfX: x}
        )

    def initializePlaceholders(self, x_shape, y_shape):
        self.tfX = tf.placeholder(
            dtype=tf.float32,
            shape=(None, *x_shape[1:]),
            name="tfX"
        )

        self.tfT = tf.placeholder(
            dtype=tf.float32,
            shape=(None, *y_shape[1:]),
            name="tfT"
        )

    def initializeCostAndTrain(self):
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.tfT,
                logits=self.forwardLogits(self.tfX)
            )
        )

        self.train_step = tf.train.AdamOptimizer().minimize(self.cost)

    def setSession(self, session):
        self.session = session

    def fit(self, x, y, session, n_epochs=10, batch_size=128, print_period=20, show_img=True, plot_title=""):
        # prerequisites
        self.setSession(session)
        self.initializeLayers(x.shape[1], y.shape[1])
        self.initializePlaceholders(x.shape, y.shape)
        self.initializeCostAndTrain()

        # initialize global variables
        initializer = tf.global_variables_initializer()
        self.session.run(initializer)

        # calculate the number of iterations based on the number of samples and batch size
        # also create lists for holding cost and accuracy
        n_iterations = x.shape[0] // batch_size
        self.costs = []
        self.accuracy = []

        # trace
        print("Starting training:\n n_epochs: ", n_epochs, "n_iterations: ", n_iterations, "print_period: ", print_period)

        for i in range(n_epochs):
            # introduce randomness into the data so we don't have periodicity when doing SGD
            x, y = shuffle(x, y)
            for j in range(n_iterations):
                x_batch = x[j*batch_size: (j+1)*batch_size]
                y_batch = y[j*batch_size: (j+1)*batch_size]

                # perform training step
                self.session.run(
                    self.train_step,
                    feed_dict={self.tfX: x_batch, self.tfT: y_batch}
                )

                if j % print_period == 0:
                    self.costs.append(
                        self.session.run(
                            self.cost,
                            feed_dict={self.tfX: x_batch, self.tfT: y_batch}
                        )
                    )

                    self.accuracy.append(
                        np.mean(np.argmax(y_batch, axis=1) == self.returnPredictions(x_batch))
                    )

                    print("Epoch: ", i, "Step: ", j, "Cost: ", self.costs[-1], "Accuracy: ", self.accuracy[-1])

        if show_img:
            g1 = go.Scatter(
                x=np.linspace(1, len(self.costs), len(self.costs)),
                y=self.costs,
                name="Cost"
            )

            g2 = go.Scatter(
                x=np.linspace(1, len(self.costs), len(self.costs)),
                y=self.accuracy,
                name="Accuracy"
            )

            figure = plotly.tools.make_subplots(2, 1, True, print_grid=False, subplot_titles=("Cost", "Accuracy"))
            figure.append_trace(g1, 1, 1)
            figure.append_trace(g2, 2, 1)
            figure["layout"].update(title=plot_title)

            plotly.offline.plot(figure)

x, y = all_parity_pairs(12)
y = pd.get_dummies(y).as_matrix().astype(dtype=np.float32)

# wide ANN
wide_ann_width = 1024
wideANN = artificialNeuralNetwork([wide_ann_width], activation_fn=tf.nn.relu)
wideANN.fit(
    x,
    y,
    tf.Session(),
    n_epochs=100,
    batch_size=200,
    plot_title="Wide ANN: single hidden layer with %d neurons" % wide_ann_width)

# deep ANN
deep_ann_depth = 5
deep_ann_width = 50
deepANN = artificialNeuralNetwork([deep_ann_width] * deep_ann_depth, activation_fn=tf.nn.relu)
deepANN.fit(
    x,
    y,
    tf.Session(),
    n_epochs=150,
    batch_size=200,
    plot_title="Deep ANN: %d hidden layers with %d neurons each" % (deep_ann_depth, deep_ann_width)
)

wide_ann_n_params = np.sum([
    np.prod(wideANN.session.run(layer.w).shape) +
    np.prod(wideANN.session.run(layer.b).shape)
    for layer in wideANN.layers
])
deep_ann_n_params = np.sum([
    np.prod(deepANN.session.run(layer.w).shape) +
    np.prod(deepANN.session.run(layer.b).shape)
    for layer in deepANN.layers
])
