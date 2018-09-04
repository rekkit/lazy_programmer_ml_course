import numpy as np
import tensorflow as tf
import plotly
import plotly.graph_objs as go
from sklearn.utils import shuffle

# load simple hidden layer from dl_layers.py
from dl_layers import hiddenLayer, recurrentLayer

# load function for creating the parity pair data
import sys
sys.path.append("D:/Repos/lazy_programmer_ml_course/05_unsupervised_deep_learning")
from utilities import all_parity_pairs_with_sequence_labels

class recurrentNeuralNetwork(object):
    def __init__(self, hidden_layer_dimensions, activation_fn):
        self.hidden_layer_dimensions = hidden_layer_dimensions
        self.activation_fn = activation_fn
        self.layers = []

    def initializeLayers(self, n_features, n_classes):
        n_in = n_features

        for i, n_out in enumerate(self.hidden_layer_dimensions):
            self.layers.append(
                recurrentLayer(n_in, n_out, i, self.activation_fn)
            )

            n_in = n_out

        # add the final layer which is a fully connected layer
        self.layers.append(
            hiddenLayer(n_in, n_classes, i+1, self.activation_fn)
        )

    def forwardLogits(self, x):
        z = x
        for layer in self.layers[:-1]:
            z = layer.forward(z)

        return self.layers[-1].forwardLogits(z)

    def predict(self, x):
        z = self.forwardLogits(x)
        z = tf.nn.softmax(z)

        return tf.argmax(z, axis=1)

    def returnPredictions(self, x):
        return self.session.run(
            self.predict(x),
            feed_dict={self.tfX: x}
        )

    def initializePlaceholders(self, x_shape, y_shape):
        self.tfX = tf.placeholder(
            dtype=np.float32,
            shape=x_shape,
            name="tfX"
        )

        self.tfT = tf.placeholder(
            dtype=np.float32,
            shape=y_shape,
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

    def fit(self, x, y, session, n_epochs, print_step=20, show_fig=True, output_fig=False):
        # initialize layers
        self.initializeLayers(
            x.shape[2],
            y.shape[2]
        )

        # initialize placeholders, cost function and training step
        self.initializePlaceholders(x.shape[1:], y.shape[1:])
        self.initializeCostAndTrain()

        # set the session to be used an initialize variables
        self.setSession(session)
        init = tf.global_variables_initializer()
        self.session.run(init)

        # create list to store the cost values
        self.costs = []
        self.accuracy = []

        for i in range(n_epochs):
            x, y = shuffle(x, y)

            for j in range(x.shape[0]):
                x_batch = x[j, ]
                y_batch = y[j, ]

                session.run(
                    self.train_step,
                    feed_dict={self.tfX: x_batch, self.tfT: y_batch}
                )

                if j % print_step == 0:
                    self.costs.append(
                        self.session.run(
                            self.cost,
                            feed_dict={self.tfX: x_batch, self.tfT: y_batch}
                        )
                    )

                    self.accuracy.append(
                        np.mean(
                            np.argmax(y_batch, axis=1) == self.returnPredictions(x_batch)
                        )
                    )

                    print("Epoch:", i, "Time step:", j, "Cost:", self.costs[-1], "Accuracy:", self.accuracy[-1])

        if show_fig:
            self.plotMetrics(output_fig=output_fig)

    def plotMetrics(self, output_fig):
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

        if output_fig:
            plotly.offline.plot(figure, image="png", image_filename="parity_rnn_cost")

        else:
            plotly.offline.plot(figure)

x, y = all_parity_pairs_with_sequence_labels(12)
y_new = np.zeros((*y.shape, 2))
for i in range(y.shape[0]):
    y_new[i] = np.array([
        [1 if e == 0 else 0 for e in y[i]],
        y[i].reshape([-1])
    ]).T.astype(np.float32)

rnn = recurrentNeuralNetwork([32], tf.nn.relu)
rnn.fit(x, y_new, session=tf.Session(), n_epochs=1, print_step=500)
