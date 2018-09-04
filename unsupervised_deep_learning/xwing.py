import numpy as np
import pandas as pd
import tensorflow as tf
import plotly
import plotly.graph_objs as go
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class Layer(object):
    def __init__(self, n_in, n_out, activation_fn=tf.nn.sigmoid):
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=False,
            seed=None,
            dtype=tf.float32
        )

        self.w = tf.Variable(initializer((n_in, n_out)))
        self.b_in = tf.Variable(tf.ones((n_in)))
        self.b_out = tf.Variable(tf.ones((n_out)))
        self.activation_fn = activation_fn

    def forwardLogits(self, x):
        return tf.matmul(x, self.w) + self.b_out

    def forwardTLogits(self, x):
        return tf.matmul(x, tf.transpose(self.w)) + self.b_in

    def forward(self, x):
        # get logits
        z = self.forwardLogits(x)

        return self.activation_fn(z)

    def forwardT(self, x):
        # get logits
        z = self.forwardTLogits(x)

        return self.activation_fn(z)


class DeepXEncoder(object):
    def __init__(self, n_features, layer_dimensions, activation_fn):
        self.layer_dimensions = layer_dimensions
        self.layers = []

        # layers
        n_in = n_features
        for n_out in self.layer_dimensions:
            layer = Layer(
                n_in=n_in,
                n_out=n_out,
                activation_fn=activation_fn
            )
            self.layers.append(layer)
            n_in = n_out

        # data placeholder
        self.tfX = tf.placeholder(dtype=tf.float32, shape=(None, n_features), name="tfX")

        # cost function
        self.cost = tf.reduce_mean(
            tf.squared_difference(
                x=self.tfX,
                y=self.forward(self.tfX)
            )
        )

        # training step
        self.train_step = tf.train.AdamOptimizer().minimize(self.cost)

    def setSession(self, session):
        self.session = session

    def forwardMiddle(self, x):
        z = x
        for layer in self.layers:
            z = layer.forward(z)

        return z

    def forward(self, x):
        z = self.forwardMiddle(x)
        for layer in self.layers[:0:-1]:
            z = layer.forwardT(z)

        return self.layers[0].forwardTLogits(z)

    def predict(self, x):
        return self.session.run(
            self.forward(x),
            feed_dict={self.tfX: x}
        )

    def returnMiddle(self, x):
        return self.session.run(
            self.forwardMiddle(x),
            feed_dict={self.tfX: x}
        )

    def fit(self, x, n_epochs=3, batch_size=128, print_period=100, print_cost=True):
        # initialize global variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # number of batches
        n_batches = x.shape[0] // batch_size

        # storing the cost
        self.costs = []

        for i in range(n_epochs):
            x = shuffle(x)
            for j in range(n_batches):
                x_batch = x[batch_size * j: batch_size * (j + 1)]

                # perform training step
                self.session.run(
                    self.train_step,
                    feed_dict={self.tfX: x_batch}
                )

                if j % print_period == 0:
                    # save the cost at this epoch / batch
                    self.costs.append(
                        self.session.run(
                            self.cost,
                            feed_dict={self.tfX: x_batch}
                        )
                    )

                    print("Epoch: ", i + 1, "Batch: ", j, "Cost: ", self.costs[-1])

        if print_cost:
            g = go.Scatter(
                x=np.arange(len(self.costs)),
                y=self.costs,
            )

            plotly.offline.plot([g])


# load data
dt = pd.read_csv("large_files/mnist/train.csv")

# split into x and y
y = np.array(dt.label.values, dtype=np.float32)
x = np.array(dt.drop("label", axis=1), dtype=np.float32)
del dt

# normalize the pixels
x = x / 255

model = DeepXEncoder(x.shape[1], [500, 300, 3], activation_fn=tf.nn.sigmoid)
model.setSession(tf.Session())
model.fit(x, n_epochs=30)

# get the middle neurons
mapping = model.returnMiddle(x)
graphs = []
for i in np.unique(y):
    g = go.Scatter3d(
        x=mapping[y == i, 0],
        y=mapping[y == i, 1],
        z=mapping[y == i, 2],
        mode="markers",
        name="Number: %d" % i
    )

    graphs.append(g)

plotly.offline.plot(graphs)
