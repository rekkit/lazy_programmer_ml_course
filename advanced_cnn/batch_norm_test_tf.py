import plotly
import plotly.graph_objs as go

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from dl_layers import batchNormalizationLayer

x = np.random.randn(10000) * 2 + 5
x_normed = (x - 5) / np.sqrt(2)

# placeholders
tfX = tf.placeholder(shape=(None, ), dtype=tf.float32)
tfT = tf.placeholder(shape=(None, ), dtype=tf.float32)

# define what forward is
layer = batchNormalizationLayer(n_channels=1, axes=[0])
forward_op = layer.forward(tfX, True)
forward_test = layer.forward(tfX, False)

# training and cost
cost = tf.reduce_sum(tf.square(forward_op - tfT))
train_op = tf.train.AdamOptimizer().minimize(cost)

# create session
sess = tf.Session()
init = tf.global_variables_initializer()

# define batch size and the number of epochs we want to do
batch_size = 50
n_epochs = 40

# initialize variables
sess.run(init)

for i in range(n_epochs):
    # shuffle the arrays
    x, x_normed = shuffle(x, x_normed)

    for j in range(x.shape[0] // batch_size):
        x_batch = x[j*batch_size: (j+1)*batch_size]
        x_normed_batch = x_normed[j*batch_size: (j+1)*batch_size]

        _, c = sess.run(
            (train_op, cost),
            feed_dict={tfX: x_batch, tfT: x_normed_batch}
        )

        print("Epoch: %d. Step: %d. Cost: %.3f" % (i, j, c))

# norm the original data using the learned function
x_normed_tf = sess.run(
    forward_test,
    feed_dict={tfX: x}
)

# plot against the original distribution
g1 = go.Histogram(x=x_normed_tf, opacity=0.7, name="TF Normed Data")
g2 = go.Histogram(x=x_normed, opacity=0.7, name="Deterministically Normed Data")

layout = go.Layout(barmode="overlay")
figure = go.Figure(data=[g1, g2], layout=layout)

# plot the figure
plotly.offline.plot(figure)
