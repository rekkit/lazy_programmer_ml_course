import numpy as np
import tensorflow as tf
import plotly
import plotly.graph_objs as go
from sklearn.utils import shuffle
from functools import reduce

# load simple hidden layer from dl_layers.py
from dl_layers import hiddenLayer, recurrentLayer, embeddingLayer

# load function for creating the parity pair data
import sys
sys.path.append("D:/Repos/lazy_programmer_ml_course/05_unsupervised_deep_learning")
from utilities import get_robert_frost

class recurrentNeuralNetwork(object):
    def __init__(self, embedding_space_dim, hidden_layer_dimensions, activation_fn):
        self.embedding_space_dim = embedding_space_dim
        self.hidden_layer_dimensions = hidden_layer_dimensions
        self.activation_fn = activation_fn
        self.layers = []

    def initializeLayers(self, vocabulary_dim):
        self.vocabulary_dim = vocabulary_dim

        # insert embedding layer
        self.layers.append(
            embeddingLayer(vocabulary_size=self.vocabulary_dim, embedding_space_dim=self.embedding_space_dim, layer_id=0)
        )

        n_in = self.embedding_space_dim

        # go through the hidden layer dimensions and initialize the recurrent layers
        for i, n_out in enumerate(self.hidden_layer_dimensions, start=len(self.layers)):
            self.layers.append(
                recurrentLayer(n_in, n_out, i, self.activation_fn)
            )

            n_in = n_out

        # add the final layer which is a fully connected layer
        self.layers.append(
            hiddenLayer(n_in, vocabulary_dim, len(self.layers)+1, self.activation_fn)
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

    def initializePlaceholders(self):
        self.tfX = tf.placeholder(
            dtype=np.int32,
            shape=(None, ),
            name="tfX"
        )

        self.tfT = tf.placeholder(
            dtype=np.int32,
            shape=(None, self.vocabulary_dim),
            name="tfT"
        )

    def initializeCostAndTrain(self):
        self.cost = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.tfT,
                logits=self.forwardLogits(self.tfX)
            )
        )

        self.train_step = tf.train.AdamOptimizer().minimize(self.cost)

    def setSession(self, session):
        self.session = session

    def fit(self, x, session, n_epochs, print_step=20, show_fig=True, output_fig=False):
        # the input is a list of lists (we have sentences which vary in length, hence it can't be a matrix)
        # in order to get the vocabulary size, I flatten the list and get the number of unique values
        vocabulary_dim = len(
                np.unique(
                    reduce(lambda left, right: left + right, sentences)
                )
            ) + 2  # +2 because of START, END

        # initialize layers using the previously calculated value
        self.initializeLayers(
            vocabulary_dim=vocabulary_dim
        )

        # initialize placeholders, cost function and training step
        self.initializePlaceholders()
        self.initializeCostAndTrain()

        # set the session to be used an initialize variables
        self.setSession(session)
        init = tf.global_variables_initializer()
        self.session.run(init)

        # create lists to store the cost / accuracy values for each epoch
        self.costs = []
        self.accuracy = []

        for i in range(n_epochs):
            x = shuffle(x)

            # create constants to hold the values for each 'batch', i.e. sentence
            j_cost = 0
            n_correct = 0
            n_words = 0

            for j in range(len(x)):
                # x_batch (input): [START, w1, w2, ..., wn]
                # y_batch (target): [w1, w2, ..., END]
                # in our word to word_id dictionary we have that {"START": 0, "END": 1}
                x_batch = [0] + x[j]
                y_batch = np.zeros((len(x_batch), vocabulary_dim), dtype=np.int32)

                # add ones to create the dummy matrix
                for k, e in enumerate(x[j]):
                    y_batch[k, e] = 1

                y_batch[-1, 1] = 1

                # run training step and get the cost value for the particular sentence
                _, sentence_cost = session.run(
                    (self.train_step, self.cost),
                    feed_dict={self.tfX: x_batch, self.tfT: y_batch}
                )

                j_cost += sentence_cost
                n_correct += np.sum(np.argmax(y_batch, axis=1) == self.returnPredictions(x_batch))
                n_words += len(x[j])

                # if j % print_step == 0:
                #     self.costs.append(
                #         self.session.run(
                #             self.cost,
                #             feed_dict={self.tfX: x_batch, self.tfT: y_batch}
                #         )
                #     )
                #
                #     self.accuracy.append(
                #         np.mean(
                #             np.argmax(y_batch, axis=1) == self.returnPredictions(x_batch)
                #         )
                #     )

                if j % print_step == 0:
                    print(
                        "Epoch:", i,
                        "Step %d of %d completed." % (j, len(x)),
                        "Cost:", j_cost / n_words,
                        "Accuracy:", n_correct / n_words
                    )

            self.costs.append(j_cost / n_words)
            self.accuracy.append(n_correct / n_words)

            print("Epoch:", i, "Cost:", self.costs[-1], "Accuracy:", self.accuracy[-1])

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
            plotly.offline.plot(figure, image="png", image_filename="poetry_generator_rnn")

        else:
            plotly.offline.plot(figure)

sentences, word2idx = get_robert_frost()

rnn = recurrentNeuralNetwork(128, [64, 64], tf.nn.relu)
rnn.fit(sentences, session=tf.Session(), n_epochs=20, print_step=100)
