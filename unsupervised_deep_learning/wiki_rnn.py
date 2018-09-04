# add the folder where the utilities.py file is stored and import get_wikipedia data
import sys
import os
sys.path.append(os.getcwd().replace("\\", "/") + "/05_unsupervised_deep_learning/")
from utilities import get_wikipedia_data

# other imports
import numpy as np
import tensorflow as tf
import plotly
import plotly.graph_objs as go
from sklearn.utils import shuffle
from functools import reduce
from dl_layers import embeddingLayer, hiddenLayer, gatedRecurrentLayer, recurrentLayer


class recurrentNeuralNetwork(object):
    def __init__(self, embedding_space_dim, hidden_layer_dimensions, activation_fn, bias_multiplier=1, sampled_softmax=False, n_samples=100):
        self.embedding_space_dim = embedding_space_dim
        self.hidden_layer_dimensions = hidden_layer_dimensions
        self.activation_fn = activation_fn
        self.bias_multiplier = bias_multiplier
        self.sampled_softmax = sampled_softmax
        self.n_samples = n_samples
        self.layers = []

    def initializeLayers(self, vocabulary_dim, recurrent_layer_type):
        with tf.name_scope("initialize_layers"):
            self.vocabulary_dim = vocabulary_dim

            # insert embedding layer
            self.layers.append(
                embeddingLayer(vocabulary_size=self.vocabulary_dim, embedding_space_dim=self.embedding_space_dim, layer_id=0)
            )

            n_in = self.embedding_space_dim

            # go through the hidden layer dimensions and initialize the recurrent layers
            for i, n_out in enumerate(self.hidden_layer_dimensions, start=len(self.layers)):
                self.layers.append(
                    recurrent_layer_type(n_in, n_out, i, self.activation_fn, self.bias_multiplier)
                )

                n_in = n_out

            # add the final layer which is a fully connected layer
            self.layers.append(
                hiddenLayer(n_in, vocabulary_dim, len(self.layers), self.activation_fn)
            )

    def forwardEmbeddings(self, x):
        with tf.name_scope("forward_embeddings"):
            return self.layers[0].forward(x)

    def forwardHiddenLayers(self, x):
        with tf.name_scope("forward_hidden_layers"):
            z = x
            for layer in self.layers[:-1]:
                z = layer.forward(z)

            return z

    def forwardLogits(self, x):
        with tf.name_scope("forward_logits"):
            z = self.forwardHiddenLayers(x)

            return self.layers[-1].forwardLogits(z)

    def forwardProbabilities(self, x):
        with tf.name_scope("forward_probabilities"):
            z = self.forwardLogits(x)

            return tf.nn.softmax(z)

    def predict(self, x):
        with tf.name_scope("predict"):
            z = self.forwardProbabilities(x)

            return tf.argmax(z, axis=1)

    def initializePlaceholders(self):
        with tf.name_scope("initialize_placeholders"):
            self.tfX = tf.placeholder(
                dtype=np.int32,
                shape=(None, ),
                name="tfX"
            )

            self.tfT = tf.placeholder(
                dtype=np.int32,
                shape=(None, ),
                name="tfT"
            )

    def initializeCost(self, n_classes):
        if self.sampled_softmax:
            self.cost = tf.reduce_sum(
                tf.nn.sampled_softmax_loss(
                    weights=tf.transpose(self.layers[-1].w),
                    biases=self.layers[-1].b,
                    labels=tf.reshape(self.tfT, (-1, 1)),
                    inputs=self.forwardHiddenLayers(self.tfX),
                    num_sampled=self.n_samples,
                    num_classes=n_classes
                )
            )
        else:
            self.cost = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.one_hot(self.tfT, n_classes),
                    logits=self.forwardLogits(self.tfX)
                )
            )

    def initializeOperations(self):
        # train operation
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

        # return probabilities
        self.return_probs_op = self.forwardProbabilities(self.tfX)

        # return predictions
        self.predict_op = self.predict(self.tfX)

        # get embeddings operation
        self.return_embeddings_op = self.forwardEmbeddings(self.tfX)

    def setSession(self, session):
        self.session = session

    def getEmbeddings(self, x):
        return self.session.run(
            self.return_embeddings_op,
            feed_dict={self.tfX: x}
        )

    def fit(self, x, session, n_epochs, recurrent_layer_type, print_step=20, show_fig=True, end_prob=0.1, output_fig=False):
        # set the session to be used
        self.setSession(session)

        # the input is a list of lists (we have sentences which vary in length, hence it can't be a matrix)
        # in order to get the vocabulary size, I flatten the list and get the number of unique values
        vocabulary_dim = len(
                np.unique(
                    reduce(lambda left, right: left + right, x)
                )
            ) + 2  # +2 because of START, END

        # initialize layers using the previously calculated value
        self.initializeLayers(
            vocabulary_dim=vocabulary_dim,
            recurrent_layer_type=recurrent_layer_type
        )

        # initialize placeholders, cost function and operations
        self.initializePlaceholders()
        self.initializeCost(
            n_classes=vocabulary_dim
        )
        self.initializeOperations()

        # initialize global variables
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
                y_batch = x[j] + [1]

                # run training step and get the cost value for the particular sentence
                _, sentence_cost, predictions = self.session.run(
                    (self.train_op, self.cost, self.predict_op),
                    feed_dict={self.tfX: x_batch, self.tfT: y_batch}
                )

                j_cost += sentence_cost
                n_correct += np.sum(y_batch == predictions)
                n_words += len(y_batch)

                if j % print_step == 0 and j > 0:
                    print(
                        "Epoch: %d." % i,
                        "Step %d of %d completed." % (j, len(x)),
                        "Cost: %.2f." % (j_cost / n_words),
                        "Accuracy: %.2f." % (n_correct / n_words)
                    )

            self.costs.append(j_cost / n_words)
            self.accuracy.append(n_correct / n_words)

        if show_fig:
            self.plotMetrics(output_fig=output_fig)

    def plotMetrics(self, output_fig=False):
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
            plotly.offline.plot(
                figure,
                image="png",
                image_filename="poetry_generator_rnn"
             )

        else:
            plotly.offline.plot(figure)

    def generate(self, pi, word2idx):
        """
        :param pi: The distribution of starting words.
        :param word2idx: The dictionary mapping words to their respective indices.
        :param end_prob: The probability that the sentence should end. The original predict method was generating
        sentences that were too short, so this is a penalty that makes sentences longer. The smaller this parameter is,
        the smaller the likelihood of the END token being added to a sentence.
        :return: Generates text based on the training data given.
        """
        # convert word2idx -> idx2word
        idx2word = {v: k for k, v in word2idx.items()}
        v = len(pi)

        # generate 4 lines at a time
        n_lines = 0

        # why? because using the START symbol will always yield the same first word!
        x = [np.random.choice(v, p=pi)]
        print(idx2word[x[0]]),

        while n_lines < 4:
            probs = self.session.run(
                self.return_probs_op,
                feed_dict={self.tfX: x}
            )[-1]
            word_idx = np.random.choice(v, p=probs)
            x.append(word_idx)
            if word_idx > 1:
                # it's a real word, not start/end token
                word = idx2word[word_idx]
                print(word),
            elif word_idx == 1:
                # end token
                n_lines += 1
                print('')
                if n_lines < 4:
                    x = [np.random.choice(v, p=pi)]  # reset to start of line
                    print(idx2word[x[0]]),


# learning
sentences, word2idx = get_wikipedia_data(n_files=1, vocab_size_limit=2000, mandatory_words=["king", "man", "queen", "woman"])
rnn = recurrentNeuralNetwork(embedding_space_dim=3, hidden_layer_dimensions=[128, 128, 256], activation_fn=tf.nn.relu,
                             bias_multiplier=100)
rnn.fit(
    x=sentences,
    session=tf.Session(),
    n_epochs=1,
    recurrent_layer_type=gatedRecurrentLayer
)

sentences_embeddings = rnn.getEmbeddings([
    word2idx[word] for word in word2idx.keys()
])
sentences_embeddings = list(sentences_embeddings)

g = go.Scatter3d(
    x=[v[0] for v in sentences_embeddings],
    y=[v[1] for v in sentences_embeddings],
    z=[v[2] for v in sentences_embeddings],
    mode="markers",
    text=list(word2idx.keys())
)

plotly.offline.plot([g])
