import numpy as np
import tensorflow as tf
from dl_layers import hiddenLayer

class VariationalAutoencoder(object):
    def __init__(self, n_features, hidden_layer_sizes, activation_fn=tf.nn.sigmoid):
        self.n_features = n_features
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_fn = activation_fn

        # initialize the encoder layers
        i = 0
        n_in = self.n_features
        self.encoder_layers = []

        for n_out in self.hidden_layer_sizes[:-1]:
            self.encoder_layers.append(hiddenLayer(n_in, n_out, i, self.activation_fn))
            n_in = n_out
            i += 1

        # initialize the latent layer
        self.n_latent = self.hidden_layer_sizes[-1] * 2  # one vector for mu, one for sigma
        self.latent_layer = hiddenLayer(n_in, self.n_latent, i, self.activation_fn)
        i += 1

        # initialize the decoder layers
        n_in = n_latent
        self.decoder_layers = []

        for n_out in reversed(self.hidden_layer_sizes[:-1]):
            self.decoder_layers.append(hiddenLayer(n_in, n_out, i, self.activation_fn))
            n_in = n_out
            i += 1

        # initialize the final layer
        self.final_layer = hiddenLayer(n_in, self.n_features, i, self.activation_fn)

        # initialize the placeholder
        self.tfX = tf.placeholder(dtype=np.float32, shape=(None, n_features))

        # save the normal and Bernoulli distribution for convenience
        self.normal = tf.contrib.distributions.Normal
        self.bernoulli = tf.contrib.distributions.Bernoulli

    def set_parameters(self):
        z = self.tfX

        for layer in self.encoder_layers:
            z = layer.forward(z)

        self.mu = z[:, :self.n_latent]
        self.sigma = tf.nn.softplus(z[:, self.n_latent:]) + 10e-6  # a buffer to keep sigma > 0

    def set_latent_variable(self):
        standard_normal = self.normal(
            np.zeros(shape=self.n_latent, dtype=np.float32),
            np.ones(shape=self.n_latent, dtype=np.float32)
        )

        # create a Normal(mu, sigma2) random variable
        e = standard_normal.sample(shape=self.n_latent)
        self.z = e * self.sigma + self.mu


