import numpy as np
from glob import glob
import tensorflow as tf
from sklearn.utils import shuffle
from skimage.io import imread, imshow
from sklearn.model_selection import train_test_split
from rekkml.layers import ActivationLayer, BatchNormalizationLayer, ConvolutionalLayer, ConvolutionalTransposeLayer, \
    HiddenLayer, ReshapeLayer, ResNetCell, MaxPoolLayer, FlattenLayer


class Generator:
    def __init__(self):
        self.params = None
        self.layers = [
            HiddenLayer(n_out=16384, save_params=True),
            ReshapeLayer(output_shape=(None, 4, 4, 1024), save_params=True),
            ConvolutionalTransposeLayer(
                filter_h=5, filter_w=5, maps_out=512, stride_horizontal=2, stride_vertical=2, padding="SAME",
                output_shape=(None, 8, 8, 512), save_params=True
            ),
            BatchNormalizationLayer(save_params=True),
            ActivationLayer(activation_fn=tf.nn.relu, save_params=True),
            ConvolutionalTransposeLayer(
                filter_h=5, filter_w=5, maps_out=256, stride_horizontal=2, stride_vertical=2, padding="SAME",
                output_shape=(None, 16, 16, 256), save_params=True
            ),
            BatchNormalizationLayer(save_params=True),
            ActivationLayer(activation_fn=tf.nn.relu, save_params=True),
            ConvolutionalTransposeLayer(
                filter_h=5, filter_w=5, maps_out=128, stride_horizontal=2, stride_vertical=2, padding="SAME",
                output_shape=(None, 32, 32, 128), save_params=True
            ),
            BatchNormalizationLayer(save_params=True),
            ActivationLayer(activation_fn=tf.nn.relu, save_params=True),
            ConvolutionalTransposeLayer(
                filter_h=5, filter_w=5, maps_out=3, stride_horizontal=2, stride_vertical=2, padding="SAME",
                output_shape=(None, 64, 64, 3), save_params=True
            ),
            ActivationLayer(activation_fn=tf.nn.tanh, save_params=True)
        ]

    def initialize_weights(self, input_shape):
        self.layers[0].append_input_shape(input_shape)
        for i in range(1, len(self.layers)):
            self.layers[i].append_input_shape(self.layers[i-1].output_shape)

        for i, layer in enumerate(self.layers):
            layer.initialize_weights(i)

    def initialize_parameters(self):
        self.params = []
        for layer in self.layers:
            for param in layer.get_params():
                self.params.append(param)

    def get_params(self):
        if self.params is None:
            raise ValueError(
                "There are no parameters to fetch. Make sure you have initialized the weights and parameters."
            )
        else:
            return self.params

    def forward(self, x, is_training):
        z = self.layers[0].forward(x)
        for i in range(1, len(self.layers)):
            if isinstance(self.layers[i], BatchNormalizationLayer):
                z = self.layers[i].forward(z, is_training)
            else:
                z = self.layers[i].forward(z)

        return z


class Discriminator:
    def __init__(self):
        self.params = None
        self.layers = [
            ConvolutionalLayer(filter_h=7, filter_w=7, maps_out=32, save_params=True),
            ActivationLayer(activation_fn=tf.nn.leaky_relu, save_params=True),
            ResNetCell(filter_h=3, filter_w=3, maps_out=32, save_params=True),
            BatchNormalizationLayer(save_params=True),
            ActivationLayer(activation_fn=tf.nn.leaky_relu, save_params=True),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=64, stride_horizontal=2, stride_vertical=2, save_params=True),
            BatchNormalizationLayer(save_params=True),
            ActivationLayer(activation_fn=tf.nn.leaky_relu, save_params=True),
            ResNetCell(filter_h=3, filter_w=3, maps_out=64, save_params=True),
            BatchNormalizationLayer(save_params=True),
            ActivationLayer(activation_fn=tf.nn.leaky_relu, save_params=True),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=128, stride_horizontal=2, stride_vertical=2, save_params=True),
            BatchNormalizationLayer(save_params=True),
            ActivationLayer(activation_fn=tf.nn.leaky_relu, save_params=True),
            ResNetCell(filter_h=3, filter_w=3, maps_out=128, save_params=True),
            BatchNormalizationLayer(save_params=True),
            ActivationLayer(activation_fn=tf.nn.leaky_relu, save_params=True),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=64, stride_horizontal=2, stride_vertical=2, save_params=True),
            BatchNormalizationLayer(save_params=True),
            ActivationLayer(activation_fn=tf.nn.leaky_relu, save_params=True),
            ConvolutionalLayer(filter_h=1, filter_w=1, maps_out=32, save_params=True),
            BatchNormalizationLayer(save_params=True),
            ActivationLayer(activation_fn=tf.nn.leaky_relu, save_params=True),
            ConvolutionalLayer(filter_h=1, filter_w=1, maps_out=16, save_params=True),
            BatchNormalizationLayer(save_params=True),
            ActivationLayer(activation_fn=tf.nn.leaky_relu, save_params=True),
            MaxPoolLayer(window_h=6, window_w=6, stride_horizontal=1, stride_vertical=1, save_params=True),
            ConvolutionalLayer(filter_h=1, filter_w=1, maps_out=2, save_params=True),
            FlattenLayer(save_params=True),
            ActivationLayer(activation_fn=tf.nn.sigmoid, save_params=True)
        ]

    def initialize_weights(self, input_shape):
        self.layers[0].append_input_shape(input_shape)
        for i in range(1, len(self.layers)):
            self.layers[i].append_input_shape(self.layers[i-1].output_shape)

        for i, layer in enumerate(self.layers):
            layer.initialize_weights(i)
            print(
                "Layer:", str(layer.__class__).split('.')[-1].replace("'>", '').strip(' ') + ".",
                "Input shape:", layer.input_shape,
                "Output shape:", layer.output_shape
            )

    def initialize_parameters(self):
        self.params = []
        for layer in self.layers:
            for param in layer.get_params():
                self.params.append(param)

    def get_params(self):
        if self.params is None:
            raise ValueError(
                "There are no parameters to fetch. Make sure you have initialized the weights and parameters."
            )
        else:
            return self.params

    def forward(self, x, is_training):
        z = self.layers[0].forward(x)
        for i in range(1, len(self.layers)):
            if isinstance(self.layers[i], BatchNormalizationLayer) or isinstance(self.layers[i], ResNetCell):
                z = self.layers[i].forward(z, is_training)
            else:
                z = self.layers[i].forward(z)

        return z


class DCGAN:
    def __init__(self):
        # placeholders, session and optimizer
        self.tfX = None
        self.tfZ = None
        self.session = None
        self.optimizer = None

        # operations
        self.discriminator_train_op = None
        self.generator_train_op = None

        # loss
        self.discriminator_training_loss = None
        self.generator_training_loss = None
        self.discriminator_test_loss = None
        self.generator_test_loss = None

        # cost
        self.discriminator_training_costs = []
        self.discriminator_test_costs = []
        self.generator_costs = []

        # instantiate a discriminator and a generator
        self.discriminator = Discriminator()
        self.generator = Generator()

    def initialize_placeholders(self, input_shape):
        self.tfX = tf.placeholder(dtype=tf.float32, shape=input_shape)
        self.tfZ = tf.placeholder(dtype=tf.float32, shape=(None, 100))

    def initialize_discriminator(self, input_shape):
        self.discriminator = Discriminator()
        self.discriminator.initialize_weights(input_shape)
        self.discriminator.initialize_parameters()

    def initialize_generator(self, input_shape):
        self.generator = Generator()
        self.generator.initialize_weights(input_shape)
        self.generator.initialize_parameters()

    def initialize_discriminator_loss(self):
        self.discriminator_training_loss = \
            - tf.reduce_mean(
                tf.log(
                    self.discriminator.forward(self.tfX, is_training=True)
                )
            ) \
            - tf.reduce_mean(
                tf.log(
                    1 - self.discriminator.forward(
                        self.generator.forward(self.tfZ, is_training=True),
                        is_training=True
                    )
                )
            )

        self.discriminator_test_loss = \
            - tf.reduce_mean(
                tf.log(
                    self.discriminator.forward(self.tfX, is_training=False)
                )
            ) \
            - tf.reduce_mean(
                tf.log(
                    1 - self.discriminator.forward(
                        self.generator.forward(self.tfZ, is_training=False),
                        is_training=False
                    )
                )
            )

    def initialize_generator_loss(self):
        self.generator_training_loss = - tf.reduce_mean(
            tf.log(
                self.discriminator.forward(
                    self.generator.forward(self.tfZ, is_training=True),
                    is_training=True
                )
            )
        )

        self.generator_test_loss = - tf.reduce_mean(
            tf.log(
                self.discriminator.forward(
                    self.generator.forward(self.tfZ, is_training=False),
                    is_training=False
                )
            )
        )

    def initialize_discriminator_training_op(self, optimizer):
        self.discriminator_train_op = optimizer.minimize(
            loss=self.discriminator_training_loss,
            var_list=self.discriminator.get_params()
        )

    def initialize_generator_training_op(self, optimizer):
        self.generator_train_op = optimizer.minimize(
            loss=self.generator_training_loss,
            var_list=self.generator.get_params()
        )

    def set_session(self):
        self.session = tf.Session()

    def fit(self, x, batch_size, n_epochs, generator_discriminator_train_ratio=2, optimizer=None, print_step=20):
        self.optimizer = optimizer if optimizer is not None else tf.train.AdamOptimizer(0.0002)
        input_shape = (None, *x.shape[1:])

        # create a session and initialize the placeholders
        self.set_session()
        self.initialize_placeholders(input_shape)

        # initialize the the discriminator and the generator, as well as their losses
        self.initialize_discriminator(input_shape)
        self.initialize_generator((None, 100))
        self.initialize_discriminator_loss()
        self.initialize_generator_loss()
        self.initialize_discriminator_training_op(self.optimizer)
        self.initialize_generator_training_op(self.optimizer)

        # train / test split
        x_train, x_test = train_test_split(x, train_size=0.2)

        # get the number of steps we need to complete a batch
        n_steps = x_train.shape[0] // batch_size

        for i in range(n_epochs):
            x_train = shuffle(x_train)

            for j in range(n_steps):
                x_batch = x_train[j*batch_size: (j+1)*batch_size]
                z_batch = np.random.uniform(size=(batch_size, 100))

                for k in range(generator_discriminator_train_ratio):
                    self.session.run(
                        self.generator_train_op,
                        feed_dict={self.tfZ: z_batch}
                    )

                self.session.run(
                    self.discriminator_train_op,
                    feed_dict={self.tfX: x_batch, self.tfZ: z_batch}
                )

                if j > 0 and j % print_step == 0:
                    self.discriminator_training_costs.append(
                        self.session.run(
                            self.discriminator_test_costs,
                            feed_dict={self.tfZ: z_batch, self.tfX: x_batch}
                        )
                    )

                    self.discriminator_test_costs.append(
                        self.session.run(
                            self.discriminator_test_costs,
                            feed_dict={self.tfZ: z_batch, self.tfX: x_test}
                        )
                    )

                    self.generator_costs.append(
                        self.session.run(
                            self.generator_test_loss,
                            feed_dict={self.tfZ: z_batch}
                        )
                    )

                    print(
                        "Epoch:", i,
                        "Step:", j,
                        "Discriminator training cost:", self.discriminator_training_costs[-1],
                        "Generator training cost:", self.generator_costs[-1],
                        "Discriminator test cost:", self.discriminator_test_costs[-1]
                    )


#import os
#os.chdir("..")

img_paths = glob("./large_files/celebrity/img_align_celeba/*")
x = np.array(
    [imread(path) for path in img_paths[:10000]]
)

model = DCGAN()
model.fit(x=x, batch_size=4, n_epochs=10)
