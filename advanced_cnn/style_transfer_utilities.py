# general imports
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from copy import deepcopy
from scipy.optimize import fmin_l_bfgs_b

# Keras imports
import keras.backend as K
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Lambda, Conv2D

class ImageHelper(object):
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = np.array(Image.open(self.img_path))
        self.channel_means = np.mean(self.img, axis=(0, 1))
        self.img_transformed = deepcopy(self.img)

    def norm_img(self, kind="pixels"):
        if kind == "pixels":
            self.img_transformed = self.img_transformed / 255
        elif kind == "rgb":
            self.img_transformed = self.img_transformed - self.channel_means

    def permute_channels(self, x=None, kind="RGB2BGR"):
        f, t = kind.split("2")
        map = {char: i for i, char in enumerate(f)}

        # get list that we need to reorder the input channels
        perm_indices = [map[char] for char in t]

        if x is None:
            self.img_transformed = self.img_transformed[:, :, perm_indices]
        else:
            return x[:, :, perm_indices]

    def get_normed_img(self, flatten=False):
        if flatten:
            return self.img_transformed.reshape([-1])
        else:
            return self.img_transformed


class ContentGenerator(object):
    def __init__(self, target_img, n_conv_layers):
        self.model = self.get_vgg(target_img.shape, n_conv_layers)
        self.target_shape = target_img.shape
        self.target_img = K.constant(
            self.model.predict(target_img.reshape([-1, *self.target_shape]))
        )  # our target is the image after we've passed it through the NN. The deeper we pass it through the NN, the
        # more high-level / abstract our image is going to be.

        # create a white-noise image that we're going to use gradient descent on to make its content similar to that
        # of the target image
        self.w = np.random.randn(*self.target_shape).reshape([-1, *self.target_shape])

        # define what the loss is
        self.loss = K.mean(
            K.square(self.target_img - self.model.output)
        )

        # define the gradients. This is important since, contrary to what we usually do, here we're optimizing with
        # respect to the input w
        self.grads = K.gradients(loss=self.loss, variables=self.model.input)

        # we're going to optimize the loss with respect to w using sklearn. To do that, we need a function that returns
        # the pair: (loss, loss')
        self.get_loss_and_grads = K.function(
            inputs=[self.model.input],
            outputs=[self.loss] + self.grads
        )

    def get_loss_and_grads_1d(self, x_flat):
        loss, grads = self.get_loss_and_grads(
            [x_flat.reshape([-1, *self.target_shape])]
        )

        return loss.astype(np.float64), grads.flatten().astype(np.float64)

    def get_vgg(self, input_shape, n_conv_layers):
        if n_conv_layers < 1 or n_conv_layers > 13:
            raise ValueError("The VGG 16 model can be truncated at layers 1 through 13. You requested the model to be "
                             "truncated at layer {0}".format(n_conv_layers))

        # get the pre-trained model
        vgg16 = VGG16(input_shape=input_shape, include_top=False)

        # create a sequential model that is going to contain the requested layers from the VGG16 model and add layers
        i = 0
        model = Sequential()

        for layer in vgg16.layers:
            model.add(layer)

            if layer.__class__ == Conv2D:
                i += 1

                if i > n_conv_layers:
                    break

        return model

    def fit(self, n_steps):
        start_time = time.time()
        for i in range(n_steps):
            self.w, l, _ = fmin_l_bfgs_b(
                func=self.get_loss_and_grads_1d,
                x0=self.w.flatten(),
                maxfun=20
            )

            # clip w
            self.w = np.clip(self.w, -127, 127)

            # trace
            time_elapsed = time.time() - start_time
            print(
                "Iteration %d of %d completed. Loss: %d. Time elapsed: %d minutes and %d seconds."
                % (i+1, n_steps, l, time_elapsed // 60, time_elapsed % 60)
            )


class StyleGenerator(object):
    def __init__(self, target_img):
        self.model = self.get_vgg(target_img.shape)
        self.input_img_shape = target_img.shape

        # now calculate the targets and get the outputs of each of the convolutional layers
        self.targets = [K.constant(y) for y in self.model.predict(target_img.reshape([-1, *self.input_img_shape]))]
        self.outputs = [layer.get_output_at(0) for layer in self.model.layers if layer.name.endswith("conv1")]

        # create the white noise that we need to change in order to get the same style as the input image
        self.w = np.random.randn(*self.input_img_shape).reshape([-1, *self.input_img_shape])

        # define the loss
        self.loss = 0
        for target, output in zip(self.targets, self.outputs):
            self.loss += self.content_loss(target, output)

        # define the gradients
        self.grads = K.gradients(
            loss=self.loss,
            variables=self.model.input
        )

        # define the function that will return loss and gradients
        self.get_loss_and_grads = K.function(
            inputs=[self.model.input],
            outputs=[self.loss] + self.grads
        )

    def get_loss_and_grads_1d(self, x_flat):
        loss, grads = self.get_loss_and_grads(
            [x_flat.reshape([-1, *self.input_img_shape])]
        )

        return loss.astype(np.float64), grads.flatten().astype(np.float64)

    def content_loss(self, x, y):
        return K.mean(
            K.square(self.gram_matrix(x) - self.gram_matrix(y))
        )

    def gram_matrix(self, x):
        # reshape the input to be of dimension [w x h, c]
        x = tf.reshape(x, [-1, tf.shape(x)[-1]])
        x = tf.matmul(x, x, transpose_a=True) / tf.cast(tf.reduce_prod(tf.shape(x)), tf.float32)

        return x

    def get_vgg(self, input_shape):
        # get the pre-trained model
        vgg16 = VGG16(input_shape=input_shape, include_top=False)

        # create a list to hold the layers of the VGG16 model that we want the output of
        outputs = [layer.get_output_at(0) for layer in vgg16.layers if layer.name.endswith("conv1")]

        # create a sequential model that is going to contain the modified VGG16 model
        model = Model(inputs=vgg16.inputs, outputs=outputs)

        return model


    def fit(self, n_steps):
        start_time = time.time()
        for i in range(n_steps):
            self.w, l, _ = fmin_l_bfgs_b(
                func=self.get_loss_and_grads_1d,
                x0=self.w.flatten(),
                maxfun=20
            )

            # clip w
            self.w = np.clip(self.w, -127, 127)

            # trace
            time_elapsed = time.time() - start_time
            print(
                "Iteration %d of %d completed. Loss: %d. Time elapsed: %d minutes and %d seconds."
                % (i + 1, n_steps, l, time_elapsed // 60, time_elapsed % 60)
            )

