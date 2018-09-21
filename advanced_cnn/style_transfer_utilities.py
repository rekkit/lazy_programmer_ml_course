# general imports
import time
import numpy as np
from PIL import Image
from copy import deepcopy
from scipy.optimize import fmin_l_bfgs_b

# Keras imports
import keras.backend as K


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
    def __init__(self, model, target_img):
        self.model = model
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

        # define the gradients. This is important since, contrary to what we usually do, here we're optimizing wrt the
        # input w
        self.grads = K.gradients(loss=self.loss, variables=self.model.input)

        # we're going to optimize the loss wrt w using sklearn. To do that, we need a function that returns the pair:
        # (loss, loss')
        self.get_loss_and_grads = K.function(
            inputs=[self.model.input],
            outputs=[self.loss] + self.grads
        )

    def get_loss_and_grads_1d(self, x_flat):
        loss, grads = self.get_loss_and_grads(
            [x_flat.reshape([-1, *self.target_shape])]
        )

        return loss.astype(np.float64), grads.reshape([-1]).astype(np.float64)

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
            