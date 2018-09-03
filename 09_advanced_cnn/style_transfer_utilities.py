import numpy as np
from PIL import Image
from copy import deepcopy

class ImageHelper(object):
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = np.array(Image.open(self.img_path))
        self.channel_means = np.mean(self.img, axis=2)
        self.img_normed = None  # here I'm going to keep the normed image

    def norm_img(self, kind="pixels"):
        x = deepcopy(self.img)

        if kind == "pixels":
            x = x / 255
        elif kind == "rgb":
            x = x - self.channel_means

        self.img_normed = x

    def get_normed_img(self, flatten=False):
        if flatten:
            return self.img_normed.reshape([-1])
        else:
            return self.img_normed


